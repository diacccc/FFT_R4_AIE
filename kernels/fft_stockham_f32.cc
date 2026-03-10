//===- fft_stockham_f32.cc --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

#ifndef FFT_SIZE
#define FFT_SIZE 256
#endif

// Radix-4 FFT requires N to be a power of 4
static_assert((FFT_SIZE & (FFT_SIZE - 1)) == 0, "FFT_SIZE must be a power of 2");
// Additional runtime check will verify it's a power of 4

// Zero initialization for output buffer
template <typename T_out, unsigned size>
static inline void zero_float(T_out *__restrict c) {
  for (unsigned i = 0; i < size; i++) {
    c[i] = 0.0f;
  }
}

template <typename T_out, unsigned size>
static inline void zero_vectorized(T_out *__restrict c) {
  constexpr unsigned r = 512 / (sizeof(T_out) * 8); // 512 bit store units for AIE2P
  static_assert(size % r == 0);
  const aie::vector<T_out, r> zeros = aie::zeros<T_out, r>();
  const T_out *__restrict c_end = c + size;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// Compute log2 of N at compile time
constexpr unsigned log2_const(unsigned n) {
  return (n <= 1) ? 0 : 1 + log2_const(n / 2);
}

// Compute log4 of N at compile time
constexpr unsigned log4_const(unsigned n) {
  return (n <= 1) ? 0 : 1 + log4_const(n / 4);
}

// Split a float into 4 bfloat16 slices
// Using error-free transformation (EFT)
// f = f0 + f1 + f2 + f3 where f0, f1, f2, f3 are representable in bfloat16
static inline void split_float_to_bf16(float f, bfloat16 splits[4]) {
  float remainder = f;
  
  for (int i = 0; i < 4; i++) {
    bfloat16 bf = (bfloat16)remainder;
    splits[i] = bf;
    remainder = remainder - (float)bf;
  }
}

// Complex multiplication using GEMM-based method with float splitting
// Computes (a + bj) * (c + dj) where:
//   - a, b are floats (split on-the-fly)
//   - c, d are pre-split into bf16 arrays
// Returns result in (real, imag) as floats
//
// Ozaki scheme: When a = a0+a1+a2+a3 and c = c0+c1+c2+c3,
// then a*c = sum_i sum_j (a_i * c_j)
static inline void complex_mul_gemm(float a, float b, 
                                     const bfloat16 c_splits[4], 
                                     const bfloat16 d_splits[4],
                                     float &result_real, float &result_imag) {
  // Split input values (a, b) into bf16 slices
  bfloat16 a_splits[4], b_splits[4];
  split_float_to_bf16(a, a_splits);
  split_float_to_bf16(b, b_splits);
  
  // Compute (a + bj) * (c + dj) = (ac - bd) + (ad + bc)j
  // using Ozaki scheme: compute all pairwise products
  
  // Real part: ac - bd
  float ac = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ac += (float)a_splits[i] * (float)c_splits[j];
    }
  }
  
  float bd = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      bd += (float)b_splits[i] * (float)d_splits[j];
    }
  }
  
  result_real = ac - bd;
  
  // Imaginary part: ad + bc
  float ad = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ad += (float)a_splits[i] * (float)d_splits[j];
    }
  }
  
  float bc = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      bc += (float)b_splits[i] * (float)c_splits[j];
    }
  }
  
  result_imag = ad + bc;
}

// Radix-4 Stockham FFT Algorithm
// ============================================================================
// Radix-4 DIT Stockham auto-sort FFT: no bit-reversal needed, naturally sorted output
// Each stage processes 4 points at a time using radix-4 butterfly
//
// Radix-4 DIT butterfly:
//   a = x[q + s*(p + 0*m)]
//   b = x[q + s*(p + 1*m)] * W_N^(q*m)
//   c = x[q + s*(p + 2*m)] * W_N^(q*2*m)
//   d = x[q + s*(p + 3*m)] * W_N^(q*3*m)
//
//   y[q + s*(4*p + 0)] = a + b + c + d
//   y[q + s*(4*p + 1)] = a - j*b - c + j*d
//   y[q + s*(4*p + 2)] = a - b + c - d
//   y[q + s*(4*p + 3)] = a + j*b - c - j*d
//
// Stage progression: n -> n/4, s -> 4*s

template <unsigned N>
static inline void fft_stockham_gemm(float *__restrict x,
                                      const bfloat16 *__restrict twiddle,
                                      float *__restrict y) {
  constexpr unsigned LOG4N = log4_const(N);
  unsigned stage_twiddle_base = 0;

  for (unsigned stage = 0; stage < LOG4N; ++stage) {
    float *src = (stage % 2 == 0) ? x : y;
    float *dst = (stage % 2 == 0) ? y : x;

    const unsigned n = N >> (2 * stage);  // n = N / 4^stage
    const unsigned s = 1u << (2 * stage); // s = 4^stage
    const unsigned m = n >> 2;             // m = n / 4
    const bfloat16 *stage_twiddle = &twiddle[stage_twiddle_base];

    // Process all butterflies
    for (unsigned q = 0; q < s; ++q) {
      // Load twiddle factors for this q
      // W^(q*m), W^(2*q*m), W^(3*q*m) - each stored as 8 bf16 values (4 real + 4 imag splits)
      bfloat16 twiddle1_real_splits[4], twiddle1_imag_splits[4];
      bfloat16 twiddle2_real_splits[4], twiddle2_imag_splits[4];
      bfloat16 twiddle3_real_splits[4], twiddle3_imag_splits[4];
      
      for (unsigned split = 0; split < 4; ++split) {
        twiddle1_real_splits[split] = stage_twiddle[split * s + q];
        twiddle1_imag_splits[split] = stage_twiddle[(4 + split) * s + q];
        
        twiddle2_real_splits[split] = stage_twiddle[(8 + split) * s + q];
        twiddle2_imag_splits[split] = stage_twiddle[(12 + split) * s + q];
        
        twiddle3_real_splits[split] = stage_twiddle[(16 + split) * s + q];
        twiddle3_imag_splits[split] = stage_twiddle[(20 + split) * s + q];
      }
      
      const bool is_unity_twiddle = (q == 0);

      for (unsigned p = 0; p < m; ++p) {
        // Load 4 input points
        const unsigned idx_a = q + s * (p + 0 * m);
        const unsigned idx_b = q + s * (p + 1 * m);
        const unsigned idx_c = q + s * (p + 2 * m);
        const unsigned idx_d = q + s * (p + 3 * m);

        const float a_real = src[2 * idx_a];
        const float a_imag = src[2 * idx_a + 1];
        const float b_real = src[2 * idx_b];
        const float b_imag = src[2 * idx_b + 1];
        const float c_real = src[2 * idx_c];
        const float c_imag = src[2 * idx_c + 1];
        const float d_real = src[2 * idx_d];
        const float d_imag = src[2 * idx_d + 1];

        // Apply twiddle factors
        float tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
        
        if (is_unity_twiddle) {
          tb_real = b_real;
          tb_imag = b_imag;
          tc_real = c_real;
          tc_imag = c_imag;
          td_real = d_real;
          td_imag = d_imag;
        } else {
          complex_mul_gemm(b_real, b_imag, twiddle1_real_splits,
                           twiddle1_imag_splits, tb_real, tb_imag);
          complex_mul_gemm(c_real, c_imag, twiddle2_real_splits,
                           twiddle2_imag_splits, tc_real, tc_imag);
          complex_mul_gemm(d_real, d_imag, twiddle3_real_splits,
                           twiddle3_imag_splits, td_real, td_imag);
        }

        // Radix-4 butterfly computation
        // y[0] = a + b + c + d
        const float y0_real = a_real + tb_real + tc_real + td_real;
        const float y0_imag = a_imag + tb_imag + tc_imag + td_imag;

        // y[1] = a - j*b - c + j*d
        // -j*(tb_real + tb_imag*j) = tb_imag - tb_real*j
        // j*(td_real + td_imag*j) = -td_imag + td_real*j
        const float y1_real = a_real + tb_imag - tc_real - td_imag;
        const float y1_imag = a_imag - tb_real - tc_imag + td_real;

        // y[2] = a - b + c - d
        const float y2_real = a_real - tb_real + tc_real - td_real;
        const float y2_imag = a_imag - tb_imag + tc_imag - td_imag;

        // y[3] = a + j*b - c - j*d
        // j*(tb_real + tb_imag*j) = -tb_imag + tb_real*j
        // -j*(td_real + td_imag*j) = td_imag - td_real*j
        const float y3_real = a_real - tb_imag - tc_real + td_imag;
        const float y3_imag = a_imag + tb_real - tc_imag - td_real;

        // Store outputs
        const unsigned out_idx0 = q + s * (4 * p + 0);
        const unsigned out_idx1 = q + s * (4 * p + 1);
        const unsigned out_idx2 = q + s * (4 * p + 2);
        const unsigned out_idx3 = q + s * (4 * p + 3);

        dst[2 * out_idx0] = y0_real;
        dst[2 * out_idx0 + 1] = y0_imag;
        dst[2 * out_idx1] = y1_real;
        dst[2 * out_idx1 + 1] = y1_imag;
        dst[2 * out_idx2] = y2_real;
        dst[2 * out_idx2 + 1] = y2_imag;
        dst[2 * out_idx3] = y3_real;
        dst[2 * out_idx3 + 1] = y3_imag;
      }
    }

    // Each stage uses 3 twiddle factors per q: W^(q*m), W^(2*q*m), W^(3*q*m)
    // Each complex twiddle is 8 bf16 values (4 real splits + 4 imag splits)
    stage_twiddle_base += 24 * s; // 3 twiddles * 8 bf16 values per twiddle
  }

  // If number of stages is even, final data sits in x; copy to y.
  if ((LOG4N & 1u) == 0) {
    for (unsigned i = 0; i < N * 2; ++i) {
      y[i] = x[i];
    }
  }
}

// Wrapper functions
extern "C" {

void fft_stockham_f32(float *input, const bfloat16 *twiddle, 
                      float *output) {
  // Perform FFT using Stockham algorithm with GEMM-based complex multiplication
  fft_stockham_gemm<FFT_SIZE>(input, twiddle, output);
}

void zero_f32(float *c) {
  zero_vectorized<float, FFT_SIZE * 2>(c);
}

} // extern "C"
