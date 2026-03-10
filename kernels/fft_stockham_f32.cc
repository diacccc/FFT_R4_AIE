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

// Emulated bf16 multiply between two split-4 values:
// (a0+a1+a2+a3) * (b0+b1+b2+b3) = sum_i sum_j ai*bj
static inline float mul_split4(const bfloat16 a_splits[4],
                               const bfloat16 b_splits[4]) {
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      acc += (float)a_splits[i] * (float)b_splits[j];
    }
  }
  return acc;
}

// Complex multiplication using GEMM-based method with float splitting
// Computes (a + bj) * (c + dj) where:
//   - a, b are floats (split on-the-fly)
//   - c, d are pre-split into bf16 arrays
// Returns result in (real, imag) as floats
//
// Ozaki scheme: When a = a0+a1+a2+a3 and c = c0+c1+c2+c3,
// then a*c = sum_i sum_j (a_i * c_j)
static __attribute__((noinline)) void complex_mul_gemm(
  float a, float b,
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
  const float ac = mul_split4(a_splits, c_splits);
  const float bd = mul_split4(b_splits, d_splits);
  result_real = ac - bd;
  
  // Imaginary part: ad + bc
  const float ad = mul_split4(a_splits, d_splits);
  const float bc = mul_split4(b_splits, c_splits);
  result_imag = ad + bc;
}

// BF16-split GEMM-style radix-4 butterfly.
// Input vector order is:
//   [a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i]
// and output vector order is:
//   [y0_r, y0_i, y1_r, y1_i, y2_r, y2_i, y3_r, y3_i]
//
// The fixed 8x8 real matrix is the real-expanded W4 butterfly matrix.
// We split both input values and matrix coefficients into 4 bf16 slices,
// then accumulate sum_i sum_j products for every matrix multiply term.
static __attribute__((noinline)) void butterfly_gemm_bf16(
  float a_real, float a_imag,
  float b_real, float b_imag,
  float c_real, float c_imag,
  float d_real, float d_imag,
  float &y0_real, float &y0_imag,
  float &y1_real, float &y1_imag,
  float &y2_real, float &y2_imag,
  float &y3_real, float &y3_imag) {
  static const float butterfly_coeff[8][8] = {
      {1, 0, 1, 0, 1, 0, 1, 0},
      {0, 1, 0, 1, 0, 1, 0, 1},
      {1, 0, 0, 1, -1, 0, 0, -1},
      {0, 1, -1, 0, 0, -1, 1, 0},
      {1, 0, -1, 0, 1, 0, -1, 0},
      {0, 1, 0, -1, 0, 1, 0, -1},
      {1, 0, 0, -1, -1, 0, 0, 1},
      {0, 1, 1, 0, 0, -1, -1, 0},
  };

  static bool coeff_ready = false;
  static bfloat16 butterfly_coeff_splits[8][8][4];
  if (!coeff_ready) {
    for (unsigned row = 0; row < 8; ++row) {
      for (unsigned col = 0; col < 8; ++col) {
        split_float_to_bf16(butterfly_coeff[row][col],
                            butterfly_coeff_splits[row][col]);
      }
    }
    coeff_ready = true;
  }

  float x_vals[8] = {a_real, a_imag, b_real, b_imag,
                     c_real, c_imag, d_real, d_imag};
  bfloat16 x_splits[8][4];
  for (unsigned k = 0; k < 8; ++k) {
    split_float_to_bf16(x_vals[k], x_splits[k]);
  }

  float y_vals[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (unsigned row = 0; row < 8; ++row) {
    float acc = 0.0f;
    for (unsigned col = 0; col < 8; ++col) {
      acc += mul_split4(x_splits[col], butterfly_coeff_splits[row][col]);
    }
    y_vals[row] = acc;
  }

  y0_real = y_vals[0];
  y0_imag = y_vals[1];
  y1_real = y_vals[2];
  y1_imag = y_vals[3];
  y2_real = y_vals[4];
  y2_imag = y_vals[5];
  y3_real = y_vals[6];
  y3_imag = y_vals[7];
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
      // Load twiddle factors for this q from contiguous 24-bf16 block:
      // [tw1(r0..r3,i0..i3), tw2(...), tw3(...)]
      bfloat16 twiddle1_real_splits[4], twiddle1_imag_splits[4];
      bfloat16 twiddle2_real_splits[4], twiddle2_imag_splits[4];
      bfloat16 twiddle3_real_splits[4], twiddle3_imag_splits[4];

      const bfloat16 *twq = stage_twiddle + 24 * q;
      twiddle1_real_splits[0] = twq[0];
      twiddle1_real_splits[1] = twq[1];
      twiddle1_real_splits[2] = twq[2];
      twiddle1_real_splits[3] = twq[3];
      twiddle1_imag_splits[0] = twq[4];
      twiddle1_imag_splits[1] = twq[5];
      twiddle1_imag_splits[2] = twq[6];
      twiddle1_imag_splits[3] = twq[7];

      twiddle2_real_splits[0] = twq[8];
      twiddle2_real_splits[1] = twq[9];
      twiddle2_real_splits[2] = twq[10];
      twiddle2_real_splits[3] = twq[11];
      twiddle2_imag_splits[0] = twq[12];
      twiddle2_imag_splits[1] = twq[13];
      twiddle2_imag_splits[2] = twq[14];
      twiddle2_imag_splits[3] = twq[15];

      twiddle3_real_splits[0] = twq[16];
      twiddle3_real_splits[1] = twq[17];
      twiddle3_real_splits[2] = twq[18];
      twiddle3_real_splits[3] = twq[19];
      twiddle3_imag_splits[0] = twq[20];
      twiddle3_imag_splits[1] = twq[21];
      twiddle3_imag_splits[2] = twq[22];
      twiddle3_imag_splits[3] = twq[23];
      
      const bool is_unity_twiddle = (q == 0);

      // Tiled two-phase processing so each butterfly input vector is contiguous
      // in local memory: [a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i].
      constexpr unsigned P_TILE = 4;
      float butterfly_in[P_TILE][8];

      for (unsigned p0 = 0; p0 < m; p0 += P_TILE) {
        const unsigned p_lim = ((p0 + P_TILE) < m) ? (p0 + P_TILE) : m;

        // Stage A: gather + twiddle multiply, then pack contiguous butterfly vectors.
        for (unsigned p = p0; p < p_lim; ++p) {
          const unsigned t = p - p0;
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

          butterfly_in[t][0] = a_real;
          butterfly_in[t][1] = a_imag;
          butterfly_in[t][2] = tb_real;
          butterfly_in[t][3] = tb_imag;
          butterfly_in[t][4] = tc_real;
          butterfly_in[t][5] = tc_imag;
          butterfly_in[t][6] = td_real;
          butterfly_in[t][7] = td_imag;
        }

        // Stage B: contiguous butterfly matmul and scatter.
        for (unsigned p = p0; p < p_lim; ++p) {
          const unsigned t = p - p0;

          float y0_real, y0_imag, y1_real, y1_imag;
          float y2_real, y2_imag, y3_real, y3_imag;
          butterfly_gemm_bf16(butterfly_in[t][0], butterfly_in[t][1],
                              butterfly_in[t][2], butterfly_in[t][3],
                              butterfly_in[t][4], butterfly_in[t][5],
                              butterfly_in[t][6], butterfly_in[t][7],
                              y0_real, y0_imag,
                              y1_real, y1_imag,
                              y2_real, y2_imag,
                              y3_real, y3_imag);

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
