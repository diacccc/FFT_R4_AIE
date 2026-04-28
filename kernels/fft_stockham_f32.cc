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

#define __AIEARCH__ 21

#include <aie_api/aie.hpp>

#ifndef FFT_SIZE
#define FFT_SIZE 256
#endif

// Radix-4 FFT requires N to be a power of 4

#define PROFILING 0


static volatile unsigned long long g_fft_stockham_cycles = 0;
static volatile unsigned long long g_fft_elemwise_cycles = 0;
static volatile unsigned long long g_fft_butterfly_cycles = 0;

constexpr unsigned kSplitCount = 4;
constexpr unsigned kTwiddlesPerQ = 3;
constexpr unsigned kComplexSplitWidth = 8; // [r0,i0,r1,i1,r2,i2,r3,i3]
constexpr unsigned kTwiddleBlockPerQ = kTwiddlesPerQ * kComplexSplitWidth;

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

// Compute log4 of N at compile time
constexpr unsigned log4_const(unsigned n) {
  return (n <= 1) ? 0 : 1 + log4_const(n / 4);
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
  unsigned long long elemwise_cycles = 0;
  unsigned long long butterfly_cycles = 0;

  // Stage 0 has s=1 and q=0 only, so no twiddle factors are needed.
// Stage 0 in contiguous-butterfly layout:
// input butterflies:  [x[4p+0], x[4p+1], x[4p+2], x[4p+3]]
// output layout:      y[p + r*m], r=0..3, m=N/4
    constexpr unsigned Q_TILE = 4;
    constexpr unsigned stage0 = 0;
    constexpr unsigned M_FLAT = Q_TILE * 8;
    
    alignas(aie::vector_decl_align) static bfloat16 coeff_buf[64] = {
        1,  0,  1,  0,  1,  0,  1,  0,
        0,  1,  0,  1,  0,  1,  0,  1,
        1,  0,  0, -1, -1,  0,  0,  1,
        0,  1,  1,  0,  0, -1, -1,  0,
        1,  0, -1,  0,  1,  0, -1,  0,
        0,  1,  0, -1,  0,  1,  0, -1,
        1,  0,  0,  1, -1,  0,  0, -1,
        0,  1, -1,  0,  0, -1,  1,  0,
    };
    aie::vector<bfloat16, 64> coeff_vecs;
    coeff_vecs = aie::load_v<64>(coeff_buf);
    const unsigned n = N >> (2 * stage0);  // n = N
    const unsigned m = (n >> 2)/Q_TILE; 
    const unsigned stride_out = N >> 2;   
    // Stage 1 
    for (unsigned p0 = 0; p0 < m; p0++) chess_prepare_for_pipelining chess_loop_range(m, m){
      aie::accum<accfloat, M_FLAT> in_vecs;
      in_vecs = aie::load_v<M_FLAT>(x + 2 * p0 * Q_TILE * 4);
      aie::accum<accfloat, M_FLAT> tmp_splits;

      using MMUL = aie::mmul<Q_TILE, 8, 8, bfloat16, bfloat16, accfloat>;
      MMUL OUT;
      aie::vector<bfloat16, M_FLAT> in_splits = in_vecs.template to_vector<bfloat16>();
      OUT.mul(in_splits, coeff_vecs);
      for (unsigned k = 1; k < kSplitCount; ++k){
        tmp_splits.from_vector(in_splits);
        in_vecs = aie::sub(in_vecs, tmp_splits);
        in_splits = in_vecs.template to_vector<bfloat16>();
        OUT.mac(in_splits, coeff_vecs);
      }
      aie::vector<cint32, M_FLAT/2> out_cint32 = (OUT.template to_vector<float>()).cast_to<cint32>();
        aie::store_v(y + 2 * p0 * Q_TILE * 4, 
          aie::transpose(out_cint32, Q_TILE, 4).cast_to<float>());
    }
    // Stage 2
alignas(aie::vector_decl_align) static const bfloat16 twiddle_T2_splits[4][32] = {
    // split 0
        // split 0
        {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.92578125f, -0.3828125f, 0.70703125f, -0.70703125f, 0.3828125f, -0.92578125f,
            1.0f, 0.0f, 0.70703125f, -0.70703125f, 0.0f, -1.0f, -0.70703125f, -0.70703125f,
            1.0f, 0.0f, 0.3828125f, -0.92578125f, -0.70703125f, -0.70703125f, -0.92578125f, 0.3828125f,
        },
        // split 1
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.00189971923828125f, 0.00012874603271484375f, 7.534027099609375e-05f, -7.534027099609375e-05f, -0.00012874603271484375f, 0.00189971923828125f,
            0.0f, 0.0f, 7.534027099609375e-05f, -7.534027099609375e-05f, 0.0f, 0.0f, -7.534027099609375e-05f, -7.534027099609375e-05f,
            0.0f, 0.0f, -0.00012874603271484375f, 0.00189971923828125f, -7.534027099609375e-05f, -7.534027099609375e-05f, 0.00189971923828125f, -0.00012874603271484375f,
        },
        // split 2
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -2.0265579223632812e-06f, 3.2782554626464844e-07f, 1.7881393432617188e-07f, -1.7881393432617188e-07f, -3.2782554626464844e-07f, 2.0265579223632812e-06f,
            0.0f, 0.0f, 1.7881393432617188e-07f, -1.7881393432617188e-07f, 0.0f, 0.0f, -1.7881393432617188e-07f, -1.7881393432617188e-07f,
            0.0f, 0.0f, -3.2782554626464844e-07f, 2.0265579223632812e-06f, -1.7881393432617188e-07f, -1.7881393432617188e-07f, 2.0265579223632812e-06f, -3.2782554626464844e-07f,
        },
        // split 3
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        },
};
    for (unsigned p1 = 0; p1 < m; p1++) chess_prepare_for_pipelining chess_loop_range(m, m){
      aie::accum<accfloat, M_FLAT> in_vecs;
      in_vecs = aie::load_v<M_FLAT>(y + 2 * p1 * Q_TILE * 4);
      aie::vector<bfloat16, M_FLAT> in_splits = in_vecs.template to_vector<bfloat16>();
      aie::vector<bfloat16, M_FLAT> tw_splits = aie::load_v<M_FLAT>(&twiddle_T2_splits[0][0]);

      auto [low, high] = aie::interleave_zip(aie::filter_odd(in_splits, 1), aie::filter_even(in_splits, 1), 1);
      aie::vector<bfloat16, M_FLAT> in_splits_inv = aie::concat(low, high);
      aie::accum<accfloat, M_FLAT> tmp_splits;
      aie::accum<accfloat, M_FLAT> tx_vecs_real;
      aie::accum<accfloat, M_FLAT> tx_vecs_imag;
      aie::accum<accfloat, M_FLAT> tx_vecs;
      
      tx_vecs_real = aie::mul(in_splits, tw_splits);
      tx_vecs_imag = aie::mul(in_splits_inv, tw_splits);
      for (unsigned i = 1; i < kSplitCount; ++i) {
        tw_splits = aie::load_v<M_FLAT>(&twiddle_T2_splits[i][0]);
        tx_vecs_real = aie::mac(tx_vecs_real, in_splits, tw_splits);
        tx_vecs_imag = aie::mac(tx_vecs_imag, in_splits_inv, tw_splits);
      }
      for (unsigned k = 1; k < kSplitCount; ++k) {
        tmp_splits.from_vector(in_splits);
        in_vecs = aie::sub(in_vecs, tmp_splits);
        in_splits = in_vecs.template to_vector<bfloat16>();
        auto [low, high] = aie::interleave_zip(aie::filter_odd(in_splits, 1), aie::filter_even(in_splits, 1), 1);
        in_splits_inv = aie::concat(low, high);
        for (unsigned i = 0; i < kSplitCount; ++i) {
          tw_splits = aie::load_v<M_FLAT>(&twiddle_T2_splits[i][0]);
          tx_vecs_real = aie::mac(tx_vecs_real, in_splits, tw_splits);
          tx_vecs_imag = aie::mac(tx_vecs_imag, in_splits_inv, tw_splits);
        }
      }
      aie::vector<float, M_FLAT> tx_vecs_real_flt = tx_vecs_real.template to_vector<float>();
      aie::vector<float, M_FLAT> tx_vecs_imag_flt = tx_vecs_imag.template to_vector<float>();
      aie::vector<float, M_FLAT/2> real = aie::sub(aie::filter_even(tx_vecs_real_flt, 1), aie::filter_odd(tx_vecs_real_flt, 1));
      aie::vector<float, M_FLAT/2> imag = aie::add(aie::filter_even(tx_vecs_imag_flt, 1), aie::filter_odd(tx_vecs_imag_flt, 1));
      auto [low_tmp, high_tmp] = aie::interleave_zip(real, imag, 1);
      tx_vecs.from_vector(aie::concat(low_tmp, high_tmp), 0);

      using MMUL = aie::mmul<Q_TILE, 8, 8, bfloat16, bfloat16, accfloat>;
      MMUL OUT;
      in_splits = tx_vecs.template to_vector<bfloat16>();
      OUT.mul(in_splits, coeff_vecs);
      for (unsigned k = 1; k < kSplitCount; ++k){
        tmp_splits.from_vector(in_splits);
        tx_vecs = aie::sub(tx_vecs, tmp_splits);
        in_splits = tx_vecs.template to_vector<bfloat16>();
        OUT.mac(in_splits, coeff_vecs);
      }
      aie::vector<cint32, M_FLAT/2> out_cint32 = (OUT.template to_vector<float>()).cast_to<cint32>();
        aie::store_v(y + 2 * p1 * Q_TILE * 4, 
          out_cint32.cast_to<float>());
    }
    
    aie::vector<cint32, M_FLAT/2> B0 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 0 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B1 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 1 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B2 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 2 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B3 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 3 * Q_TILE * 4));
    auto [B01, B10] = aie::interleave_zip(B0, B1, 4);
    auto [B23, B32] = aie::interleave_zip(B2, B3, 4);
    auto [B0123_0, B0123_2] = aie::interleave_zip(B01, B23, 8);
    auto [B0123_1, B0123_3] = aie::interleave_zip(B10, B32, 8);
    aie::store_v(y + 2 * 0 * Q_TILE * 4, aie::transpose(B0123_0, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 1 * Q_TILE * 4, aie::transpose(B0123_2, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 2 * Q_TILE * 4, aie::transpose(B0123_1, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 3 * Q_TILE * 4, aie::transpose(B0123_3, Q_TILE, 4).cast_to<float>());
    

alignas(aie::vector_decl_align) static const bfloat16 twiddle_T3_64_splits[4][4][32] = {
    // T0
    {
        // split 0
        {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.92578125f, -0.3828125f, 0.70703125f, -0.70703125f, 0.3828125f, -0.92578125f,
            1.0f, 0.0f, 0.70703125f, -0.70703125f, 0.0f, -1.0f, -0.70703125f, -0.70703125f,
            1.0f, 0.0f, 0.3828125f, -0.92578125f, -0.70703125f, -0.70703125f, -0.92578125f, 0.3828125f,
        },
        // split 1
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.00189971923828125f, 0.00012874603271484375f, 7.534027099609375e-05f, -7.534027099609375e-05f, -0.00012874603271484375f, 0.00189971923828125f,
            0.0f, 0.0f, 7.534027099609375e-05f, -7.534027099609375e-05f, 0.0f, 0.0f, -7.534027099609375e-05f, -7.534027099609375e-05f,
            0.0f, 0.0f, -0.00012874603271484375f, 0.00189971923828125f, -7.534027099609375e-05f, -7.534027099609375e-05f, 0.00189971923828125f, -0.00012874603271484375f,
        },
        // split 2
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -2.0265579223632812e-06f, 3.2782554626464844e-07f, 1.7881393432617188e-07f, -1.7881393432617188e-07f, -3.2782554626464844e-07f, 2.0265579223632812e-06f,
            0.0f, 0.0f, 1.7881393432617188e-07f, -1.7881393432617188e-07f, 0.0f, 0.0f, -1.7881393432617188e-07f, -1.7881393432617188e-07f,
            0.0f, 0.0f, -3.2782554626464844e-07f, 2.0265579223632812e-06f, -1.7881393432617188e-07f, -1.7881393432617188e-07f, 2.0265579223632812e-06f, -3.2782554626464844e-07f,
        },
        // split 3
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        },
    },

    // T1
    {
        // split 0
        {
            1.0f, 0.0f, 0.99609375f, -0.09814453125f, 0.98046875f, -0.1953125f, 0.95703125f, -0.291015625f,
            1.0f, 0.0f, 0.8828125f, -0.470703125f, 0.5546875f, -0.83203125f, 0.09814453125f, -0.99609375f,
            1.0f, 0.0f, 0.6328125f, -0.7734375f, -0.1953125f, -0.98046875f, -0.8828125f, -0.470703125f,
            1.0f, 0.0f, 0.291015625f, -0.95703125f, -0.83203125f, -0.5546875f, -0.7734375f, 0.6328125f,
        },
        // split 1
        {
            0.0f, 0.0f, -0.00090789794921875f, 0.0001277923583984375f, 0.000316619873046875f, 0.00022220611572265625f, -9.1075897216796875e-05f, 0.000732421875f,
            0.0f, 0.0f, -0.00089263916015625f, -0.00069427490234375f, 0.000881195068359375f, 0.000560760498046875f, -0.0001277923583984375f, 0.00090789794921875f,
            0.0f, 0.0f, 0.00157928466796875f, 0.00042724609375f, 0.00022220611572265625f, -0.000316619873046875f, 0.00089263916015625f, -0.00069427490234375f,
            0.0f, 0.0f, -0.000732421875f, 9.1075897216796875e-05f, 0.000560760498046875f, -0.000881195068359375f, 0.00042724609375f, 0.00157928466796875f,
        },
        // split 2
        {
            0.0f, 0.0f, -1.1324882507324219e-06f, -4.0233135223388672e-07f, -1.1920928955078125e-07f, -2.9802322387695312e-08f, 1.7881393432617188e-07f, -1.4603137969970703e-06f,
            0.0f, 0.0f, 1.430511474609375e-06f, 6.5565109252929688e-07f, 1.5497207641601562e-06f, 8.9406967163085938e-07f, 4.0233135223388672e-07f, 1.1324882507324219e-06f,
            0.0f, 0.0f, 1.4901161193847656e-06f, -1.7881393432617188e-07f, -2.9802322387695312e-08f, 1.1920928955078125e-07f, -1.430511474609375e-06f, 6.5565109252929688e-07f,
            0.0f, 0.0f, 1.4603137969970703e-06f, -1.7881393432617188e-07f, 8.9406967163085938e-07f, -1.5497207641601562e-06f, -1.7881393432617188e-07f, 1.4901161193847656e-06f,
        },
        // split 3
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        },
    },

    // T2
    {
        // split 0
        {
            1.0f, 0.0f, 0.98046875f, -0.1953125f, 0.92578125f, -0.3828125f, 0.83203125f, -0.5546875f,
            1.0f, 0.0f, 0.83203125f, -0.5546875f, 0.3828125f, -0.92578125f, -0.1953125f, -0.98046875f,
            1.0f, 0.0f, 0.5546875f, -0.83203125f, -0.3828125f, -0.92578125f, -0.98046875f, -0.1953125f,
            1.0f, 0.0f, 0.1953125f, -0.98046875f, -0.92578125f, -0.3828125f, -0.5546875f, 0.83203125f,
        },
        // split 1
        {
            0.0f, 0.0f, 0.000316619873046875f, 0.00022220611572265625f, -0.00189971923828125f, 0.00012874603271484375f, -0.000560760498046875f, -0.000881195068359375f,
            0.0f, 0.0f, -0.000560760498046875f, -0.000881195068359375f, -0.00012874603271484375f, 0.00189971923828125f, 0.00022220611572265625f, -0.000316619873046875f,
            0.0f, 0.0f, 0.000881195068359375f, 0.000560760498046875f, 0.00012874603271484375f, 0.00189971923828125f, -0.000316619873046875f, 0.00022220611572265625f,
            0.0f, 0.0f, -0.00022220611572265625f, -0.000316619873046875f, 0.00189971923828125f, 0.00012874603271484375f, -0.000881195068359375f, -0.000560760498046875f,
        },
        // split 2
        {
            0.0f, 0.0f, -1.1920928955078125e-07f, -2.9802322387695312e-08f, -2.0265579223632812e-06f, 3.2782554626464844e-07f, -8.9406967163085938e-07f, -1.5497207641601562e-06f,
            0.0f, 0.0f, -8.9406967163085938e-07f, -1.5497207641601562e-06f, -3.2782554626464844e-07f, 2.0265579223632812e-06f, -2.9802322387695312e-08f, 1.1920928955078125e-07f,
            0.0f, 0.0f, 1.5497207641601562e-06f, 8.9406967163085938e-07f, 3.2782554626464844e-07f, 2.0265579223632812e-06f, 1.1920928955078125e-07f, -2.9802322387695312e-08f,
            0.0f, 0.0f, 2.9802322387695312e-08f, 1.1920928955078125e-07f, 2.0265579223632812e-06f, 3.2782554626464844e-07f, -1.5497207641601562e-06f, -8.9406967163085938e-07f,
        },
        // split 3
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        },
    },

    // T3
    {
        // split 0
        {
            1.0f, 0.0f, 0.95703125f, -0.291015625f, 0.83203125f, -0.5546875f, 0.6328125f, -0.7734375f,
            1.0f, 0.0f, 0.7734375f, -0.6328125f, 0.1953125f, -0.98046875f, -0.470703125f, -0.8828125f,
            1.0f, 0.0f, 0.470703125f, -0.8828125f, -0.5546875f, -0.83203125f, -0.99609375f, 0.09814453125f,
            1.0f, 0.0f, 0.09814453125f, -0.99609375f, -0.98046875f, -0.1953125f, -0.291015625f, 0.95703125f,
        },
        // split 1
        {
            0.0f, 0.0f, -9.1075897216796875e-05f, 0.000732421875f, -0.000560760498046875f, -0.000881195068359375f, 0.00157928466796875f, 0.00042724609375f,
            0.0f, 0.0f, -0.00042724609375f, -0.00157928466796875f, -0.00022220611572265625f, -0.000316619873046875f, -0.00069427490234375f, 0.00089263916015625f,
            0.0f, 0.0f, 0.00069427490234375f, 0.00089263916015625f, -0.000881195068359375f, 0.000560760498046875f, 0.00090789794921875f, -0.0001277923583984375f,
            0.0f, 0.0f, -0.0001277923583984375f, 0.00090789794921875f, -0.000316619873046875f, 0.00022220611572265625f, 0.000732421875f, -9.1075897216796875e-05f,
        },
        // split 2
        {
            0.0f, 0.0f, 1.7881393432617188e-07f, -1.4603137969970703e-06f, -8.9406967163085938e-07f, -1.5497207641601562e-06f, 1.4901161193847656e-06f, -1.7881393432617188e-07f,
            0.0f, 0.0f, 1.7881393432617188e-07f, -1.4901161193847656e-06f, 2.9802322387695312e-08f, 1.1920928955078125e-07f, 6.5565109252929688e-07f, -1.430511474609375e-06f,
            0.0f, 0.0f, -6.5565109252929688e-07f, -1.430511474609375e-06f, -1.5497207641601562e-06f, 8.9406967163085938e-07f, 1.1324882507324219e-06f, 4.0233135223388672e-07f,
            0.0f, 0.0f, 4.0233135223388672e-07f, 1.1324882507324219e-06f, 1.1920928955078125e-07f, -2.9802322387695312e-08f, -1.4603137969970703e-06f, 1.7881393432617188e-07f,
        },
        // split 3
        {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        },
    },
};
    // Stage 3

    for (unsigned p2 = 0; p2 < m; p2++) chess_prepare_for_pipelining chess_loop_range(m, m){
      aie::accum<accfloat, M_FLAT> in_vecs;
      in_vecs = aie::load_v<M_FLAT>(y + 2 * p2 * Q_TILE * 4);
      aie::vector<bfloat16, M_FLAT> in_splits = in_vecs.template to_vector<bfloat16>();
      aie::vector<bfloat16, M_FLAT> tw_splits = aie::load_v<M_FLAT>(&twiddle_T3_64_splits[p2][0][0]);

      auto [low, high] = aie::interleave_zip(aie::filter_odd(in_splits, 1), aie::filter_even(in_splits, 1), 1);
      aie::vector<bfloat16, M_FLAT> in_splits_inv = aie::concat(low, high);
      aie::accum<accfloat, M_FLAT> tmp_splits;
      aie::accum<accfloat, M_FLAT> tx_vecs_real;
      aie::accum<accfloat, M_FLAT> tx_vecs_imag;
      aie::accum<accfloat, M_FLAT> tx_vecs;
      
      tx_vecs_real = aie::mul(in_splits, tw_splits);
      tx_vecs_imag = aie::mul(in_splits_inv, tw_splits);
      for (unsigned i = 1; i < kSplitCount; ++i) {
        tw_splits = aie::load_v<M_FLAT>(&twiddle_T3_64_splits[p2][i][0]);
        tx_vecs_real = aie::mac(tx_vecs_real, in_splits, tw_splits);
        tx_vecs_imag = aie::mac(tx_vecs_imag, in_splits_inv, tw_splits);
      }
      for (unsigned k = 1; k < kSplitCount; ++k) {
        tmp_splits.from_vector(in_splits);
        in_vecs = aie::sub(in_vecs, tmp_splits);
        in_splits = in_vecs.template to_vector<bfloat16>();
        auto [low, high] = aie::interleave_zip(aie::filter_odd(in_splits, 1), aie::filter_even(in_splits, 1), 1);
        in_splits_inv = aie::concat(low, high);
        for (unsigned i = 0; i < kSplitCount; ++i) {
          tw_splits = aie::load_v<M_FLAT>(&twiddle_T3_64_splits[p2][i][0]);
          tx_vecs_real = aie::mac(tx_vecs_real, in_splits, tw_splits);
          tx_vecs_imag = aie::mac(tx_vecs_imag, in_splits_inv, tw_splits);
        }
      }
      aie::vector<float, M_FLAT> tx_vecs_real_flt = tx_vecs_real.template to_vector<float>();
      aie::vector<float, M_FLAT> tx_vecs_imag_flt = tx_vecs_imag.template to_vector<float>();
      aie::vector<float, M_FLAT/2> real = aie::sub(aie::filter_even(tx_vecs_real_flt, 1), aie::filter_odd(tx_vecs_real_flt, 1));
      aie::vector<float, M_FLAT/2> imag = aie::add(aie::filter_even(tx_vecs_imag_flt, 1), aie::filter_odd(tx_vecs_imag_flt, 1));
      auto [low_tmp, high_tmp] = aie::interleave_zip(real, imag, 1);
      tx_vecs.from_vector(aie::concat(low_tmp, high_tmp), 0);

      using MMUL = aie::mmul<Q_TILE, 8, 8, bfloat16, bfloat16, accfloat>;
      MMUL OUT;
      in_splits = tx_vecs.template to_vector<bfloat16>();
      OUT.mul(in_splits, coeff_vecs);
      for (unsigned k = 1; k < kSplitCount; ++k){
        tmp_splits.from_vector(in_splits);
        tx_vecs = aie::sub(tx_vecs, tmp_splits);
        in_splits = tx_vecs.template to_vector<bfloat16>();
        OUT.mac(in_splits, coeff_vecs);
      }
      aie::vector<cint32, M_FLAT/2> out_cint32 = (OUT.template to_vector<float>()).cast_to<cint32>();
        aie::store_v(y + 2 * p2 * Q_TILE * 4, 
          aie::transpose(out_cint32, Q_TILE, 4).cast_to<float>());
    }

    {
    aie::vector<cint32, M_FLAT/2> B0 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 0 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B1 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 1 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B2 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 2 * Q_TILE * 4));
    aie::vector<cint32, M_FLAT/2> B3 = aie::load_v<M_FLAT/2>((cint32*)(y + 2 * 3 * Q_TILE * 4));
    auto [B01, B10] = aie::interleave_zip(B0, B1, 8);
    auto [B23, B32] = aie::interleave_zip(B2, B3, 8);
    auto [B0123_0, B0123_2] = aie::interleave_unzip(B01, B23, 4);
    auto [B0123_1, B0123_3] = aie::interleave_unzip(B10, B32, 4);
    aie::store_v(y + 2 * 0 * Q_TILE * 4, aie::transpose(B0123_0, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 1 * Q_TILE * 4, aie::transpose(B0123_2, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 2 * Q_TILE * 4, aie::transpose(B0123_1, Q_TILE, 4).cast_to<float>());
    aie::store_v(y + 2 * 3 * Q_TILE * 4, aie::transpose(B0123_3, Q_TILE, 4).cast_to<float>());
    }
  if constexpr (PROFILING) {
    g_fft_elemwise_cycles = elemwise_cycles;
    g_fft_butterfly_cycles = butterfly_cycles;
  }
}

// Wrapper functions
extern "C" {

void fft_stockham_f32(float *input, const bfloat16 *twiddle, 
                      float *output) {
  unsigned long long start = 0;
  unsigned long long end = 0;
  if constexpr (PROFILING) {
    event0();
    start = get_cycles();
  }

  // Perform FFT using Stockham algorithm with GEMM-based complex multiplication
  for (int r = 0; r < 10000; ++r)
  fft_stockham_gemm<FFT_SIZE>(input, twiddle, output);

  if constexpr (PROFILING) {
    end = get_cycles();
    event1();
   
    g_fft_stockham_cycles = end - start;
    *((unsigned long long *)output + 0) = g_fft_stockham_cycles;
    *((unsigned long long *)output + 1) = g_fft_elemwise_cycles;
    *((unsigned long long *)output + 2) = g_fft_butterfly_cycles;
  }

}

void zero_f32(float *c) {
  zero_vectorized<float, FFT_SIZE * 2>(c);
}

} // extern "C"
