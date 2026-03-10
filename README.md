<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# FFT Implementation for AIE Using GEMM-based Approach

This directory contains an FFT (Fast Fourier Transform) implementation for AMD AIE (AI Engine) devices that leverages the low-precision GEMM hardware for high-precision scientific computing.

## Overview

The FFT implementation uses a novel GEMM-based approach to perform discrete Fourier transform on complex-valued input data:
- **Input signal**: Complex float32 - 2 values per sample (real + imaginary)
- **Twiddle factors**: Complex float32 - 2 values per factor (real + imaginary)
- **Output signal**: Complex float32 - 2 values per sample (real + imaginary)

### GEMM-Based Complex Multiplication with Ozaki Scheme

The key innovation is performing complex multiplication using the Ozaki scheme with bfloat16 hardware:

1. **Float Splitting**: Each float32 value is split into 4 bfloat16 slices using error-free transformation:
   ```
   a = (float)a_0 + (float)a_1 + (float)a_2 + (float)a_3
   ```
   where a_0, a_1, a_2, a_3 are bfloat16 values.

2. **Pairwise Multiplication**: For complex multiplication (a + bj) * (c + dj):
   - Split a, b, c, d into 4 bf16 slices each
   - Compute ac = Σᵢ Σⱼ (aᵢ × cⱼ) using 16 bf16 multiplications
   - Compute bd, ad, bc similarly
   - Result: real = ac - bd, imag = ad + bc

3. **Benefits**:
   - Uses low-precision (bf16) hardware for high-precision (float32) computation
   - Each float multiplication becomes 16 bf16 multiplications
   - These map naturally to 4x8x8 GEMM kernels on AMD AIE-ML2
   - Maintains numerical accuracy suitable for scientific computing

The current implementation performs splitting and multiplication in scalar loops. For production use, these operations would be:
- Vectorized using AIE SIMD instructions
- Batched across butterfly operations sharing the same twiddle factors
- Executed via hardware GEMM intrinsics (4x8x8 for AIE-ML2)

## Files

**Kernel Implementation:**
- **kernels/fft_gemm_f32.cc** - GEMM-based FFT kernel with float splitting

**Host Code:**
- **test.cpp** - Host test code for FFT execution and verification (uses FFTW3 for reference)

**MLIR Design:**
- **single_core/single_core.py** - MLIR code generator for single-core FFT design
- **single_core/Makefile** - Build configuration

**Build System:**
- **makefile-common** - Common build settings
- **CMakeLists.txt** - CMake configuration with FFTW3 support

## Usage

### Building

From the `single_core` directory:

```bash
# Build with default FFT size (256)
make

# Build with custom FFT size (must be power of 2)
make N=512

# Build with specific device
make devicename=npu2 N=256
```

### Running

```bash
# Run with default parameters
./single_core.exe

# Run with custom parameters
./single_core.exe -v 2 --warmup 1 --iters 10
```

### Parameters

- **N**: FFT size (must be a power of 2, default: 256)
- **dtype_in**: Input data type (fixed to bf16)
- **dtype_out**: Output data type (fixed to f32)
- **devicename**: Target device (npu or npu2, default: npu2)

## Implementation Details

### FFT Algorithm

The implementation uses the **Cooley-Tukey FFT algorithm** (Radix-2 Decimation-in-Time):

1. **Bit-reversal permutation**: Reorder input data
2. **Butterfly operations**: Log₂(N) stages of butterfly computations
3. **Twiddle factor multiplication**: Complex multiplication with W[k] = exp(-j*2π*k/N)

The algorithm has O(N log N) complexity, which is significantly more efficient than the naive O(N²) DFT.

**Butterfly operation at each stage:**
```
For inputs u and v with twiddle factor w:
  output_i = u + v*w
  output_j = u - v*w
```

### Complex Operations

Complex multiplication is performed as:
```
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

### Data Layout

- Input signal: `[real_0, imag_0, real_1, imag_1, ..., real_N-1, imag_N-1]`
- Twiddle factors: `[real_0, imag_0, real_1, imag_1, ..., real_N-1, imag_N-1]`
- Output signal: `[real_0, imag_0, real_1, imag_1, ..., real_N-1, imag_N-1]`

## Performance

The current implementation uses the Cooley-Tukey FFT algorithm with O(N log N) complexity. For optimal performance on AIE, the kernel should be further vectorized using SIMD intrinsics.

### Computational Complexity

- **Operations**: O(N log N) with Cooley-Tukey FFT
  - Stages: log₂(N)
  - Butterflies per stage: N/2
  - Total butterflies: (N/2) * log₂(N)
- **Memory**: O(N) for each buffer (input, twiddle, output)
- **In-place capable**: Current implementation uses separate input/output buffers

## Future Enhancements

1. **Vectorization**: Implement SIMD operations using AIE intrinsics for butterfly operations
2. **In-place FFT**: Reduce memory usage by computing FFT in-place
3. **Radix-4/8**: Implement higher-radix FFT for better performance
4. **Multi-core**: Distribute FFT computation across multiple AIE cores
5. **Mixed radix**: Support non-power-of-2 FFT sizes
6. **Optimized twiddle access**: Cache frequently used twiddle factors

## Verification

The host code performs verification by:
1. Computing a reference FFT on the host using **FFTW3** library
2. Comparing AIE output against reference with tolerance:
   - Absolute tolerance: 0.01 (for f32)
   - Relative tolerance: 0.01 (for f32)

## Dependencies

- **FFTW3**: Single-precision FFT library for reference computation
  - On Ubuntu/Debian: `sudo apt-get install libfftw3-dev`
  - On other systems: See [FFTW installation guide](http://www.fftw.org/download.html)

## Notes

- FFT size N must be a power of 2 (required for Radix-2 Cooley-Tukey algorithm)
- Twiddle factors are pre-computed on the host as: W[k] = cos(2πk/N) - j*sin(2πk/N)
- The implementation uses bit-reversal permutation for in-order output
- Cooley-Tukey FFT provides O(N log N) complexity vs O(N²) for naive DFT
- The implementation is optimized for clarity; further SIMD optimization is possible

## Reference

Based on the AIE matrix multiplication examples from:
- AMD/Xilinx MLIR-AIE repository
- AIE programming guide
