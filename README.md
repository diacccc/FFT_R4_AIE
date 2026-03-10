<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Radix-4 Stockham FFT (Single Core AIE)

This directory contains a single-core AI Engine implementation of a **radix-4 Stockham FFT** for complex `f32` input/output, with twiddle factors stored as pre-split `bf16` slices.

The implementation is designed for MLIR-AIE flows and includes:
- AIE kernel code (`kernels/fft_stockham_f32.cc`)
- MLIR design generator (`single_core/single_core.py`)
- Host test and verification against FFTW (`test.cpp`)

## What Is Implemented

- FFT algorithm: **radix-4 DIT Stockham autosort**
- Data type path:
   - Input samples: complex `f32` (`[real, imag]` interleaved)
   - Twiddles: complex values represented as `8 x bf16` (4 real slices + 4 imag slices)
   - Output samples: complex `f32`
- Complex twiddle multiplication uses an Ozaki-style split/accumulate path:
   - Each scalar float is split into 4 `bf16` values.
   - Products are reconstructed by summing all pairwise slice products.
- Butterfly stage is also organized in a GEMM-friendly BF16 form:
   - Twiddle-multiplied values are re-split into 4 `bf16` slices per scalar.
   - The radix-4 butterfly is expressed as an 8x8 real matvec (complex-expanded W4 matrix).
   - Matrix terms are accumulated with split-pair products, matching a 16-term split multiply structure.

## Directory Layout

- `kernels/fft_stockham_f32.cc`
   - Core radix-4 Stockham kernel.
   - Exposes `fft_stockham_f32` and `zero_f32`.
- `single_core/single_core.py`
   - Creates the single-core AIE graph (shim/mem/core tiles, object FIFOs, runtime sequence).
- `single_core/Makefile`
   - Main entry point for build/run in this design.
- `test.cpp`
   - Host-side XRT app.
   - Generates random complex input and packed twiddles.
   - Verifies against FFTW forward FFT.
- `makefile-common`
   - Shared make logic used by this design.

## FFT and Twiddle Layout

### Complex sample layout

Input and output buffers use interleaved complex storage:

```text
[real_0, imag_0, real_1, imag_1, ..., real_(N-1), imag_(N-1)]
```

### Twiddle layout (stage-major, q-lane packed)

For each stage with stride `s = 4^stage`, each `q` lane stores 3 twiddles:
- `W^(q*m)`
- `W^(2*q*m)`
- `W^(3*q*m)`

where `m = N / (4^(stage+1))`.

Each complex twiddle is stored as 8 bf16 values:
- `real_split0..3`
- `imag_split0..3`

So each stage consumes `24 * s` bf16 elements.

## Kernel Compute Organization (Vectorization-Friendly)

Each radix-4 butterfly is implemented in two scalar stages that mirror the intended vectorized path:

1. Twiddle elementwise multiply in split BF16 form
- For `b`, `c`, `d`, complex multiplication with stage twiddles is computed as split products (`sum_i sum_j`) over 4-way bf16 slices.

2. BF16 GEMM-style radix-4 butterfly
- Inputs are arranged as:
   - `[a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i]`
- Butterfly is computed as an 8x8 real matrix multiply to produce:
   - `[y0_r, y0_i, y1_r, y1_i, y2_r, y2_i, y3_r, y3_i]`
- This structure is directly aligned with later loop-unrolling and vector-lane GEMM mapping.

## Constraints

- `N` must be a **power of 4** for this radix-4 flow (e.g. 4, 16, 64, 256, 1024, ...).
- Kernel is currently specialized for `f32` input/output and split `bf16` twiddles.
- Build requires C++23 support on host (for `std::bfloat16_t` in host code).

## Build and Run

Run all commands from `single_core/`.

### 1. Build

```bash
make
```

Common variants:

```bash
# Change FFT size
make N=1024

# Target device
make devicename=npu2 N=256

# Use alternative MLIR variants
env use_placed=1 make
env use_iron=1 make
```

### 2. Run

```bash
make run
```

Tune host runtime options:

```bash
make run verbosity=2 warmup=1 iters=10 N=256
```

### 3. Trace (optional)

```bash
make trace trace_size=65536 N=256
```

This generates `trace.txt` and parses it to JSON with the project trace parser.

## Verification

`test.cpp` verifies AIE output against FFTW (`fftwf_plan_dft_1d`, forward transform):

1. Generates random complex input in `[-1, 1]`.
2. Generates stage-packed twiddles with bf16 splitting.
3. Runs kernel and reads output buffer.
4. Computes reference FFT with FFTW.
5. Compares elementwise with absolute/relative tolerances from `common.h`.

At higher verbosity (`verbosity>=2`), a CSV of expected vs obtained values is emitted as:

```text
fft_results_N<FFT_SIZE>.csv
```

## Performance Notes

- Algorithmic complexity remains `O(N log N)`.
- This implementation is correctness-first and straightforward to read.
- The split/accumulate complex multiply is a scalar reference for a GEMM-friendly formulation; further vectorization/intrinsic tuning is expected for peak throughput.

## Known Practical Notes

- `single_core/README.md` in this folder tree is inherited from a matrix-multiplication template and does not describe this FFT design.
- Use this README as the source of truth for the current `fft_r4` implementation.
