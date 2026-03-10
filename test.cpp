//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <fftw3.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN float
#endif
#ifndef DTYPE_TWIDDLE
#define DTYPE_TWIDDLE std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT float
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using INPUT_DATATYPE = DTYPE_IN;        // Input signal (complex float)
using TWIDDLE_DATATYPE = DTYPE_TWIDDLE; // Twiddle factors (pre-split complex bf16)
using OUTPUT_DATATYPE = DTYPE_OUT;      // Output signal (complex float)
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
float abs_tol = matmul_common::get_abs_tol<OUTPUT_DATATYPE>();
float rel_tol = matmul_common::get_rel_tol<OUTPUT_DATATYPE>();

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("FFT Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);
  // Add FFT-specific option
  options.add_options()("fft_size", "FFT size (must be power of 2)", 
                        cxxopts::value<int>()->default_value("256"));

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();
  int c_col_maj = vm["c_col_maj"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  // For FFT, we use N as the FFT size
  int FFT_SIZE = vm["N"].as<int>();
  
  // Radix-4 FFT requires N to be a power of 4
  if (FFT_SIZE < 1 || (FFT_SIZE & (FFT_SIZE - 1)) != 0) {
    std::cerr << "Error: FFT_SIZE must be a power of 2 (for radix-4, preferably power of 4)\n";
    return 1;
  }
  // Check if it's a power of 4
  int log4_check = FFT_SIZE;
  while (log4_check > 1) {
    if (log4_check % 4 != 0) {
      std::cerr << "Warning: FFT_SIZE=" << FFT_SIZE << " is not a power of 4. "
                << "Radix-4 FFT requires power of 4 (4, 16, 64, 256, 1024, ...)\n";
      return 1;
    }
    log4_check /= 4;
  }
  
  if (verbosity >= 1) {
    std::cout << "FFT size: " << FFT_SIZE << std::endl;
  }

  // Complex data: 2 values (real + imag) per sample
  // Radix-4 Twiddle factors are pre-split: 4 bf16 splits per float value
  // Each complex twiddle = 8 bf16 values (4 for real + 4 for imag)
  // Each radix-4 butterfly uses 3 complex twiddles: W^(q*m), W^(2*q*m), W^(3*q*m)
  int INPUT_VOLUME = FFT_SIZE * 2;      // Complex input signal (2 floats)
  int TWIDDLE_VOLUME = FFT_SIZE * 8;    // Pre-split twiddle factors (approximately 8 bf16 per sample)
  int OUTPUT_VOLUME = FFT_SIZE * 2;     // Complex output signal (2 floats)

  size_t INPUT_SIZE = (INPUT_VOLUME * sizeof(INPUT_DATATYPE));
  size_t TWIDDLE_SIZE = (TWIDDLE_VOLUME * sizeof(TWIDDLE_DATATYPE));
  size_t OUTPUT_SIZE = (OUTPUT_VOLUME * sizeof(OUTPUT_DATATYPE));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_twiddle =
      xrt::bo(device, TWIDDLE_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  // Input signal (complex: real and imaginary parts)
  // Restrict values to range [-1, 1]
  INPUT_DATATYPE *bufInput = bo_input.map<INPUT_DATATYPE *>();
  std::vector<INPUT_DATATYPE> InputVec(INPUT_VOLUME);
  for (int i = 0; i < INPUT_VOLUME; i++) {
    // Generate random value in range [-1, 1]
    float rand_val = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
    InputVec[i] = static_cast<INPUT_DATATYPE>(rand_val);
  }
  memcpy(bufInput, InputVec.data(), (InputVec.size() * sizeof(INPUT_DATATYPE)));
  
  // Radix-4 Twiddle factors packed stage-major with contiguous q-blocks:
  // Each stage has stride s = 4^stage and requires 3 twiddles per q:
  //   W_N^(q*m), W_N^(2*q*m), W_N^(3*q*m) where m = N/(4^(stage+1))
  // Storage layout per stage (size 24*s):
  //   for q in [0..s-1], store 24 bf16 values contiguously:
  //   [tw1 real[4], tw1 imag[4], tw2 real[4], tw2 imag[4], tw3 real[4], tw3 imag[4]]
  TWIDDLE_DATATYPE *bufTwiddle = bo_twiddle.map<TWIDDLE_DATATYPE *>();
  std::vector<TWIDDLE_DATATYPE> TwiddleVec(TWIDDLE_VOLUME);
  const double PI = 3.14159265358979323846;
  
  // Helper function to split float into 4 bf16 slices
  auto split_to_bf16 = [](float f, TWIDDLE_DATATYPE splits[4]) {
    float remainder = f;
    for (int i = 0; i < 4; i++) {
      TWIDDLE_DATATYPE bf = static_cast<TWIDDLE_DATATYPE>(remainder);
      splits[i] = bf;
      remainder = remainder - static_cast<float>(bf);
    }
  };
  
  std::fill(TwiddleVec.begin(), TwiddleVec.end(), static_cast<TWIDDLE_DATATYPE>(0));
  int stage_twiddle_base = 0;
  int s = 1;
  // Radix-4: iterate over stages where s = 4^stage
  while (s < FFT_SIZE) {
    int n = FFT_SIZE / s;
    int m = n / 4;  // Radix-4: quarter of the block size
    
    for (int q = 0; q < s; ++q) {
      // Generate 3 twiddle factors for radix-4 butterfly:
      // W^(q*m), W^(2*q*m), W^(3*q*m)
      for (int tw = 0; tw < 3; ++tw) {
        int k = q * m * (tw + 1);
        double angle = -2.0 * PI * k / FFT_SIZE;
        float twiddle_real = static_cast<float>(cos(angle));
        float twiddle_imag = static_cast<float>(sin(angle));

        TWIDDLE_DATATYPE real_splits[4];
        TWIDDLE_DATATYPE imag_splits[4];
        split_to_bf16(twiddle_real, real_splits);
        split_to_bf16(twiddle_imag, imag_splits);

        // Store in contiguous q-block: 24 bf16 per q (3 twiddles * 8 values)
        int q_base = stage_twiddle_base + q * 24;
        int tw_base = q_base + tw * 8;
        TwiddleVec[tw_base + 0] = real_splits[0];
        TwiddleVec[tw_base + 1] = real_splits[1];
        TwiddleVec[tw_base + 2] = real_splits[2];
        TwiddleVec[tw_base + 3] = real_splits[3];
        TwiddleVec[tw_base + 4] = imag_splits[0];
        TwiddleVec[tw_base + 5] = imag_splits[1];
        TwiddleVec[tw_base + 6] = imag_splits[2];
        TwiddleVec[tw_base + 7] = imag_splits[3];
      }
    }
    stage_twiddle_base += 24 * s;  // 3 twiddles * 8 bf16 per twiddle
    s <<= 2;  // Multiply by 4 for radix-4
  }
  memcpy(bufTwiddle, TwiddleVec.data(), (TwiddleVec.size() * sizeof(TWIDDLE_DATATYPE)));

  // Initialize output buffer
  OUTPUT_DATATYPE *bufOut = bo_out.map<OUTPUT_DATATYPE *>();
  std::vector<OUTPUT_DATATYPE> OutputVec(OUTPUT_VOLUME);
  memset(bufOut, 0, OUTPUT_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "FFT_SIZE  = " << FFT_SIZE << "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
  }

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_twiddle.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;
  // For radix-4 FFT: N*log4(N) stages, each with N/4 radix-4 butterflies
  // Each radix-4 butterfly: 12 complex multiplications, ~6 ops each = 72 ops
  // But some are trivial (unity twiddle), so approximate as ~36 ops per butterfly
  float ops = FFT_SIZE * log2(FFT_SIZE) / log2(4.0) * 36.0;

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_twiddle, bo_out,
                      bo_tmp1, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    if (do_verify) {
      memcpy(OutputVec.data(), bufOut, (OutputVec.size() * sizeof(OUTPUT_DATATYPE)));
      if (verbosity >= 1) {
        std::cout << "Verifying against reference FFT (using FFTW) ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      
      // Compute reference FFT using FFTW
      std::vector<OUTPUT_DATATYPE> RefOutput(OUTPUT_VOLUME, 0.0f);
      
      // Allocate FFTW arrays
      fftwf_complex *fft_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
      fftwf_complex *fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
      
      // Copy input data to FFTW format
      for (int i = 0; i < FFT_SIZE; i++) {
        fft_in[i][0] = float(InputVec[2*i]);     // real part
        fft_in[i][1] = float(InputVec[2*i+1]);   // imaginary part
      }
      
      // Create FFTW plan and execute
      fftwf_plan plan = fftwf_plan_dft_1d(FFT_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
      fftwf_execute(plan);
      
      // Copy FFTW output to RefOutput
      for (int i = 0; i < FFT_SIZE; i++) {
        RefOutput[2*i] = fft_out[i][0];     // real part
        RefOutput[2*i+1] = fft_out[i][1];   // imaginary part
      }
      
      // Clean up FFTW
      fftwf_destroy_plan(plan);
      fftwf_free(fft_in);
      fftwf_free(fft_out);
      
      // Compare results
      errors = 0;
      float max_abs_error = 0.0f;
      float min_abs_error = std::numeric_limits<float>::max();
      float max_rel_error = 0.0f;
      int max_abs_error_idx = 0;
      int min_abs_error_idx = 0;
      int max_rel_error_idx = 0;
      
      for (int i = 0; i < OUTPUT_VOLUME; i++) {
        float diff = std::abs(OutputVec[i] - RefOutput[i]);
        float ref_val = std::abs(RefOutput[i]);
        float rel_error = (ref_val > 1e-10f) ? (diff / ref_val) : 0.0f;
        
        // Track maximum and minimum errors
        if (diff > max_abs_error) {
          max_abs_error = diff;
          max_abs_error_idx = i;
        }
        if (diff < min_abs_error) {
          min_abs_error = diff;
          min_abs_error_idx = i;
        }
        if (rel_error > max_rel_error) {
          max_rel_error = rel_error;
          max_rel_error_idx = i;
        }
        
        if (diff > abs_tol && diff > rel_tol * ref_val) {
          if (verbosity >= 2) {
            std::cout << "Mismatch at index " << i << ": got " << OutputVec[i]
                      << ", expected " << RefOutput[i] << std::endl;
          }
          errors++;
        }
      }
      
      std::cout << "Minimum absolute error: " << min_abs_error 
                << " at index " << min_abs_error_idx << std::endl;
      std::cout << "Maximum absolute error: " << max_abs_error 
                << " at index " << max_abs_error_idx << std::endl;
      std::cout << "Maximum relative error: " << max_rel_error 
                << " at index " << max_rel_error_idx << std::endl;
      
      // Write results to CSV file when verbosity >= 2
      if (verbosity >= 2) {
        std::string csv_filename = "fft_results_N" + std::to_string(FFT_SIZE) + ".csv";
        std::ofstream csv_file(csv_filename);
        
        if (csv_file.is_open()) {
          // Write CSV header
          csv_file << "Index,Sample,Component,Expected,Obtained,AbsError,RelError\n";
          
          // Write data for each complex sample
          for (int i = 0; i < FFT_SIZE; i++) {
            int real_idx = 2*i;
            int imag_idx = 2*i + 1;
            
            float real_expected = RefOutput[real_idx];
            float real_obtained = OutputVec[real_idx];
            float real_abs_error = std::abs(real_obtained - real_expected);
            float real_rel_error = (std::abs(real_expected) > 1e-10f) ? 
                                   (real_abs_error / std::abs(real_expected)) : 0.0f;
            
            float imag_expected = RefOutput[imag_idx];
            float imag_obtained = OutputVec[imag_idx];
            float imag_abs_error = std::abs(imag_obtained - imag_expected);
            float imag_rel_error = (std::abs(imag_expected) > 1e-10f) ? 
                                   (imag_abs_error / std::abs(imag_expected)) : 0.0f;
            
            // Write real part
            csv_file << real_idx << "," << i << ",Real,"
                    << std::setprecision(10) << real_expected << ","
                    << real_obtained << ","
                    << real_abs_error << ","
                    << real_rel_error << "\n";
            
            // Write imaginary part
            csv_file << imag_idx << "," << i << ",Imag,"
                    << std::setprecision(10) << imag_expected << ","
                    << imag_obtained << ","
                    << imag_abs_error << ","
                    << imag_rel_error << "\n";
          }
          
          csv_file.close();
          std::cout << "\nResults written to " << csv_filename << std::endl;
        } else {
          std::cerr << "Warning: Could not open " << csv_filename << " for writing." << std::endl;
        }
      }
      
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: FFT results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // Only write out trace of last iteration.
  if (trace_size > 0) {
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());
  }
  
  if (verbosity >= 2) {
    std::cout << "First few output values (complex):" << std::endl;
    for (int i = 0; i < std::min(8, FFT_SIZE); i++) {
      std::cout << "  [" << i << "] = " << bufOut[2*i] 
                << " + j*" << bufOut[2*i+1] << std::endl;
    }
  }

  std::cout << std::endl
            << "Avg NPU FFT time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << ops / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU FFT time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << ops / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU FFT time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << ops / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << " (out of " << OUTPUT_VOLUME << " values)";
    std::cout << "\n\n";

    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
