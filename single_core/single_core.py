#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.
import argparse
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype


microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE FFT MLIR Design (Single Core)",
        description="Emits MLIR code for an FFT design of the given size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("-N", type=int, default=256, help="FFT size (must be power of 2)")
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "f32"], default="f32", 
        help="Input signal and twiddle factor data type"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["f32"],
        default="f32",
        help="Output signal data type"
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_fft(
        args.dev,
        args.N,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_fft(
    dev,
    N,
    dtype_in_str,
    dtype_out_str,
    trace_size,
):
    # N must be power of 2
    assert (N & (N - 1)) == 0, f"FFT size N={N} must be a power of 2"
    
    enable_tracing = True if trace_size > 0 else False

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    # Complex data: 2 values per sample (real + imaginary)
    # Twiddle factors are pre-split: 8 bf16 per complex (4 real + 4 imag splits)
    INPUT_sz = N * 2      # Complex input signal (float)
    TWIDDLE_sz = N * 8    # Pre-split complex twiddle factors (bf16)
    OUTPUT_sz = N * 2     # Complex output signal (float)

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # Data types for FFT buffers
            # Input/Output: complex float (2 values per sample)
            # Twiddle: pre-split bf16 (8 values per complex twiddle)
            #   - Radix-4: 3 twiddles per butterfly (W^(q*m), W^(2*q*m), W^(3*q*m))
            # #TODO: input_ty = np.ndarray[(N, 2), np.dtype[dtype_in]]
            input_ty = np.ndarray[(N * 2,), np.dtype[dtype_in]]
            twiddle_ty = np.ndarray[(N * 8,), np.dtype[bfloat16]]
            output_ty = np.ndarray[(N * 2,), np.dtype[dtype_out]]

            # AIE Core Function declarations
            zero = external_func(f"zero_f32", inputs=[output_ty])
            fft = external_func(
                f"fft_stockham_f32",
                inputs=[input_ty, twiddle_ty, output_ty],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input signal (complex)
            inInput = object_fifo("inInput", shim_tile, mem_tile, 2, input_ty)
            memInput = object_fifo("memInput", mem_tile, compute_tile2, 2, input_ty)
            object_fifo_link(inInput, memInput)

            # Twiddle factors (complex)
            inTwiddle = object_fifo("inTwiddle", shim_tile, mem_tile, 2, twiddle_ty)
            memTwiddle = object_fifo("memTwiddle", mem_tile, compute_tile2, 2, twiddle_ty)
            object_fifo_link(inTwiddle, memTwiddle)

            # Output signal (complex)
            memOutput = object_fifo("memOutput", compute_tile2, mem_tile, 2, output_ty)
            outOutput = object_fifo("outOutput", mem_tile, shim_tile, 2, output_ty)
            object_fifo_link(memOutput, outOutput)

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [compute_tile2]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # Core function for FFT computation
            @core(compute_tile2, f"fft_stockham_f32.o", stack_size=0x4000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    # Acquire output buffer
                    elem_out = memOutput.acquire(ObjectFifoPort.Produce, 1)
                    zero(elem_out)

                    # Acquire input signal and twiddle factors
                    elem_input = memInput.acquire(ObjectFifoPort.Consume, 1)
                    elem_twiddle = memTwiddle.acquire(ObjectFifoPort.Consume, 1)
                    
                    # Perform FFT computation
                    fft(elem_input, elem_twiddle, elem_out)
                    
                    # Release buffers
                    memInput.release(ObjectFifoPort.Consume, 1)
                    memTwiddle.release(ObjectFifoPort.Consume, 1)
                    memOutput.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(
                np.ndarray[(INPUT_sz,), np.dtype[dtype_in]],
                np.ndarray[(TWIDDLE_sz,), np.dtype[bfloat16]],
                np.ndarray[(OUTPUT_sz,), np.dtype[dtype_out]],
            )
            def sequence(Input, Twiddle, Output):

                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
                        coretile_events=[
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_2,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.events.CoreEvent.INSTR_EVENT_0,
                            trace_utils.events.CoreEvent.INSTR_EVENT_1,
                            trace_utils.events.CoreEvent.MEMORY_STALL,
                            trace_utils.events.CoreEvent.LOCK_STALL,
                            trace_utils.events.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # DMA transfers for FFT
                # Transfer input signal from host to device
                npu_dma_memcpy_nd(
                    metadata=inInput,
                    bd_id=1,
                    mem=Input,
                    sizes=[1, 1, 1, INPUT_sz],
                    strides=[0, 0, 0, 1],
                )

                # Transfer twiddle factors from host to device
                npu_dma_memcpy_nd(
                    metadata=inTwiddle,
                    bd_id=2,
                    mem=Twiddle,
                    sizes=[1, 1, 1, TWIDDLE_sz],
                    strides=[0, 0, 0, 1],
                )

                # Transfer output signal from device to host
                npu_dma_memcpy_nd(
                    metadata=outOutput,
                    bd_id=0,
                    mem=Output,
                    sizes=[1, 1, 1, OUTPUT_sz],
                    strides=[0, 0, 0, 1],
                )

                # Wait for DMA to complete
                dma_wait(outOutput)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
