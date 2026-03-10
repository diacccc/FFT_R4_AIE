module {
  aie.device(npu2) {
    func.func private @zero_f32(memref<512xf32>)
    func.func private @fft_stockham_f32(memref<512xf32>, memref<2048xbf16>, memref<512xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inInput(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @memInput(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo.link [@inInput] -> [@memInput]([] [])
    aie.objectfifo @inTwiddle(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @memTwiddle(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@inTwiddle] -> [@memTwiddle]([] [])
    aie.objectfifo @memOutput(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @outOutput(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo.link [@memOutput] -> [@outOutput]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memOutput(Produce, 1) : !aie.objectfifosubview<memref<512xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        func.call @zero_f32(%1) : (memref<512xf32>) -> ()
        %2 = aie.objectfifo.acquire @memInput(Consume, 1) : !aie.objectfifosubview<memref<512xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %4 = aie.objectfifo.acquire @memTwiddle(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @fft_stockham_f32(%3, %5, %1) : (memref<512xf32>, memref<2048xbf16>, memref<512xf32>) -> ()
        aie.objectfifo.release @memInput(Consume, 1)
        aie.objectfifo.release @memTwiddle(Consume, 1)
        aie.objectfifo.release @memOutput(Produce, 1)
      }
      aie.end
    } {link_with = "fft_stockham_f32.o", stack_size = 16384 : i32}
    aiex.runtime_sequence @sequence(%arg0: memref<512xf32>, %arg1: memref<2048xbf16>, %arg2: memref<512xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, metadata = @inInput} : memref<512xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 2 : i64, metadata = @inTwiddle} : memref<2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, metadata = @outOutput} : memref<512xf32>
      aiex.npu.dma_wait {symbol = @outOutput}
    }
  }
}

