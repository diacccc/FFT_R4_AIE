module {
  aie.device(npu2) {
    memref.global "public" @outOutput_cons : memref<512xf32>
    memref.global "public" @outOutput : memref<512xf32>
    memref.global "public" @memOutput_cons : memref<512xf32>
    memref.global "public" @memOutput : memref<512xf32>
    memref.global "public" @memTwiddle_cons : memref<2048xbf16>
    memref.global "public" @memTwiddle : memref<2048xbf16>
    memref.global "public" @inTwiddle_cons : memref<2048xbf16>
    memref.global "public" @inTwiddle : memref<2048xbf16>
    memref.global "public" @memInput_cons : memref<512xf32>
    memref.global "public" @memInput : memref<512xf32>
    memref.global "public" @inInput_cons : memref<512xf32>
    memref.global "public" @inInput : memref<512xf32>
    func.func private @zero_f32(memref<512xf32>)
    func.func private @fft_stockham_f32(memref<512xf32>, memref<2048xbf16>, memref<512xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %outOutput_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "outOutput_cons_prod_lock_0"}
    %outOutput_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "outOutput_cons_cons_lock_0"}
    %memOutput_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 8192 : i32, sym_name = "memOutput_cons_buff_0"} : memref<512xf32> 
    %memOutput_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 10240 : i32, sym_name = "memOutput_cons_buff_1"} : memref<512xf32> 
    %memOutput_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "memOutput_cons_prod_lock_0"}
    %memOutput_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "memOutput_cons_cons_lock_0"}
    %memOutput_buff_0 = aie.buffer(%tile_0_2) {address = 24576 : i32, sym_name = "memOutput_buff_0"} : memref<512xf32> 
    %memOutput_buff_1 = aie.buffer(%tile_0_2) {address = 26624 : i32, sym_name = "memOutput_buff_1"} : memref<512xf32> 
    %memOutput_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memOutput_prod_lock_0"}
    %memOutput_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memOutput_cons_lock_0"}
    %memTwiddle_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, sym_name = "memTwiddle_cons_buff_0"} : memref<2048xbf16> 
    %memTwiddle_cons_buff_1 = aie.buffer(%tile_0_2) {address = 20480 : i32, sym_name = "memTwiddle_cons_buff_1"} : memref<2048xbf16> 
    %memTwiddle_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memTwiddle_cons_prod_lock_0"}
    %memTwiddle_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memTwiddle_cons_cons_lock_0"}
    %inTwiddle_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "inTwiddle_cons_buff_0"} : memref<2048xbf16> 
    %inTwiddle_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 4096 : i32, sym_name = "inTwiddle_cons_buff_1"} : memref<2048xbf16> 
    %inTwiddle_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inTwiddle_cons_prod_lock_0"}
    %inTwiddle_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inTwiddle_cons_cons_lock_0"}
    %inTwiddle_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "inTwiddle_prod_lock_0"}
    %inTwiddle_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "inTwiddle_cons_lock_0"}
    %memInput_cons_buff_0 = aie.buffer(%tile_0_2) {address = 28672 : i32, sym_name = "memInput_cons_buff_0"} : memref<512xf32> 
    %memInput_cons_buff_1 = aie.buffer(%tile_0_2) {address = 30720 : i32, sym_name = "memInput_cons_buff_1"} : memref<512xf32> 
    %memInput_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memInput_cons_prod_lock_0"}
    %memInput_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memInput_cons_cons_lock_0"}
    %inInput_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 12288 : i32, sym_name = "inInput_cons_buff_0"} : memref<512xf32> 
    %inInput_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 14336 : i32, sym_name = "inInput_cons_buff_1"} : memref<512xf32> 
    %inInput_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inInput_cons_prod_lock_0"}
    %inInput_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inInput_cons_cons_lock_0"}
    %inInput_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "inInput_prod_lock_0"}
    %inInput_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "inInput_cons_lock_0"}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 1, DMA : 2>
      aie.connect<DMA : 2, South : 2>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memOutput_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_f32(%memOutput_buff_0) : (memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memTwiddle_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @fft_stockham_f32(%memInput_cons_buff_0, %memTwiddle_cons_buff_0, %memOutput_buff_0) : (memref<512xf32>, memref<2048xbf16>, memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memTwiddle_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memOutput_cons_lock_0, Release, 1)
      aie.use_lock(%memOutput_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_f32(%memOutput_buff_1) : (memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memTwiddle_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @fft_stockham_f32(%memInput_cons_buff_1, %memTwiddle_cons_buff_1, %memOutput_buff_1) : (memref<512xf32>, memref<2048xbf16>, memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memTwiddle_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memOutput_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%memOutput_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_f32(%memOutput_buff_0) : (memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memTwiddle_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @fft_stockham_f32(%memInput_cons_buff_0, %memTwiddle_cons_buff_0, %memOutput_buff_0) : (memref<512xf32>, memref<2048xbf16>, memref<512xf32>) -> ()
      aie.use_lock(%memInput_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memTwiddle_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memOutput_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "fft_stockham_f32.o", stack_size = 16384 : i32}
    aiex.runtime_sequence @sequence(%arg0: memref<512xf32>, %arg1: memref<2048xbf16>, %arg2: memref<512xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, metadata = @inInput} : memref<512xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 2 : i64, metadata = @inTwiddle} : memref<2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, metadata = @outOutput} : memref<512xf32>
      aiex.npu.dma_wait {symbol = @outOutput}
    }
    aie.shim_dma_allocation @inInput(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inInput_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inInput_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inInput_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inInput_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inInput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inInput_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inInput_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inTwiddle_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inTwiddle_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inTwiddle_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inTwiddle_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inTwiddle_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inTwiddle_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inTwiddle_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inTwiddle_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inTwiddle_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inTwiddle_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inTwiddle_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inTwiddle_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%memOutput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memOutput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%memOutput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memOutput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%memOutput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%memOutput_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%memOutput_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%memOutput_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memInput_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memInput_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memInput_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memInput_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memTwiddle_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memTwiddle_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memTwiddle_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memTwiddle_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memTwiddle_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memTwiddle_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memOutput_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memOutput_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memOutput_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memOutput_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memOutput_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @inTwiddle(MM2S, 1, 0)
    aie.shim_dma_allocation @outOutput(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

