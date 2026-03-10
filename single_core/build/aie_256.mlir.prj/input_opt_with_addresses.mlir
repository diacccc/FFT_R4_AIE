module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @inInput_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @inInput_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memInput_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memInput_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @inTwiddle_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @inTwiddle_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @memTwiddle_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @memTwiddle_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @memOutput_buff_1() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memOutput_buff_0() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memOutput_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memOutput_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @outOutput_cons() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @outOutput() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memOutput_cons() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memOutput() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memTwiddle_cons() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @memTwiddle() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @inTwiddle_cons() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @inTwiddle() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @memInput_cons() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @memInput() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @inInput_cons() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.mlir.global external @inInput() {addr_space = 0 : i32} : !llvm.array<512 x f32>
  llvm.func @zero_f32(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @fft_stockham_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @memInput_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @memTwiddle_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @memOutput_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @memInput_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @memTwiddle_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @memOutput_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(53 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(49 : i32) : i32
    %11 = llvm.mlir.constant(52 : i32) : i32
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(4294967294 : index) : i64
    %16 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%17: i64):  // 2 preds: ^bb0, ^bb2
    %18 = llvm.icmp "slt" %17, %15 : i64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%11, %13) : (i32, i32) -> ()
    %19 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @zero_f32(%19) : (!llvm.ptr) -> ()
    llvm.call @llvm.aie2p.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%9, %13) : (i32, i32) -> ()
    %20 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %21 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @fft_stockham_f32(%21, %20, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%11, %13) : (i32, i32) -> ()
    %22 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @zero_f32(%22) : (!llvm.ptr) -> ()
    llvm.call @llvm.aie2p.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%9, %13) : (i32, i32) -> ()
    %23 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %24 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @fft_stockham_f32(%24, %23, %22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %12) : (i32, i32) -> ()
    %25 = llvm.add %17, %16 : i64
    llvm.br ^bb1(%25 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%11, %13) : (i32, i32) -> ()
    %26 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @zero_f32(%26) : (!llvm.ptr) -> ()
    llvm.call @llvm.aie2p.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%9, %13) : (i32, i32) -> ()
    %27 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.call @fft_stockham_f32(%28, %27, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %12) : (i32, i32) -> ()
    llvm.return
  }
}

