; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@inInput_cons_buff_1 = external global [512 x float]
@inInput_cons_buff_0 = external global [512 x float]
@memInput_cons_buff_1 = external global [512 x float]
@memInput_cons_buff_0 = external global [512 x float]
@inTwiddle_cons_buff_1 = external global [2048 x bfloat]
@inTwiddle_cons_buff_0 = external global [2048 x bfloat]
@memTwiddle_cons_buff_1 = external global [2048 x bfloat]
@memTwiddle_cons_buff_0 = external global [2048 x bfloat]
@memOutput_buff_1 = external global [512 x float]
@memOutput_buff_0 = external global [512 x float]
@memOutput_cons_buff_1 = external global [512 x float]
@memOutput_cons_buff_0 = external global [512 x float]
@outOutput_cons = external global [512 x float]
@outOutput = external global [512 x float]
@memOutput_cons = external global [512 x float]
@memOutput = external global [512 x float]
@memTwiddle_cons = external global [2048 x bfloat]
@memTwiddle = external global [2048 x bfloat]
@inTwiddle_cons = external global [2048 x bfloat]
@inTwiddle = external global [2048 x bfloat]
@memInput_cons = external global [512 x float]
@memInput = external global [512 x float]
@inInput_cons = external global [512 x float]
@inInput = external global [512 x float]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

declare void @zero_f32(ptr)

declare void @fft_stockham_f32(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_f32(ptr @memOutput_buff_0)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @fft_stockham_f32(ptr @memInput_cons_buff_0, ptr @memTwiddle_cons_buff_0, ptr @memOutput_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_f32(ptr @memOutput_buff_1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @fft_stockham_f32(ptr @memInput_cons_buff_1, ptr @memTwiddle_cons_buff_1, ptr @memOutput_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_f32(ptr @memOutput_buff_0)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @fft_stockham_f32(ptr @memInput_cons_buff_0, ptr @memTwiddle_cons_buff_0, ptr @memOutput_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
