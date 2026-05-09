// E2E coverage for GPUPushDownDMABoundsToConsumers + buffer_resource_cast
// validBytes hack on coalesced_gather_dma sources whose innermost row is
// not statically DWORD-aligned, plus the runtime-aligned-but-dynamic case
// where both wrap and consumer-pad must be skipped.
//
// All-1.0 inputs => output cell = K.
//
// Two cases:
//   matmul_f16_padded_64x43x64   : K=43 (86 B/row, NOT DWORD-aligned).
//                                  Producer-side validBytes wrap + consumer
//                                  pad both fire.
//   matmul_f16_dynamic_80x80x80  : All dims = 80 (160 B/row, DWORD-aligned)
//                                  but non-tile-divisible, so codegen
//                                  produces a tile-induced dynamic inner
//                                  slice with affine.min UB equal to the
//                                  K-tile. Both skips in
//                                  GPUPushDownDMABoundsToConsumers must
//                                  fire (no validBytes wrap, no pad).

func.func @matmul_f16_padded_64x43x64() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<64x43xf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<43x64xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>)
      -> tensor<64x64xf32>
  %res = linalg.matmul
      ins(%lhs, %rhs : tensor<64x43xf16>, tensor<43x64xf16>)
      outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  check.expect_almost_eq_const(%res, dense<43.0> : tensor<64x64xf32>)
      : tensor<64x64xf32>
  return
}

func.func @matmul_f16_dynamic_80x80x80() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<80x80xf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<80x80xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<80x80xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<80x80xf32>)
      -> tensor<80x80xf32>
  %res = linalg.matmul
      ins(%lhs, %rhs : tensor<80x80xf16>, tensor<80x80xf16>)
      outs(%fill : tensor<80x80xf32>) -> tensor<80x80xf32>
  check.expect_almost_eq_const(%res, dense<80.0> : tensor<80x80xf32>)
      : tensor<80x80xf32>
  return
}
