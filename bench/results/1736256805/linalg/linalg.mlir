builtin.module {
  func.func @distribute(%A : memref<2x2xf64>, %B : memref<2x2xf64>, %AB : memref<2x2xf64>, %C : memref<2x2xf64>, %AC : memref<2x2xf64>, %D : memref<2x2xf64>) -> i32 {
    linalg.add ins(%B, %C : memref<2x2xf64>, memref<2x2xf64>) outs(%AC : memref<2x2xf64>)
    linalg.matmul ins(%A, %AC : memref<2x2xf64>, memref<2x2xf64>) outs(%D : memref<2x2xf64>)
    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
  }
}
