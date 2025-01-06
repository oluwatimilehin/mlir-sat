builtin.module {
  func.func @distribute(%A : memref<2x2xf64>, %B : memref<2x2xf64>, %C : memref<2x2xf64>) -> i32 {
    %AB = memref.alloc() : memref<2x2xf64>
    linalg.matmul ins(%A, %B : memref<2x2xf64>, memref<2x2xf64>) outs(%AB : memref<2x2xf64>)
    %AC = memref.alloc() : memref<2x2xf64>
    linalg.matmul ins(%A, %C : memref<2x2xf64>, memref<2x2xf64>) outs(%AC : memref<2x2xf64>)
    linalg.add ins(%AB, %AC : memref<2x2xf64>, memref<2x2xf64>) outs(%C : memref<2x2xf64>)
    memref.dealloc %AB : memref<2x2xf64>
    memref.dealloc %AC : memref<2x2xf64>
    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
  }
}
