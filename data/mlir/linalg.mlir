// I should ideally allocate AB and AC in the function, but `dealloc` keeps unnecessary operations around after the conversion, and xDSL does not play nicely with memref.alloca
func.func @distribute(%A : memref<2x2xf64>, %B : memref<2x2xf64>, %AB: memref<2x2xf64>,  %C : memref<2x2xf64>, %AC: memref<2x2xf64>, %D : memref<2x2xf64>) -> i32 {
  // Compute A × B
  linalg.matmul ins(%A, %B : memref<2x2xf64>, memref<2x2xf64>)
                outs(%AB : memref<2x2xf64>)

  // Compute A × C
  linalg.matmul ins(%A, %C : memref<2x2xf64>, memref<2x2xf64>)
                outs(%AC : memref<2x2xf64>)

  // Compute A × B + A x C
  linalg.add ins(%AB, %AC : memref<2x2xf64>, memref<2x2xf64>)
             outs(%D : memref<2x2xf64>)

  
  %c0 = arith.constant 0 : i32
  func.return %c0 : i32
}
