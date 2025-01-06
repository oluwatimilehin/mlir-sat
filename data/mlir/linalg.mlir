  
  
  func.func @distribute(%A : memref<2x2xf64>, %B : memref<2x2xf64>, %C: memref<2x2xf64>) {
    //%matmul_res = func.call @matmul(%A, %B, %C) : (memref<2x2xf64>, memref<2x2xf64>, memref<2x2xf64>) -> memref<2x2xf64>
    //%sum_res = linalg.add 

    // Temporary buffers for (A × B) and (A × C)
    %AB = memref.alloc() : memref<2x2xf64>
    %AC = memref.alloc() : memref<2x2xf64>
    
    // Compute A × B
    linalg.matmul ins(%A, %B : memref<2x2xf64>, memref<2x2xf64>) 
                  outs(%AB : memref<2x2xf64>)
    
    // Compute A × C
    linalg.matmul ins(%A, %C : memref<2x2xf64>, memref<2x2xf64>) 
                  outs(%AC : memref<2x2xf64>)
    
    // Add the results: C = (A × B) + (A × C)
    linalg.add ins(%AB, %AC : memref<2x2xf64>, memref<2x2xf64>) 
               outs(%C : memref<2x2xf64>)
    
    // Free temporary buffers
    memref.dealloc %AB : memref<2x2xf64>
    memref.dealloc %AC : memref<2x2xf64>
    return
  }



func.func @matmul(%A : memref<2x2xf64>, %B : memref<2x2xf64>, %C : memref<2x2xf64>) {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A, %B : memref<2x2xf64>, memref<2x2xf64>) outs(%C : memref<2x2xf64>) {
    ^0(%a : f64, %b : f64, %acc_old : f64):
      %prod = arith.mulf %a, %b : f64
      %acc_new = arith.addf %acc_old, %prod : f64
      linalg.yield %acc_new : f64
    }
    func.return
  }


func.func @add(%arg0 : memref<2x2xf64>, %arg1 : memref<2x2xf64>, %arg2 : memref<2x2xf64>) {
  // Perform element-wise addition using linalg.generic
  linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
  }
  ins(%arg0, %arg1 : memref<2x2xf64>, memref<2x2xf64>) // Inputs
  outs(%arg2 : memref<2x2xf64>) // Output
  {
    ^bb0(%arg2_0 : f64, %arg3 : f64, %arg4 : f64): // Iteration block
      // Perform element-wise addition
      %1 = arith.addf %arg2_0, %arg3 : f64
      // Yield the result
      linalg.yield %1 : f64
  }
  func.return
}
