builtin.module {
  func.func @fillI64Tensor2D(%tensor : tensor<?x?xi64>) -> tensor<?x?xi64> {
    %val = arith.constant 10 : i64
    %c0 = arith.constant 0 : index
    %rows = tensor.dim %tensor, %c0 : tensor<?x?xi64>
    %c1 = arith.constant 1 : index
    %cols = tensor.dim %tensor, %c1 : tensor<?x?xi64>
    %init = tensor.empty(%rows, %cols) : tensor<?x?xi64>
    %tensor_filled = linalg.fill ins(%val : i64) outs(%init : tensor<?x?xi64>) -> tensor<?x?xi64>
    func.return %tensor_filled : tensor<?x?xi64>
  }
  func.func @main() -> i32 {
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index
    %x_cast = tensor.empty(%c100, %c10) : tensor<?x?xi64>
    %x_filled = func.call @fillI64Tensor2D(%x_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %x = tensor.cast %x_filled : tensor<?x?xi64> to tensor<100x10xi64>
    %c150 = arith.constant 150 : index
    %y_cast = tensor.empty(%c10, %c150) : tensor<?x?xi64>
    %y_filled = func.call @fillI64Tensor2D(%y_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %y = tensor.cast %y_filled : tensor<?x?xi64> to tensor<10x150xi64>
    %c8 = arith.constant 8 : index
    %z_cast = tensor.empty(%c150, %c8) : tensor<?x?xi64>
    %z_filled = func.call @fillI64Tensor2D(%z_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %z = tensor.cast %z_filled : tensor<?x?xi64> to tensor<150x8xi64>
    %res = func.call @_2mm(%x, %y, %z) : (tensor<100x10xi64>, tensor<10x150xi64>, tensor<150x8xi64>) -> tensor<100x8xi64>
    %res_cast = tensor.cast %res : tensor<100x8xi64> to tensor<?x?xi64>
    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
  }
  func.func @_2mm(%x : tensor<100x10xi64>, %y : tensor<10x150xi64>, %z : tensor<150x8xi64>) -> tensor<100x8xi64> {
    %xy_init = tensor.empty() : tensor<100x150xi64>
    %xy = linalg.matmul ins(%x, %y : tensor<100x10xi64>, tensor<10x150xi64>) outs(%xy_init : tensor<100x150xi64>) -> tensor<100x150xi64>
    %xy_z_init = tensor.empty() : tensor<100x8xi64>
    %xy_z = linalg.matmul ins(%xy, %z : tensor<100x150xi64>, tensor<150x8xi64>) outs(%xy_z_init : tensor<100x8xi64>) -> tensor<100x8xi64>
    func.return %xy_z : tensor<100x8xi64>
  }
}
