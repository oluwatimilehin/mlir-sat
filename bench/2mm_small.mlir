

module {
  func.func @_10mm(%arg0: tensor<73x77xi32>, %arg1: tensor<77x79xi32>) -> tensor<73x79xi32> {
    %0 = tensor.empty() : tensor<73x79xi32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<73x77xi32>, tensor<77x79xi32>) outs(%0 : tensor<73x79xi32>) -> tensor<73x79xi32>
    return %1 : tensor<73x79xi32>
  }
}
