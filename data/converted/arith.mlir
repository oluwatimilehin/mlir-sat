builtin.module {
  func.func @test_mult_shifts(%arg0 : i32, %arg1 : i32) -> i32 {
    %c4 = arith.constant 8 : i32
    %0 = arith.muli %c4, %arg0 : i32
    %c16 = arith.constant 128 : i32
    %1 = arith.muli %c16, %arg1 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = arith.muli %arg0, %arg1 : i32
    %result = arith.addi %2, %3 : i32
    func.return %result : i32
  }
  func.func @main() -> i32 {
    %val1 = arith.constant 123 : i32
    %val2 = arith.constant 456 : i32
    %result = func.call @test_mult_shifts(%val1, %val2) : (i32, i32) -> i32
    func.return %result : i32
  }
}
