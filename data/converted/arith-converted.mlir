builtin.module {
  func.func @test_mult_shifts(%arg0 : i32, %arg1 : i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %0 = arith.muli %arg0, %c2 : i32
    %c4 = arith.constant 4 : i32
    %1 = arith.muli %0, %c4 : i32
    %c8 = arith.constant 8 : i32
    %2 = arith.muli %arg1, %c8 : i32
    %c16 = arith.constant 16 : i32
    %3 = arith.muli %2, %c16 : i32
    %4 = arith.addi %1, %3 : i32
    %5 = arith.muli %arg0, %arg1 : i32
    %result = arith.addi %4, %5 : i32
    func.return %result : i32
  }
  func.func @main() -> i32 {
    %val1 = arith.constant 123 : i32
    %val2 = arith.constant 456 : i32
    %result = func.call @test_mult_shifts(%val1, %val2) : (i32, i32) -> i32
    func.return %result : i32
  }
}
