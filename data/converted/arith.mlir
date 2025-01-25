builtin.module {
  func.func @test_mult_shifts(%a : i32, %b : i32) -> i32 {
    %0 = arith.muli %a, %b : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %1 = arith.addi %c4, %c8 : i32
    %result = arith.muli %0, %1 : i32
    func.return %result : i32
  }
  func.func @main() -> i32 {
    %val1 = arith.constant 123 : i32
    %val2 = arith.constant 456 : i32
    %result = func.call @test_mult_shifts(%val1, %val2) : (i32, i32) -> i32
    func.return %result : i32
  }
}
