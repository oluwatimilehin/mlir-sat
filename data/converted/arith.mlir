builtin.module {
  func.func @test_mult_shifts(%arg0 : i32, %arg1 : i32) -> i32 {
    %mlirsat_mul_left_shift_const3 = arith.constant 3 : i32
    %0 = arith.shli %arg0, %mlirsat_mul_left_shift_const3 : i32
    %mlirsat_mul_left_shift_const7 = arith.constant 7 : i32
    %1 = arith.shli %arg1, %mlirsat_mul_left_shift_const7 : i32
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
