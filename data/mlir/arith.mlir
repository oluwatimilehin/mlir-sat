builtin.module {
  // computes ab * (4a/a) + (8b/b) 
  func.func @test_mult_shifts(%arg0 : i32, %arg1 : i32) -> i32 {
    // Calculate arg0 * arg1
    %0 = arith.muli %arg0, %arg1 : i32
    
    // Calculate arg0 * 4
    %c4 = arith.constant 4 : i32
    %1 = arith.muli %arg0, %c4 : i32
    
    // Divide (arg0 * 4) by arg0
    %2 = arith.divsi %1, %arg0 : i32
    
    // Calculate arg1 * 8
    %c8 = arith.constant 8 : i32
    %3 = arith.muli %arg1, %c8 : i32
    
    // Divide (arg1 * 8) by arg1
    %4 = arith.divsi %3, %arg1 : i32
    
    // Add the division results
    %5 = arith.addi %2, %4 : i32
    
    // Multiply final result
    %result = arith.muli %0, %5 : i32
    
    func.return %result : i32
  }

  func.func @main() -> i32 {
    %val1 = arith.constant 123 : i32
    %val2 = arith.constant 456 : i32

    %result = func.call @test_mult_shifts(%val1, %val2) : (i32, i32) -> i32

    func.return %result : i32
  }
}

