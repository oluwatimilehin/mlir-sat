builtin.module {
  // computes ab * ((4a/a) + (8b/b))
  func.func @test_mult_shifts(%a : i32, %b : i32) -> i32 {
    // Calculate a * b
    %0 = arith.muli %a, %b : i32
    
    // Calculate a * 4
    %c4 = arith.constant 4 : i32
    %1 = arith.muli %a, %c4 : i32
    
    // Divide (a * 4) by a
    %2 = arith.divsi %1, %a : i32
    
    // Calculate b * 8
    %c8 = arith.constant 8 : i32
    %3 = arith.muli %b, %c8 : i32
    
    // Divide (b * 8) by b
    %4 = arith.divsi %3, %b : i32
    
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

