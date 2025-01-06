builtin.module {
  func.func @test_mult_shifts(%arg0: i32, %arg1: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    
    // Series of multiplications that could be optimized to shifts
    %0 = arith.muli %arg0, %c2 : i32      // Multiply arg0 by 2
    %1 = arith.muli %0, %c4 : i32         // Multiply the result by 4 (or shift left by 2)
    %2 = arith.muli %arg1, %c8 : i32      // Multiply arg1 by 8
    %3 = arith.muli %2, %c16 : i32        // Multiply the result by 16 (or shift left by 4)
    
    // Add results together
    %4 = arith.addi %1, %3 : i32          // Add the two intermediate results
    
    // Multiply arg0 and arg1, add it to the previous result
    %5 = arith.muli %arg0, %arg1 : i32    // Multiply arg0 by arg1
    %result = arith.addi %4, %5 : i32     // Add the result to the previous computation
    
    printf.print_format "The actual result is {}\n", %result : i32
    func.return %result : i32             // Return the computed result
  }

  func.func @main() -> i32 {
    %val1 = arith.constant 123 : i32      // Constant value 123
    %val2 = arith.constant 456 : i32      // Constant value 456
    %result = func.call @test_mult_shifts(%val1, %val2) : (i32, i32) -> i32
    
    // Print the expected result for comparison
    %expected = arith.constant 115440 : i32    // Replace with proper expected result logic if needed
    printf.print_format "The expected result is {}\n", %expected : i32
    
    %c0 = arith.constant 0 : i32          // Return value of main
    func.return %c0 : i32
  }
}
