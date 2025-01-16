func.func @classic(%a: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %mul = arith.muli %a, %c2 : i32
  %div = arith.divsi %mul, %c2 : i32
  
  func.return %div : i32
}

func.func @main() -> i32 {
    %c = arith.constant 1000 : i32
    %result = func.call @classic(%c) : (i32) -> i32
    
    func.return %result : i32
}

