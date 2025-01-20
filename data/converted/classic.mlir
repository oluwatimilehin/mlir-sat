builtin.module {
  func.func @classic(%a : i32) -> i32 {
    func.return %a : i32
  }
  func.func @main() -> i32 {
    %c = arith.constant 1000 : i32
    %result = func.call @classic(%c) : (i32) -> i32
    func.return %result : i32
  }
}
