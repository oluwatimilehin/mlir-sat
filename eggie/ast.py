from dataclasses import dataclass
from typing import List


@dataclass
class ExprAST:
    pass


@dataclass
class ExprTypeAST(ExprAST):
    pass


@dataclass
class TensorTypeAST(ExprTypeAST):
    i: int
    j: int
    t: str

    def __str__(self) -> str:
        i_str = f"{self.i}" if self.i != -1 else "?"
        j_str = f"{self.j}" if self.j != -1 else "?"
        return f"tensor<{i_str}x{j_str}x{self.t}>"


@dataclass
class IntegerTypeAST(ExprTypeAST):
    width: int

    def __str__(self) -> str:
        return f"i{self.width}"


@dataclass
class IndexTypeAST(ExprTypeAST):
    name = "index"

    def __str__(self) -> str:
        return self.name + "\n"


@dataclass
class SSAExprAST(ExprAST):
    name: str
    type: ExprTypeAST


@dataclass
class OperationAST(ExprAST):
    pass


@dataclass
class BlockAST(ExprAST):
    args: List[SSAExprAST]
    ops: List[OperationAST]

    def __str__(self) -> str:
        # todo: handle args
        result = ""
        for op in self.ops:
            result += str(op) + "\n"

        return result


@dataclass
class RegionAST(ExprAST):
    blocks: List[BlockAST]

    def __str__(self) -> str:
        result = "builtin.module { \n"

        for block in self.blocks:
            result += str(block)

        result += "}"
        return result


@dataclass
class ArithConstantAst(OperationAST):
    val: int
    out: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.out.name} = arith.constant {self.val} : {self.out.type} \n"
        return result


@dataclass
class FuncAST(OperationAST):
    name: str
    args: List[SSAExprAST]
    body: List[OperationAST]
    type: TensorTypeAST

    def __str__(self) -> str:
        args_list = [f"%{ssa.name} : {ssa.type}" for ssa in self.args]
        args_str = "(" + ", ".join(args_list) + ")"
        result = f"func.func @{self.name}{args_str} -> {self.type}" + " { \n"
        for op in self.body:
            result += str(op)

        result += "} \n"
        return result


@dataclass
class FuncCallAST(OperationAST):
    callee: str
    args: List[SSAExprAST]
    out: SSAExprAST

    def __str__(self) -> str:
        arg_names = [f"%{ssa.name}" for ssa in self.args]
        args_str = "(" + ", ".join(arg_names) + ")"

        arg_types = [f"{ssa.type}" for ssa in self.args]
        arg_types_str = "(" + ", ".join(arg_types) + ")"
        result = f"%{self.out.name} = func.call {self.callee}{args_str} : {arg_types_str} -> {self.out.type} \n"
        return result


@dataclass
class FuncReturnAST(OperationAST):
    return_val: SSAExprAST

    def __str__(self) -> str:
        result = f"func.return %{self.return_val.name} : {self.return_val.type} \n"
        return result


@dataclass
class LinalgFillAST(OperationAST):
    scalar: SSAExprAST
    out: SSAExprAST
    return_val: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.return_val.name} = linalg.fill ins(%{self.scalar.name} : {self.scalar.type}) outs(%{self.out.name} : {self.out.type}) -> {self.return_val.type} \n"
        return result


@dataclass
class LinalgMatmulAST(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST
    return_val: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.return_val.name} = linalg.matmul ins(%{self.x.name}, %{self.y.name} : {self.x.type}, {self.y.type}) outs(%{self.out.name} : {self.out.type}) -> {self.return_val.type} \n"
        return result


@dataclass
class PrintFormatAST(OperationAST):
    format_str: str
    vals: List[SSAExprAST]

    def __str__(self) -> str:
        result = f'printf.print_format "{self.format_str}"'
        vals_str = "," + ",".join([f"%{val.name}" for val in self.vals])

        if self.vals:
            result += vals_str
        result += "\n"
        return result


@dataclass
class TensorCastAST(OperationAST):
    source: SSAExprAST
    dest: TensorTypeAST
    out: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.out.name} = tensor.cast %{self.source.name} : {self.source.type} to {self.dest} \n"
        return result


@dataclass
class TensorDimAST(OperationAST):
    source: SSAExprAST
    index: SSAExprAST
    out: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.out.name} = tensor.dim %{self.source.name}, %{self.index.name} : {self.source.type} \n"
        return result


@dataclass
class TensorEmptyAST(OperationAST):
    args: List[SSAExprAST]
    return_val: SSAExprAST

    def __str__(self) -> str:
        args_list = [f"%{arg.name}" for arg in self.args]
        args_str = "(" + ", ".join(args_list) + ")"
        result = f"%{self.return_val.name} = tensor.empty{args_str}: {self.return_val.type} \n"
        return result
