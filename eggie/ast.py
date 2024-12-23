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
        return f"tensor<{self.i}x{self.j}x{self.t}>"


@dataclass
class IntegerTypeAST(ExprTypeAST):
    width: int

    def __str__(self) -> str:
        return f"i{self.width}"


@dataclass
class IndexTypeAST(ExprTypeAST):
    name = "index"

    def __str__(self) -> str:
        return self.name


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
    name: str
    type: str

    def __str__(self) -> str:
        result = f"%{self.name} = arith.constant {self.val} : {self.type}"
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
class FuncReturnAST(OperationAST):
    return_val: SSAExprAST

    def __str__(self) -> str:
        result = f"func.return %{self.return_val.name} : {self.return_val.type} \n"
        return result


@dataclass
class LinalgFillAST(OperationAST):
    in_name: str
    in_type: str
    out: SSAExprAST
    return_val: SSAExprAST

    def __str__(self) -> str:
        result = f"%{self.return_val.name} = linalg.fill ins(%{self.in_name} : {self.in_type}) outs(%{self.out.name} : {self.out.type}) -> {self.return_val.type} \n"
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
    index: str
    out: str

    def __str__(self) -> str:
        result = f"%{self.out} = tensor.dim %{self.source.name}, %{self.index} : {self.source.type} \n"
        return result


@dataclass
class TensorEmptyAST(OperationAST):
    args: List[str]
    return_val: SSAExprAST

    def __str__(self) -> str:
        args_list = [f"%{arg}" for arg in self.args]
        args_str = "(" + ", ".join(args_list) + ")"
        result = f"%{self.return_val.name} = tensor.empty{args_str}: {self.return_val.type} \n"
        return result
