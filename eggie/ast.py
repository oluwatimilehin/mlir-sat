from dataclasses import dataclass
from typing import List


@dataclass
class ExprAST:
    pass


@dataclass
class TensorTypeAST(ExprAST):
    i: int
    j: int
    t: str


@dataclass
class SSAExprAST(ExprAST):
    name: str
    type: TensorTypeAST


@dataclass
class OperationAST(ExprAST):
    pass


@dataclass
class BlockAST(ExprAST):
    args: List[SSAExprAST]
    ops: List[OperationAST]


@dataclass
class RegionAST(ExprAST):
    blocks: List[BlockAST]


@dataclass
class FuncAST(OperationAST):
    name: str
    args: List[SSAExprAST]
    body: List[OperationAST]
    type: TensorTypeAST


@dataclass
class FuncReturnAST(OperationAST):
    return_val: SSAExprAST


@dataclass
class TensorEmptyAST(OperationAST):
    return_val: SSAExprAST


@dataclass
class LinalgMatmulAST(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST
