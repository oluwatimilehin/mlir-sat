from dataclasses import dataclass
from typing import List


@dataclass
class ExprAST:
    pass


@dataclass
class ExprTypeAST(ExprAST):
    pass


@dataclass
class ShapedTypeAST(ExprTypeAST):
    pass


@dataclass
class MemrefTypeAST(ShapedTypeAST):
    i: int
    j: int
    t: str

    def __str__(self) -> str:
        i_str = f"{self.i}" if self.i != -1 else "?"
        j_str = f"{self.j}" if self.j != -1 else "?"
        return f"memref<{i_str}x{j_str}x{self.t}>"


@dataclass
class TensorTypeAST(ShapedTypeAST):
    i: int
    j: int
    t: str

    def __str__(self) -> str:
        i_str = f"{self.i}" if self.i != -1 else "?"
        j_str = f"{self.j}" if self.j != -1 else "?"
        return f"tensor<{i_str}x{j_str}x{self.t}>"


@dataclass
class NoneTypeAST(ExprTypeAST):
    def __str__(self) -> str:
        return "()"


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
    args: List[ExprAST]
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
    out: ExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return None

    def __str__(self) -> str:
        result = f"%{self.out.name} = arith.constant {self.val} : {self.out.type} \n"
        return result


@dataclass
class ArithAddiAst(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return [arg for arg in [self.x, self.y] if isinstance(arg, OperationAST)]

    def __str__(self) -> str:
        result = f"%{self.out.name} = arith.addi %{self.x.name}, %{self.y.name} : {self.out.type} \n"
        return result


@dataclass
class ArithMuliAst(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return [arg for arg in [self.x, self.y] if isinstance(arg, OperationAST)]

    def __str__(self) -> str:
        result = f"%{self.out.name} = arith.muli %{self.x.name}, %{self.y.name} : {self.out.type} \n"
        return result


@dataclass
class FuncAST(OperationAST):
    name: str
    args: List[ExprAST]
    body: List[OperationAST]
    type: TensorTypeAST

    def _get_dependencies(self, op: OperationAST, accum):
        if not op.dependencies:
            accum.append(op)
            return

        for dependency in op.dependencies:
            self._get_dependencies(dependency, accum)

        accum.append(op)

    def __str__(self) -> str:
        args_list = [f"%{ssa.name} : {ssa.type}" for ssa in self.args]
        args_str = "(" + ", ".join(args_list) + ")"
        result = f"func.func @{self.name}{args_str} -> {self.type}" + " { \n"

        all_ops = []

        for op in self.body:
            accum = []
            self._get_dependencies(op, accum)
            all_ops += accum

        # The accumulator should ideally be a map so that we don't have this logic to eliminate duplicates, but it's a little tricky to do with instructions that don't have return types, so I'm doing this hack below
        printed = set()
        for op in all_ops:
            op_str = str(op)
            if not op_str in printed:
                printed.add(op_str)
                result += op_str

        result += "} \n"
        return result


@dataclass
class FuncCallAST(OperationAST):
    callee: str
    args: List[ExprAST]
    out: ExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return [op for op in self.args + [self.out] if isinstance(op, OperationAST)]

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
    type: ExprTypeAST

    @property
    def name(self):
        return f"return:{self.return_val}"

    @property
    def dependencies(self):
        return [self.return_val] if isinstance(self.return_val, OperationAST) else None

    def __str__(self) -> str:
        result = f"func.return %{self.return_val.name} : {self.return_val.type} \n"
        return result


@dataclass
class LinalgAddAST(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST
    return_val: SSAExprAST

    @property
    def name(self):
        return self.return_val.name if not self.is_void else self.out.name

    @property
    def type(self):
        return self.return_val.type if not self.is_void else self.out.type

    @property
    def is_void(self):
        return isinstance(self.return_val.type, NoneTypeAST)

    @property
    def dependencies(self):
        return [op for op in [self.x, self.y, self.out] if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f"linalg.add ins(%{self.x.name}, %{self.y.name} : {self.x.type}, {self.y.type}) outs(%{self.out.name} : {self.out.type})"

        if not self.is_void:
            result = f"%{self.return_val.name} = {result}  -> {self.return_val.type}"

        result += "\n"
        return result


@dataclass
class LinalgFillAST(OperationAST):
    scalar: SSAExprAST
    out: SSAExprAST
    return_val: SSAExprAST

    @property
    def name(self):
        return self.return_val.name

    @property
    def type(self):
        return self.return_val.type

    @property
    def dependencies(self):
        return [op for op in [self.scalar, self.out] if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f"%{self.return_val.name} = linalg.fill ins(%{self.scalar.name} : {self.scalar.type}) outs(%{self.out.name} : {self.out.type}) -> {self.return_val.type} \n"
        return result


@dataclass
class LinalgMatmulAST(OperationAST):
    x: SSAExprAST
    y: SSAExprAST
    out: SSAExprAST
    return_val: SSAExprAST

    @property
    def name(self):
        return self.return_val.name if not self.is_void else self.out.name

    @property
    def type(self):
        return self.return_val.type if not self.is_void else self.out.type

    @property
    def is_void(self):
        return isinstance(self.return_val.type, NoneTypeAST)

    @property
    def dependencies(self):
        return [op for op in [self.x, self.y, self.out] if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f"linalg.matmul ins(%{self.x.name}, %{self.y.name} : {self.x.type}, {self.y.type}) outs(%{self.out.name} : {self.out.type})"

        if not self.is_void:
            result = f"%{self.return_val.name} = {result}  -> {self.return_val.type}"

        result += "\n"
        return result


@dataclass
class MemrefAllocAST(OperationAST):
    out: SSAExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return None

    def __str__(self) -> str:
        result = f"%{self.name} = memref.alloc() : {self.type}\n"
        return result


@dataclass
class MemrefDeallocAST(OperationAST):
    arg: SSAExprAST

    @property
    def name(self):
        return self.arg.name

    @property
    def type(self):
        return NoneTypeAST

    @property
    def dependencies(self):
        return [self.arg] if isinstance(self.arg, OperationAST) else None

    def __str__(self) -> str:
        result = f"memref.dealloc %{self.arg.name} : {self.arg.type}\n"
        return result


@dataclass
class PrintFormatAST(OperationAST):
    format_str: str
    vals: List[ExprAST]

    @property
    def dependencies(self):
        return [op for op in self.vals if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f'printf.print_format "{self.format_str}"'
        vals_str = "," + ",".join([f"%{val.name}" for val in self.vals])

        if self.vals:
            result += vals_str
        result += "\n"
        return result


@dataclass
class TensorCastAST(OperationAST):
    source: ExprAST
    dest: TensorTypeAST
    out: ExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return [op for op in [self.source, self.dest] if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f"%{self.out.name} = tensor.cast %{self.source.name} : {self.source.type} to {self.dest} \n"
        return result


@dataclass
class TensorDimAST(OperationAST):
    source: ExprAST
    index: ExprAST
    out: SSAExprAST

    @property
    def name(self):
        return self.out.name

    @property
    def type(self):
        return self.out.type

    @property
    def dependencies(self):
        return [op for op in [self.source, self.index] if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        result = f"%{self.out.name} = tensor.dim %{self.source.name}, %{self.index.name} : {self.source.type} \n"
        return result


@dataclass
class TensorEmptyAST(OperationAST):
    args: List[ExprAST]
    return_val: SSAExprAST

    @property
    def name(self):
        return self.return_val.name

    @property
    def type(self):
        return self.return_val.type

    @property
    def dependencies(self):
        return [op for op in self.args if isinstance(op, OperationAST)]

    def __str__(self) -> str:
        args_list = [f"%{arg.name}" for arg in self.args]
        args_str = "(" + ", ".join(args_list) + ")"
        result = f"%{self.return_val.name} = tensor.empty{args_str}: {self.return_val.type} \n"
        return result
