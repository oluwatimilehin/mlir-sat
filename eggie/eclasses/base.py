from __future__ import annotations
from egglog import *


class SSAType(Expr):
    @method()
    def __init__(self): ...

    @method()
    @classmethod
    def tensor(cls, i: i64Like, j: i64Like, t: StringLike) -> SSAType: ...

    @method()
    @classmethod
    def integer(cls, width: i64Like) -> SSAType: ...

    @method()
    @classmethod
    def index(cls) -> SSAType: ...


# class TensorT(Expr):
#     @method()
#     def __init__(self, i: i64Like, j: i64Like, t: StringLike):
#         ...


class SSA(Expr):
    # todo: we have other non-tensor types like i64, f64, etc.
    @method()
    def __init__(self, name: String, type: SSAType): ...


class Block(Expr):
    @method()
    def __init__(self, args: Vec[SSA], ops: Vec[Operation]): ...


class Region(Expr):
    @method()
    def __init__(self, blocks: Vec[Block]): ...


class Operation(Expr):
    @method()
    def __init__(self) -> None: ...
