from __future__ import annotations
from egglog import *


class TensorT(Expr):
    @method()
    def __init__(self, i: i64Like, j: i64Like, t: StringLike):
        pass


class SSA(Expr):
    # todo: we have other non-tensor types like i64, f64, etc.
    @method()
    def __init__(self, name: StringLike, type: TensorT):
        pass


class Block(Expr):
    @method()
    def __init__(self, args: Vec[SSA], ops: Vec[Operation]):
        pass


class Region(Expr):
    @method()
    def __init__(self, blocks: Vec[Block]):
        pass


class Operation(Expr):
    @method()
    def __init__(self) -> None:
        pass
