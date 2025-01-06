from __future__ import annotations
from egglog import *


class SSAType(Expr):
    @method()
    def __init__(self): ...

    @method()
    @classmethod
    def integer(cls, width: i64Like) -> SSAType: ...

    @method()
    @classmethod
    def index(cls) -> SSAType: ...

    @method()
    @classmethod
    def memref(cls, i: i64Like, j: i64Like, t: StringLike) -> SSAType: ...

    @method()
    @classmethod
    def none(cls) -> SSAType: ...

    @method()
    @classmethod
    def tensor(cls, i: i64Like, j: i64Like, t: StringLike) -> SSAType: ...


class SSA(Expr):
    @method()
    def __init__(self): ...


class SSALiteral(Expr):
    @method()
    @classmethod
    def value(self, name: String, type: SSAType) -> SSA: ...


class Block(Expr):
    @method()
    def __init__(self, args: Vec[SSA], ops: Vec[SSA]): ...


class Region(Expr):
    @method()
    def __init__(self, blocks: Vec[Block]): ...


class Operation(Expr):
    @method()
    def __init__(self) -> None: ...
