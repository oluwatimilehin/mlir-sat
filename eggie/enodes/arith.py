from __future__ import annotations
from egglog import Expr, method
from .base import *


class Arith(Expr):
    @method()
    def __init__(self) -> None: ...

    @method(cost=100)
    @classmethod
    def addi(cls, op1: SSA, op2: SSA, out: SSA) -> SSA: ...

    @method()
    @classmethod
    def constant(cls, val: i64Like, out: SSA) -> SSA: ...

    @method(cost=200)
    @classmethod
    def muli(cls, op1: SSA, op2: SSA, out: SSA) -> SSA: ...

    @method(cost=100)
    @classmethod
    def shli(cls, lhs: SSA, rhs: SSA, out: SSA) -> SSA: ...
