from __future__ import annotations
from egglog import *
from .base import *


class Linalg(Expr):
    @method()
    def __init__(self) -> None:
        pass

    @method(cost=1000)
    @classmethod
    def add(self, op1: SSA, op2: SSA, out: SSA, return_val: SSA) -> SSA: ...

    @method()
    @classmethod
    def fill(self, ins: SSA, dest: SSA, return_val: SSA) -> SSA: ...

    @method(cost=2000)
    @classmethod
    def matmul(self, op1: SSA, op2: SSA, out: SSA, return_val: SSA) -> SSA: ...
