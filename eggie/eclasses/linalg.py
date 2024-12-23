from __future__ import annotations
from egglog import *
from .base import *


class Linalg(Expr):
    @method()
    def __init__(self) -> None:
        pass

    @method()
    @classmethod
    def fill(self, scalar: SSA, out: SSA, return_val: SSA) -> Operation: ...

    @method()
    @classmethod
    def matmul(self, op1: SSA, op2: SSA, out: SSA, return_val: SSA) -> Operation: ...
