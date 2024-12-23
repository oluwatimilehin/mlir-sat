from __future__ import annotations
from egglog import *
from .base import *


class Linalg(Expr):
    @method()
    def __init__(self) -> None:
        pass

    @method()
    @classmethod
    # TODO: I should have a separate class for the scalar name/type combo here and in arith.constant
    def fill(
        self, scalar_name: String, scalar_type: String, out: SSA, return_val: SSA
    ) -> Operation: ...

    @method()
    @classmethod
    def matmul(self, op1: SSA, op2: SSA, out: SSA, return_val: SSA) -> Operation: ...
