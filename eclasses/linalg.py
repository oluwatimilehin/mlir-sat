from __future__ import annotations
from egglog import *
from .base import *


class Linalg(Expr):
    @method()
    def __init__(self) -> None:
        pass

    @method()
    @classmethod
    def matmul(self, ins: Vec[SSA], outs: Vec[SSA], return_val: SSA) -> Operation: ...
