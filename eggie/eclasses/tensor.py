from __future__ import annotations
from egglog import *
from .base import *


class Tensor(Expr):
    @method()
    def __init__(self): ...

    @method()
    @classmethod
    def empty(self, args: Vec[SSA], out: SSA) -> Operation: ...

    @method()
    @classmethod
    def dim(self, source: SSA, index: SSA, out: SSA) -> Operation: ...

    @method()
    @classmethod
    def cast(self, source: SSA, dest: SSAType, out: SSA) -> Operation: ...
