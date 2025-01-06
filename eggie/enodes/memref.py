from __future__ import annotations
from egglog import *
from .base import *


class Memref(Expr):
    @method()
    @classmethod
    def alloc(self, out: SSA) -> SSA: ...

    @method()
    @classmethod
    def dealloc(self, arg: SSA) -> SSA: ...
