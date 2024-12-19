from __future__ import annotations
from egglog import *
from .base import *


class Tensor(Expr):
    @method()
    def __init__(self): ...

    @method()
    @classmethod
    def empty(self, return_val: SSA) -> Operation: ...
