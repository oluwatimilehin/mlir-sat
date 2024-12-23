from __future__ import annotations
from egglog import *
from .base import *


class Tensor(Expr):
    @method()
    def __init__(self): ...

    @method()
    @classmethod
    # supports args of type 'index'
    def empty(self, args: Vec[String], return_val: SSA) -> Operation: ...
