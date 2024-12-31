from __future__ import annotations
from egglog import Expr, method
from .base import *


class Func(Expr):
    @method()
    def __init__(self) -> None: ...

    @method()
    @classmethod
    def call(cls, callee: String, args: Vec[SSA], out: SSA) -> SSA: ...

    @method()
    @classmethod
    def func(
        cls, name: StringLike, args: Vec[SSA], ops: Vec[SSA], return_type: SSAType
    ) -> SSA: ...

    @method()
    @classmethod
    def ret(cls, return_val: SSA, return_type: SSAType) -> SSA: ...
