from __future__ import annotations
from egglog import Expr, method
from .base import *


class Func(Expr):
    @method()
    def __init__(self) -> None: ...

    @method()
    @classmethod
    def call(cls, callee: String, args: Vec[SSA], out: SSA) -> Operation: ...

    @method()
    @classmethod
    def func(
        cls,
        name: StringLike,
        args: Vec[SSA],
        ops: Vec[Operation],
        return_type: SSAType,
    ) -> Operation: ...

    @method()
    @classmethod
    def ret(cls, return_val: SSA) -> Operation: ...
