from __future__ import annotations
from egglog import Expr, method
from .base import *


class Func(Expr):
    @method()
    def __init__(self) -> None: ...

    @method()
    @classmethod
    def func(
        cls,
        name: StringLike,
        args: Vec[SSA],
        ops: Vec[Operation],
        return_type: TensorT,
    ) -> Operation: ...

    @method()
    @classmethod
    def ret(cls, return_val: SSA) -> Operation: ...
