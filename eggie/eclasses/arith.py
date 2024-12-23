from __future__ import annotations
from egglog import Expr, method
from .base import *


class Arith(Expr):
    @method()
    def __init__(self) -> None: ...

    @method()
    @classmethod
    def constant(
        cls, val: i64Like, name: StringLike, type: StringLike
    ) -> Operation: ...
