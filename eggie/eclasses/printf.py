from __future__ import annotations
from egglog import Expr, method
from .base import *


class Printf(Expr):
    @method()
    def __init__(self) -> None: ...

    @method()
    @classmethod
    def print_format(cls, format_str: String, vals: Vec[SSA]) -> Operation: ...
