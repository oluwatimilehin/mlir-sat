from xdsl.context import MLContext

from xdsl.dialects import (
    arith,
    func,
    linalg,
    printf,
    scf,
    tensor,
)

from xdsl.dialects.builtin import Builtin


class TestUtil:

    @classmethod
    def context(cls) -> MLContext:
        ctx = MLContext()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(printf.Printf)
        ctx.load_dialect(linalg.Linalg)
        ctx.load_dialect(scf.Scf)
        ctx.load_dialect(tensor.Tensor)
        return ctx
