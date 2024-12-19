from pathlib import Path

from xdsl.context import MLContext

from xdsl.dialects import (
    arith,
    func,
    linalg,
    printf,
    scf,
    tensor,
)

from xdsl.dialects.builtin import Builtin, ModuleOp

from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from util.converter import Converter

from egglog import *
from eggie.parser import Parser
from eggie.lexer import Lexer


def context() -> MLContext:
    ctx = MLContext()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(tensor.Tensor)
    return ctx


if __name__ == "__main__":
    path = Path(
        "bench/10mm.mlir"
    )  # todo: make this a command line argument/more generic
    with open(path) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{path}")
        module_op = mlir_parser.parse_module()

        # printer = Printer(print_generic_format=False)
        # printer.print(module_op)

        egglog_region = Converter.to_egglog(module_op)

        egraph = EGraph(save_egglog_string=True)
        egraph.run(1)
        extracted = egraph.extract(egglog_region)

        converted_module_op = Converter.to_mlir(extracted, context())
        assert module_op.is_structurally_equivalent(converted_module_op)
