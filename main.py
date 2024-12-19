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
        "bench/2mm_small.mlir"
    )  # todo: make this a command line argument/more generic
    with open(path) as f:
        parser = IRParser(context(), f.read(), name=f"{path}")
        module_op = parser.parse_module()

        printer = Printer(print_generic_format=False)
        printer.print(module_op)

        egglog_region = Converter.to_egglog(module_op)

        egraph = EGraph(save_egglog_string=True)
        # egraph.register(egglog_region)

        egraph.run(1)
        # print(egraph.display())
        # print(egraph.as_egglog_string)

        extracted = egraph.extract(egglog_region)
        print(extracted)

        parser = Parser(Lexer(str(extracted)))
        print(parser.parse())


# todo:
# - take a MLIR file and parse it using xdsl
# - Come up with Egglog classes for MLIR and convert to Egglog
# - Convert back to MLIR
# - At first, I want to confirm that I can go back and forth and the files are equivalent.
