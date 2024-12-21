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
    path = Path("bench/2mm_small.mlir")
    with open(path) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{path}")
        module_op = mlir_parser.parse_module()

        egglog_region = Converter.to_egglog(module_op)

        egraph = EGraph(save_egglog_string=True)
        egraph.run(1)
        extracted = egraph.extract(egglog_region)

        print(extracted)
        # egraph.display()

        converted_module_op = Converter.to_mlir(extracted, context())
        assert module_op.is_structurally_equivalent(converted_module_op)

        printer = Printer(print_generic_format=False)
        print("Original module op:")
        printer.print(module_op)

        print("New module op:")
        printer.print(converted_module_op)
