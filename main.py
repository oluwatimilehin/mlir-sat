from pathlib import Path

from egglog import *

from xdsl.context import MLContext

from xdsl.dialects import (
    arith,
    func,
    linalg,
    memref,
    printf,
    scf,
    tensor,
)

from xdsl.dialects.builtin import Builtin

from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from converter import Converter

from eggie.rewrites import rewrites_ruleset


def context() -> MLContext:
    ctx = MLContext()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(tensor.Tensor)
    return ctx


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = f"{current_dir}/data"

    models_path = f"{data_dir}/mlir"
    eggs_path = f"{data_dir}/eggs"
    converted_path = f"{data_dir}/converted"

    mlir_files = [f for f in Path(models_path).iterdir() if f.suffix == ".mlir"]

    for mlir_file in mlir_files:
        file_name = Path(mlir_file).stem
        print(f"Processing mlir: {file_name}")

        with open(mlir_file) as f:
            mlir_parser = IRParser(context(), f.read(), name=f"{mlir_file}")
            module_op = mlir_parser.parse_module()

            egglog_region = Converter.to_egglog(module_op)
            egg_file_name = f"{eggs_path}/{file_name}.egg"
            with open(egg_file_name, "w") as f:
                f.write(str(egglog_region))

            egraph = EGraph(save_egglog_string=True)
            egglog_region = egraph.let("expr", egglog_region)
            egraph.run(10, ruleset=rewrites_ruleset)

            # egraph.display()

            print(f"Extracting expression")
            extracted = egraph.extract(egglog_region)

            converted_egg_file = f"{converted_path}/{file_name}.egg"
            with open(converted_egg_file, "w") as f:
                f.write(str(extracted))

            converted_module_op = Converter.to_mlir(extracted, context())
            converted_mlir_file = f"{converted_path}/{file_name}.mlir"

            with open(converted_mlir_file, "w") as f:
                printer = Printer(stream=f)
                printer.print(converted_module_op)

            # assert module_op.is_structurally_equivalent(converted_module_op)
