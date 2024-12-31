import os
import subprocess
import sys

from io import StringIO
from pathlib import Path

from xdsl.context import MLContext
from xdsl.dialects import (
    arith,
    func,
    linalg,
    memref,
    printf,
    ptr,
    riscv,
    riscv_cf,
    riscv_func,
    riscv_scf,
    scf,
    bufferization,
    tensor,
)
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser as IRParser
from xdsl.passes import PipelinePass
from xdsl.printer import Printer


from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_riscv_scf_to_riscv_cf,
    convert_scf_to_riscv_scf,
)

from xdsl.transforms import (
    canonicalize,
    convert_linalg_to_loops,
    convert_memref_to_ptr,
    dead_code_elimination,
    empty_tensor_to_alloc_tensor,
    convert_ptr_to_riscv,
    lower_affine,
    loop_hoist_memref,
    lower_riscv_func,
    memref_stream_legalize,
    mlir_opt,
    reconcile_unrealized_casts,
    riscv_register_allocation,
)


def context() -> MLContext:
    ctx = MLContext(allow_unregistered=True)
    ctx.load_dialect(arith.Arith)
    ctx.load_attr(bufferization.Bufferization)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(ptr.Ptr)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(riscv_cf.RISCV_Cf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    ctx.load_dialect(riscv_scf.RISCV_Scf)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(tensor.Tensor)
    return ctx


def transform(module_op, ctx):
    passes = PipelinePass(
        [
            mlir_opt.MLIROptPass(
                arguments=[
                    "--linalg-generalize-named-ops",
                    "--mlir-print-op-generic",
                    "--one-shot-bufferize=bufferize-function-boundaries=true",
                    # "--one-shot-bufferize=unknown-type-conversion=identity-layout-map",
                ]
            ),
            empty_tensor_to_alloc_tensor.EmptyTensorToAllocTensorPass(),
            convert_linalg_to_loops.ConvertLinalgToLoopsPass(),
            # canonicalize.CanonicalizePass(),
            lower_affine.LowerAffinePass(),
            # mlir_opt.MLIROptPass(
            #     arguments=["--expand-strided-metadata", "--memref-expand"]
            # ),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_memref_to_ptr.ConvertMemrefToPtr(),
            convert_ptr_to_riscv.ConvertPtrToRiscvPass(),
            dead_code_elimination.DeadCodeElimination(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
            riscv_register_allocation.RISCVRegisterAllocation(),
            canonicalize.CanonicalizePass(),
            lower_riscv_func.LowerRISCVFunc(),
            convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass(),
            # reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
        ]
    ).passes

    printer = Printer()
    for pipeline_pass in passes:
        print(f"Applying: {pipeline_pass.name}")
        pipeline_pass.apply(ctx, module_op)

        # transform(module_op, context())
        # printer.print(module_op)


if __name__ == "__main__":
    mlir_file = "generic.mlir"
    with open(mlir_file) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{mlir_file}")
        module_op = mlir_parser.parse_module()

        printer = Printer()
        transform(module_op, context())
        printer.print(module_op)

        io = StringIO()
        riscv.print_assembly(module_op, io)

        print(f"Value: {io.getvalue()}")
