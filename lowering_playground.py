from io import StringIO

from xdsl.context import MLContext
from xdsl.dialects import (
    arith,
    bufferization,
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
    tensor,
)
from xdsl.dialects.builtin import Builtin
from xdsl.interpreter import Interpreter, OpCounter
from xdsl.interpreters import register_implementations
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.parser import Parser as IRParser
from xdsl.passes import PipelinePass
from xdsl.printer import Printer


from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_print_format_to_riscv_debug,
    convert_riscv_scf_to_riscv_cf,
    convert_scf_to_riscv_scf,
)

from xdsl.transforms import (
    canonicalize,
    convert_linalg_to_loops,
    dead_code_elimination,
    lower_affine,
    lower_riscv_func,
    mlir_opt,
    reconcile_unrealized_casts,
    riscv_register_allocation,
)


def emulate_riscv(program: str):
    from xdsl.interpreters.riscv_emulator import run_riscv

    run_riscv(program, unlimited_regs=False, verbosity=3)


def context() -> MLContext:
    ctx = MLContext()
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


def transform(module_op, ctx, is_linalg=False):
    # passes_list =

    linalg_lowering = (
        [
            mlir_opt.MLIROptPass(
                arguments=["--allow-unregistered-dialect", "--convert-linalg-to-loops"],
            )
        ]
        if is_linalg
        else []
    )

    passes = PipelinePass(
        [
            canonicalize.CanonicalizePass(),
            lower_affine.LowerAffinePass(),
            *linalg_lowering,
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_print_format_to_riscv_debug.ConvertPrintFormatToRiscvDebugPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            dead_code_elimination.DeadCodeElimination(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
            canonicalize.CanonicalizePass(),
            riscv_register_allocation.RISCVRegisterAllocation(),
            canonicalize.CanonicalizePass(),
            lower_riscv_func.LowerRISCVFunc(),
            convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass(),
        ]
    ).passes

    for pipeline_pass in passes:
        print(f"Applying: {pipeline_pass.name}")
        pipeline_pass.apply(ctx, module_op)


if __name__ == "__main__":
    mlir_file = "data/mlir/linalg.mlir"
    with open(mlir_file) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{mlir_file}")
        module_op = mlir_parser.parse_module()

        printer = Printer()
        transform(module_op, context(), True)
        printer.print(module_op)

        io = StringIO()
        riscv.print_assembly(module_op, io)

        print(f"Value: {io.getvalue()}")

        riscv_op_counter = OpCounter()
        riscv_interpreter = Interpreter(module_op, listener=riscv_op_counter)

        register_implementations(
            riscv_interpreter, context(), include_wgpu=False, include_onnx=False
        )

        riscv_interpreter.call_op("main", ())
