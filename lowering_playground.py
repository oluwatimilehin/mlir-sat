from io import StringIO
from math import prod

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
from xdsl.interpreters.riscv_cf import RiscvCfFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr

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
    dead_code_elimination,
    lower_affine,
    lower_riscv_func,
    mlir_opt,
    reconcile_unrealized_casts,
    riscv_register_allocation,
)


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
    linalg_lowering = (
        [
            mlir_opt.MLIROptPass(
                arguments=[
                    "--convert-linalg-to-loops",
                ],
            )
        ]
        if is_linalg
        else []
    )

    passes = PipelinePass(
        [
            lower_affine.LowerAffinePass(),
            *linalg_lowering,
            canonicalize.CanonicalizePass(),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_print_format_to_riscv_debug.ConvertPrintFormatToRiscvDebugPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            dead_code_elimination.DeadCodeElimination(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
            riscv_register_allocation.RISCVRegisterAllocation(),
            lower_riscv_func.LowerRISCVFunc(),
            convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass(),
        ]
    ).passes

    for pipeline_pass in passes:
        pipeline_pass.apply(ctx, module_op)


def run_linalg():
    mlir_file = "data/converted/linalg.mlir"
    print("---Running Linalg---")
    with open(mlir_file) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{mlir_file}")
        module_op = mlir_parser.parse_module()

        transform(module_op, context(), True)
        # printer = Printer()
        # printer.print(module_op)

        # io = StringIO()
        # riscv.print_assembly(module_op, io)

        # print(f"Value: {io.getvalue()}")

        shape = (2, 2)
        mat_len = prod(shape)

        a_shaped = ShapedArray(
            TypedPtr.new_float64([i + 1 for i in range(mat_len)]), shape
        )
        b_shaped = ShapedArray(
            TypedPtr.new_float64([(i + 1) / 100 for i in range(mat_len)]), shape
        )

        c_shaped = ShapedArray(
            TypedPtr.new_float64([(i + 4) / 100 for i in range(mat_len)] * mat_len),
            shape,
        )

        ab_buffer = ShapedArray(TypedPtr.new_float64([0.0] * mat_len), shape)
        ac_buffer = ShapedArray(TypedPtr.new_float64([0.0] * mat_len), shape)
        out = ShapedArray(TypedPtr.new_float64([0.0] * mat_len), shape)

        print(f"A: {a_shaped}")
        print(f"B: {b_shaped}")
        print(f"C: {c_shaped}")
        print(f"D: {out}")

        riscv_op_counter = OpCounter()
        riscv_interpreter = Interpreter(module_op, listener=riscv_op_counter)

        register_implementations(
            riscv_interpreter, context(), include_wgpu=False, include_onnx=False
        )
        riscv_interpreter.register_implementations(RiscvCfFunctions())
        riscv_interpreter.call_op(
            "distribute",
            (
                a_shaped.data_ptr.raw,
                b_shaped.data_ptr.raw,
                ab_buffer.data_ptr.raw,
                c_shaped.data_ptr.raw,
                ac_buffer.data_ptr.raw,
                out.data_ptr.raw,
            ),
        )

        print(f"Result: {out}")


def run_arith():
    mlir_file = "data/converted/arith.mlir"
    print("---Running Arith---")
    with open(mlir_file) as f:
        mlir_parser = IRParser(context(), f.read(), name=f"{mlir_file}")
        module_op = mlir_parser.parse_module()

        transform(module_op, context(), True)

        # printer = Printer()
        # printer.print(module_op)

        # io = StringIO()
        # riscv.print_assembly(module_op, io)

        # print(f"Value: {io.getvalue()}")

        riscv_op_counter = OpCounter()
        riscv_interpreter = Interpreter(module_op, listener=riscv_op_counter)

        register_implementations(
            riscv_interpreter, context(), include_wgpu=False, include_onnx=False
        )
        riscv_interpreter.register_implementations(RiscvCfFunctions())
        print(f"Result: {riscv_interpreter.call_op("main", ())[0]}")


if __name__ == "__main__":
    run_linalg()
    run_arith()
