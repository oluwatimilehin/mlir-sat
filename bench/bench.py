import copy
import os
import statistics
import time


from collections import namedtuple
from math import prod
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from egglog import *

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
from xdsl.dialects.builtin import Builtin, ModuleOp
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
    canonicalize as canonicalization,
    dead_code_elimination,
    lower_affine,
    lower_riscv_func,
    mlir_opt,
    reconcile_unrealized_casts,
    riscv_register_allocation,
)

from converter import Converter

from eggie.rewrites import rewrites_ruleset


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


def lower(module_op, ctx, canonicalize=True, is_linalg=False):
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

    canonicalize_pass = (
        [
            mlir_opt.MLIROptPass(
                arguments=[
                    "--canonicalize",
                ],
            ),
            canonicalization.CanonicalizePass(),
        ]
        if canonicalize
        else []
    )

    passes = PipelinePass(
        [
            lower_affine.LowerAffinePass(),
            *linalg_lowering,
            *canonicalize_pass,
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


def run_linalg(riscv_interpreter):
    shape = (2, 2)
    mat_len = prod(shape)

    a_shaped = ShapedArray(TypedPtr.new_float64([i + 1 for i in range(mat_len)]), shape)
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

    return out


def run_arith(riscv_interpreter):
    return riscv_interpreter.call_op("main", ())[0]


dialect_to_runner = {"linalg": run_linalg, "arith": run_arith}
BenchmarkResult = namedtuple("BenchmarkResult", "num_ops mean_latency median_latency")


def benchmark(
    ctx: MLContext, baseline_mlir: ModuleOp, converted_mlir: ModuleOp, dialect: str
):
    canonicalized = copy.deepcopy(baseline_mlir)
    eqsat_and_canonicalized = copy.deepcopy(converted_mlir)

    is_linalg = dialect == "linalg"

    lower(baseline_mlir, ctx, False, is_linalg)
    lower(canonicalized, ctx, True, is_linalg)
    lower(converted_mlir, ctx, False, is_linalg)
    lower(eqsat_and_canonicalized, ctx, True, is_linalg)

    name_to_module_op = {
        "baseline": baseline_mlir,
        "canonicalized": canonicalized,
        "eqsat": converted_mlir,
        "eqsat+canonicalized": eqsat_and_canonicalized,
    }

    runner = dialect_to_runner[dialect]
    name_to_results = {}

    for name, module_op in name_to_module_op.items():
        riscv_op_counter = OpCounter()
        riscv_interpreter = Interpreter(module_op, listeners=(riscv_op_counter,))

        register_implementations(riscv_interpreter, context())

        times = []
        # warmup
        print(f"Result for {name}: {runner(riscv_interpreter)}")
        num_ops = sum(dict(riscv_op_counter.ops).values())

        for _ in range(10):
            start_time = time.perf_counter()
            runner(riscv_interpreter)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        mean = statistics.mean(times)
        median = statistics.median(times)

        name_to_results[name] = BenchmarkResult(num_ops, mean, median)

    return name_to_results


def visualize(dir: Path, data: Dict[str, Dict[str, BenchmarkResult]]):
    median_latency_data = {}
    num_ops_data = {}

    for dialect, results in data.items():
        median_latency_data[dialect] = {
            category: results[category].median_latency
            for category in results
            if category != "baseline"
        }
        num_ops_data[dialect] = {
            category: results[category].num_ops
            for category in results
            if category != "baseline"
        }

    median_latency_speedup = {}
    for dialect, latencies in median_latency_data.items():
        baseline_latency = data[dialect]["baseline"].median_latency
        median_latency_speedup[dialect] = {
            category: baseline_latency / latency
            for category, latency in latencies.items()
        }

    num_ops_reduction = {}
    for dialect, num_ops in num_ops_data.items():
        baseline_num_ops = data[dialect]["baseline"].num_ops
        num_ops_reduction[dialect] = {
            category: baseline_num_ops / num_ops
            for category, num_ops in num_ops.items()
        }

    #
    categories = ["canonicalized", "eqsat", "eqsat+canonicalized"]
    dialects = list(median_latency_speedup.keys())
    x = np.arange(len(dialects))
    width = 0.2

    # --- Mean Latency Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, category in enumerate(categories):
        speedups = [median_latency_speedup[dialect][category] for dialect in dialects]
        # Offset the bar positions for each category
        bar_positions = x + (i - len(categories) / 2) * width + width / 2
        ax.bar(bar_positions, speedups, width, label=category, alpha=0.8)

    # Add a horizontal baseline at 1 (logarithmic baseline)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Baseline")

    ax.set_ylim(bottom=1)
    ax.set_xticks(x)  # Center x-axis ticks on dialect positions
    ax.set_xticklabels(dialects)
    ax.set_xlabel("Benchmark", fontweight="bold", fontsize=12)
    ax.set_ylabel("Median Latency Speedup", fontweight="bold", fontsize=12)
    ax.set_title(
        "Median Latency Speedup Per Benchmark",
        fontweight="bold",
        fontsize=14,
    )
    ax.legend(title="Optimization Passes", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{dir}/median_latency_speedup.jpg", dpi=300)
    plt.close(fig)

    # --- Num Ops Reduction Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, category in enumerate(categories):
        # Extract reductions for the current category across all dialects
        reductions = [num_ops_reduction[dialect][category] for dialect in dialects]
        # Offset the bar positions for each category
        bar_positions = x + (i - len(categories) / 2) * width + width / 2
        ax.bar(bar_positions, reductions, width, label=category, alpha=0.8)

    # Add a horizontal baseline at 1 (logarithmic baseline)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Baseline")

    # Aesthetics
    # ax.set_yscale("log")
    ax.set_ylim(bottom=1)  # Ensure the y-axis starts at 1
    ax.set_xticks(x)  # Center x-axis ticks on dialect positions
    ax.set_xticklabels(dialects)
    ax.set_xlabel("Benchmark", fontweight="bold", fontsize=12)
    ax.set_ylabel("Num Ops Reduction", fontweight="bold", fontsize=12)
    ax.set_title("Num Ops Reduction Per Benchmark", fontweight="bold", fontsize=14)
    ax.legend(title="Optimization Passes", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{dir}/num_ops_reduction.jpg", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    mlir_dir = f"{current_dir.parent}/data/mlir"

    results_dir = f"{current_dir}/results/{int(time.time())}"
    os.makedirs(results_dir)

    mlir_files = [f for f in Path(mlir_dir).iterdir() if f.suffix == ".mlir"]
    ctx = context()

    results = {}

    for mlir_file in mlir_files:
        dialect = Path(mlir_file).stem
        print(f"Processing: {dialect}")

        with open(mlir_file) as f:
            mlir_parser = IRParser(ctx, f.read(), name=f"{mlir_file}")
            module_op = mlir_parser.parse_module()

            egglog_region = Converter.to_egglog(module_op)

            egraph = EGraph(save_egglog_string=True)
            egglog_region = egraph.let("expr", egglog_region)
            egraph.run(1000, ruleset=rewrites_ruleset)

            print(f"Extracting expression")
            extracted = egraph.extract(egglog_region)

            dialect_path = f"{results_dir}/{dialect}"
            if not os.path.exists(dialect_path):
                os.makedirs(dialect_path)

            converted_module_op = Converter.to_mlir(extracted, ctx)
            converted_mlir_file = f"{dialect_path}/{dialect}.mlir"

            with open(converted_mlir_file, "w") as f:
                printer = Printer(stream=f)
                printer.print(converted_module_op)

            converted_egg_file = f"{dialect_path}/{dialect}.egg"
            with open(converted_egg_file, "w") as f:
                f.write(str(extracted))

            benchmark_res = benchmark(ctx, module_op, converted_module_op, dialect)
            results = results | {dialect: benchmark_res}

    print(results)
    visualize(results_dir, results)
