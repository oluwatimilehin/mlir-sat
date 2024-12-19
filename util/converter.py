from eclasses.base import Operation, TensorT, SSA, Region, Block
from eclasses.function import Function
from eclasses.linalg import Linalg
from eclasses.tensor import Tensor
from egglog import Vec
from eggie.parser import Parser

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from typing import List


class Converter:
    @classmethod
    def to_egglog(cls, module_op: ModuleOp) -> Region:
        blocks: List[Block] = []

        region = module_op.body
        current_block = region.blocks.first

        while current_block is not None:
            ops: List[Operation] = []
            for op in current_block.ops:
                if isinstance(op, FuncOp):
                    processed_func = cls._process_func_op(op)
                    ops.append(processed_func)

            blocks.append(Block(Vec[SSA](), Vec[Operation](*ops)))
            current_block = current_block.next_block

        return Region(Vec[Block](*blocks))

    @classmethod
    def to_mlir(cls, region: Region, context: MLContext) -> ModuleOp:
        egglog_parser = Parser(region)
        region_ast = egglog_parser.parse()
        mlir_parser = IRParser(context, str(region_ast))
        return mlir_parser.parse_module(str(region_ast))

    @classmethod
    def _to_tensorT(cls, type: TensorType) -> TensorT:
        shape = type.get_shape()
        element_type = type.get_element_type()

        if isinstance(element_type, IntegerType):
            element_type = f"i{element_type.width.data}"
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return TensorT(shape[0], shape[1], element_type)

    @classmethod
    def _process_func_op(cls, func_op: FuncOp) -> Operation:
        function_name = func_op.properties["sym_name"].data

        function_return_type = cls._to_tensorT(func_op.function_type.outputs.data[0])
        func_op_args = func_op.args

        argsVec: List[SSA] = []

        for arg in func_op_args:
            arg_name = arg.name_hint
            arg_type = arg.type

            if isinstance(arg_type, TensorType):
                tensor_type = cls._to_tensorT(arg_type)
            else:
                raise ValueError(f"Unsupported argument type: {arg_type}")

            argsVec.append(SSA(arg_name, tensor_type))

        opsVec: List[Operation] = []

        # Sample programs are in the form of a region with a single block
        block = func_op.regions[0].blocks.first

        printer = Printer()

        # I'm using `printer.print_ssa_value` to get the name of the SSA value, because not all SSA values return a name and I haven't yet figured out how to get it otherwise
        # TODO: figure out how to get the name of the SSA value without using the printer
        for op in block.ops:
            if isinstance(op, EmptyOp):
                op_type = cls._to_tensorT(op.tensor.type)
                op_name = printer.print_ssa_value(op.results[0])
                opsVec.append(Tensor.empty(SSA(op_name, op_type)))

            if isinstance(op, MatmulOp):
                inputs = op.inputs
                outputs = op.outputs


                egg_ins: List[SSA] = []
                egg_outs: List[SSA] = []

                for input in inputs:
                    input_name = printer.print_ssa_value(input)
                    egg_ins.append(SSA(input_name, cls._to_tensorT(input.type)))

                for output in outputs:
                    output_name = printer.print_ssa_value(output)
                    egg_outs.append(SSA(output_name, cls._to_tensorT(output.type)))

                function_output_name = printer.print_ssa_value(op.results[0])
                function_output_type = cls._to_tensorT(op.results[0].type)
                matmul_return_val = SSA(function_output_name, function_output_type)
                opsVec.append(
                    Linalg.matmul(
                        egg_ins[0], egg_ins[1], egg_outs[0], matmul_return_val
                    )
                )

            if isinstance(op, ReturnOp):
                return_type = cls._to_tensorT(op.arguments.types[0])
                return_arg = printer.print_ssa_value(op.arguments[0])

                opsVec.append(Function.ret(SSA(return_arg, return_type)))

        return Function.func(
            function_name,
            Vec[SSA](*argsVec),
            Vec[Operation](*opsVec),
            function_return_type,
        )
