from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer
from eclasses.base import Operation, TensorT, SSA, Region, Block
from eclasses.function import Function
from eclasses.linalg import Linalg
from eclasses.tensor import Tensor
from egglog import Vec
from typing import List

class Converter:

    @classmethod
    def to_egglog(cls, module_op: ModuleOp) -> Region:
        blocks: List[Block] = []

        region = module_op.body
        current_block = region.blocks.first

        while current_block is not None:
            ops : List[Operation] = []
            for op in current_block.ops:
                if isinstance(op, FuncOp):
                    processed_func = cls._process_func_op(op)
                    ops.append(processed_func)

            blocks.append(Block(Vec[SSA](), Vec[Operation](*ops)))
            current_block = current_block.next_block

        return Region(Vec[Block](*blocks))
    

    @classmethod
    def to_mlir(cls, region: Region) -> ModuleOp:
        pass

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
        print(f"function name: {function_name}")
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
                print(f"Processing EmptyOp")
                op_type = cls._to_tensorT(op.tensor.type)
                op_name = printer.print_ssa_value(op.results[0])
                print(f"\nop_name: {op_name}")
                print(f"op_result: {op.results}")
                opsVec.append(Tensor.empty(SSA(op_name, op_type)))

            if isinstance(op, MatmulOp):
                print(f"Processing MatmulOp")
                inputs = op.inputs
                outputs = op.outputs

                print(f"inputs: {inputs}")
                print(f"outputs: {outputs}")

                egg_ins: List[SSA] = []
                egg_outs: List[SSA] = []

                for input in inputs:
                    input_name = printer.print_ssa_value(input)
                    print(f"\n input_name: {input_name}", f"input_type: {input.type}")
                    egg_ins.append(SSA(input_name, cls._to_tensorT(input.type)))

                for output in outputs:
                    output_name = printer.print_ssa_value(output)

                    print(
                        f"\n output_name: {output_name}",
                        f"output_type: {output.type}",
                    )
                    egg_outs.append(SSA(output_name, cls._to_tensorT(output.type)))

                function_output_name = printer.print_ssa_value(op.results[0])
                function_output_type = cls._to_tensorT(op.results[0].type)
                matmul_return_val = SSA(function_output_name, function_output_type)
                opsVec.append(Linalg.matmul(Vec[SSA](*egg_ins), Vec[SSA](*egg_outs), matmul_return_val))

            if isinstance(op, ReturnOp):
                print(f"Processing ReturnOp")
                return_type = cls._to_tensorT(op.arguments.types[0])
                return_arg = printer.print_ssa_value(op.arguments[0])

                opsVec.append(Function.ret(SSA(return_arg, return_type)))

        print(f"opsVec: {opsVec}")
        return Function.func(function_name, Vec[SSA](*argsVec), Vec[Operation](*opsVec), function_return_type)
