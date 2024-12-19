from eggie.eclasses.base import Operation, TensorT, SSA, Region, Block
from eggie.eclasses.function import Function
from eggie.eclasses.linalg import Linalg
from eggie.eclasses.tensor import Tensor
from eggie.parser import Parser

from egglog import Vec

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import SSAValue
from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from dataclasses import dataclass, field
from typing import List


@dataclass(eq=False, repr=False)
class SSANameGetter:
    """
    This class is a hack to deal with SSAValues not always having name hints.
    xDSL does some trickery in the `Printer` class to still print the right names for those values, but to avoid any printing, I've copied the logic in the xDSL printer class here and remove the print line
    """

    _ssa_values: dict[SSAValue, str] = field(default_factory=dict, init=False)
    """
    maps SSA Values to their "allocated" names
    """
    _ssa_names: list[dict[str, int]] = field(
        default_factory=lambda: [dict()], init=False
    )
    _next_valid_name_id: list[int] = field(default_factory=lambda: [0], init=False)

    @property
    def ssa_names(self):
        return self._ssa_names[-1]

    def _get_new_valid_name_id(self):
        self._next_valid_name_id[-1] += 1
        return str(self._next_valid_name_id[-1] - 1)

    def get_ssa_name(self, value: SSAValue):
        """
        Get an SSA value. This assigns a name to the value if the value
        does not have one in the current context.
        If the value has a name hint, it will use it as a prefix, and otherwise assign
        a number as the name. Numbers are assigned in order.

        Returns the name used for  the value.
        """
        if value in self._ssa_values:
            name = self._ssa_values[value]
        elif value.name_hint:
            curr_ind = self.ssa_names.get(value.name_hint, 0)
            suffix = f"_{curr_ind}" if curr_ind != 0 else ""
            name = f"{value.name_hint}{suffix}"
            self._ssa_values[value] = name
            self.ssa_names[value.name_hint] = curr_ind + 1
        else:
            name = self._get_new_valid_name_id()
            self._ssa_values[value] = name

        return name


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

        ssa_name_getter = SSANameGetter()

        for op in block.ops:
            if isinstance(op, EmptyOp):
                op_type = cls._to_tensorT(op.tensor.type)
                op_name = ssa_name_getter.get_ssa_name(op.results[0])
                opsVec.append(Tensor.empty(SSA(op_name, op_type)))

            if isinstance(op, MatmulOp):
                inputs = op.inputs
                outputs = op.outputs

                egg_ins: List[SSA] = []
                egg_outs: List[SSA] = []

                for input in inputs:
                    input_name = ssa_name_getter.get_ssa_name(input)
                    egg_ins.append(SSA(input_name, cls._to_tensorT(input.type)))

                for output in outputs:
                    output_name = ssa_name_getter.get_ssa_name(output)
                    egg_outs.append(SSA(output_name, cls._to_tensorT(output.type)))

                matmul_output_name = ssa_name_getter.get_ssa_name(op.results[0])
                matmul_output_type = cls._to_tensorT(op.results[0].type)
                matmul_return_val = SSA(matmul_output_name, matmul_output_type)
                opsVec.append(
                    Linalg.matmul(
                        egg_ins[0], egg_ins[1], egg_outs[0], matmul_return_val
                    )
                )

            if isinstance(op, ReturnOp):
                return_type = cls._to_tensorT(op.arguments.types[0])
                return_arg = ssa_name_getter.get_ssa_name(op.arguments[0])

                opsVec.append(Function.ret(SSA(return_arg, return_type)))

        return Function.func(
            function_name,
            Vec[SSA](*argsVec),
            Vec[Operation](*opsVec),
            function_return_type,
        )
