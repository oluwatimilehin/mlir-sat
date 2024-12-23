from eggie.eclasses.base import Operation, TensorT, SSA, Region, Block
from eggie.eclasses.arith import Arith
from eggie.eclasses.func import Func
from eggie.eclasses.linalg import Linalg
from eggie.eclasses.tensor import Tensor

from egglog import Vec, String

from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerType, IndexType
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.irdl.operations import IRDLOperation
from xdsl.ir import SSAValue


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

    def reset(self):
        self._ssa_names: list[dict[str, int]] = field(
            default_factory=lambda: [dict()], init=False
        )
        self._next_valid_name_id: list[int] = field(
            default_factory=lambda: [0], init=False
        )


class MLIRParser:
    def __init__(self, module_op: ModuleOp) -> None:
        self._module_op = module_op
        self._ssa_name_getter = SSANameGetter()

    def parse(self) -> Region:
        blocks: List[Block] = []

        region = self._module_op.body
        current_block = region.blocks.first

        while current_block is not None:
            ops: List[Operation] = []
            for op in current_block.ops:
                ops.append(self._process_op(op))

            blocks.append(Block(Vec[SSA].empty(), Vec[Operation](*ops)))
            self._ssa_name_getter.reset()
            current_block = current_block.next_block

        return Region(Vec[Block](*blocks))

    def _to_tensorT(self, type: TensorType) -> TensorT:
        shape = type.get_shape()
        element_type = type.get_element_type()

        if isinstance(element_type, IntegerType):
            element_type = f"i{element_type.width.data}"
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return TensorT(shape[0], shape[1], element_type)

    def _process_op(self, op: IRDLOperation) -> Operation:
        match op.dialect_name():
            case "arith":
                return self._process_arith(op)
            case "func":
                return self._process_func(op)
            case "linalg":
                return self._process_linalg(op)
            case "tensor":
                return self._process_tensor(op)
            case _:
                raise ValueError(f"Unsupported dialect for operation: {op}")

    def _process_arith(self, op: IRDLOperation) -> Operation:
        match op.name:
            case ConstantOp.name:
                val = op.value.value.data
                out_type = str(op.value.type)
                out_name = self._ssa_name_getter.get_ssa_name(op.results[0])
                return Arith.constant(val, out_name, out_type)

            case "-":
                raise ValueError(f"Unsupported arith operation: {op}")

    def _process_func(self, op: IRDLOperation) -> Operation:
        match op.name:
            case FuncOp.name:
                return self._process_func_op(op)
            case ReturnOp.name:
                return_type = self._to_tensorT(op.arguments.types[0])
                return_arg = self._ssa_name_getter.get_ssa_name(op.arguments[0])
                return Func.ret(SSA(return_arg, return_type))
            case "_":
                raise ValueError(f"Unsupported func operation: {op}")

    def _process_linalg(self, op: IRDLOperation) -> Operation:
        match op.name:
            case MatmulOp.name:
                return self._process_matmul_op(op)
            case _:
                raise ValueError(f"Unsupported linalg operation: {op}")

    def _process_tensor(self, op: IRDLOperation) -> Operation:
        match op.name:
            case EmptyOp.name:
                op_type = self._to_tensorT(op.tensor.type)
                op_name = self._ssa_name_getter.get_ssa_name(op.results[0])
                argsList: List[String] = []

                for operand in op.operands:
                    if not isinstance(operand.type, IndexType):
                        raise ValueError(
                            f"Unsupported type received for tensor.empty operand: {operand.type}"
                        )

                    argsList.append(String(operand.name_hint))

                args_vec = Vec[String](*argsList) if argsList else Vec[String].empty()
                return Tensor.empty(args_vec, SSA(op_name, op_type))
            case "_":
                raise ValueError(f"Unsupported func operation: {op}")

    def _process_matmul_op(self, op: MatmulOp) -> Operation:
        inputs = op.inputs
        outputs = op.outputs

        egg_ins: List[SSA] = []
        egg_outs: List[SSA] = []

        for input in inputs:
            input_name = self._ssa_name_getter.get_ssa_name(input)
            egg_ins.append(SSA(input_name, self._to_tensorT(input.type)))

        for output in outputs:
            output_name = self._ssa_name_getter.get_ssa_name(output)
            egg_outs.append(SSA(output_name, self._to_tensorT(output.type)))

        matmul_output_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        matmul_output_type = self._to_tensorT(op.results[0].type)
        matmul_return_val = SSA(matmul_output_name, matmul_output_type)

        return Linalg.matmul(egg_ins[0], egg_ins[1], egg_outs[0], matmul_return_val)

    def _process_func_op(self, func_op: FuncOp) -> Operation:
        function_name = func_op.properties["sym_name"].data

        function_return_type = self._to_tensorT(func_op.function_type.outputs.data[0])

        argsVec: List[SSA] = []

        for arg in func_op.args:
            arg_name = arg.name_hint
            arg_type = arg.type

            if isinstance(arg_type, TensorType):
                tensor_type = self._to_tensorT(arg_type)
            else:
                raise ValueError(f"Unsupported argument type: {arg_type}")

            argsVec.append(SSA(arg_name, tensor_type))

        opsVec: List[Operation] = []

        # Sample programs are in the form of a region with a single block
        block = func_op.regions[0].blocks.first
        for op in block.ops:
            opsVec.append(self._process_op(op))

        return Func.func(
            function_name,
            Vec[SSA](*argsVec),
            Vec[Operation](*opsVec),
            function_return_type,
        )
