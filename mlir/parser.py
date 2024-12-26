from eggie.eclasses.base import Operation, SSAType, SSA, Region, Block
from eggie.eclasses.arith import Arith
from eggie.eclasses.func import Func
from eggie.eclasses.linalg import Linalg
from eggie.eclasses.tensor import Tensor
from eggie.eclasses.printf import Printf

from egglog import Vec, String

from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerType, IndexType
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.func import FuncOp, ReturnOp, CallOp
from xdsl.dialects.linalg import MatmulOp, FillOp
from xdsl.dialects.tensor import EmptyOp, DimOp, CastOp
from xdsl.dialects.printf import PrintFormatOp
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

    def _get_egg_type(self, mlir_type) -> SSAType:
        if isinstance(mlir_type, TensorType):
            return self._to_egg_tensor_type(mlir_type)

        if isinstance(mlir_type, IntegerType):
            return SSAType.integer(mlir_type.width.data)

        if isinstance(mlir_type, IndexType):
            return SSAType.index()

        raise ValueError(f"Unsupported mlir type provided: {mlir_type}")

    def _to_egg_tensor_type(self, tensor_type: TensorType) -> SSAType:
        shape = tensor_type.get_shape()
        element_type = tensor_type.get_element_type()

        if isinstance(element_type, IntegerType):
            element_type = f"i{element_type.width.data}"
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return SSAType.tensor(shape[0], shape[1], element_type)

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
            case "printf":
                return self._process_print(op)
            case _:
                raise ValueError(f"Unsupported dialect for operation: {op}")

    def _process_arith(self, op: IRDLOperation) -> Operation:
        match op.name:
            case ConstantOp.name:
                val = op.value.value.data
                out_type = self._get_egg_type(op.value.type)
                out_name = self._ssa_name_getter.get_ssa_name(op.results[0])
                return Arith.constant(val, SSA(out_name, out_type))
            case _:
                raise ValueError(f"Unsupported arith operation: {op}")

    def _process_func(self, op: IRDLOperation) -> Operation:
        match op.name:
            case CallOp.name:
                return self._process_call_op(op)
            case FuncOp.name:
                return self._process_func_op(op)
            case ReturnOp.name:
                return_type = self._get_egg_type(op.arguments.types[0])
                return_arg = self._ssa_name_getter.get_ssa_name(op.arguments[0])
                return Func.ret(SSA(return_arg, return_type))
            case _:
                raise ValueError(f"Unsupported func operation: {op}")

    def _process_linalg(self, op: IRDLOperation) -> Operation:
        match op.name:
            case MatmulOp.name:
                return self._process_matmul_op(op)
            case FillOp.name:
                return self._process_fill_op(op)
            case _:
                raise ValueError(f"Unsupported linalg operation: {op}")

    def _process_tensor(self, op: IRDLOperation) -> Operation:
        match op.name:
            case EmptyOp.name:
                return self._process_empty_op(op)
            case DimOp.name:
                return self._process_dim_op(op)
            case CastOp.name:
                return self._process_cast_op(op)
            case _:
                raise ValueError(f"Unsupported tensor operation: {op}")

    def _process_print(self, op: IRDLOperation) -> Operation:
        match op.name:
            case PrintFormatOp.name:
                format_str = str(op.format_str).strip('"')
                print(f"string: {format_str}")
                vals_list = []
                for op in op.operands:
                    vals_list.append(SSA(op.name_hint, self._get_egg_type(op.type)))

                vals_vec = Vec[SSA](*vals_list) if vals_list else Vec[SSA].empty()

                return Printf.print_format(format_str, vals_vec)
            case _:
                raise ValueError(f"Unsupported printf operation: {op}")

    # TODO: Would be good to move these to dialect-specific classes
    def _process_cast_op(self, op: CastOp) -> Operation:
        source_name = self._ssa_name_getter.get_ssa_name(op.source)
        source: SSA = SSA(source_name, self._get_egg_type(op.source.type))

        dest: SSAType = self._get_egg_type(op.dest.type)

        out_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        out_type = self._get_egg_type(op.results[0].type)

        return Tensor.cast(source, dest, SSA(out_name, out_type))

    def _process_dim_op(self, op: DimOp) -> Operation:
        source_name = self._ssa_name_getter.get_ssa_name(op.source)
        source_type = self._get_egg_type(op.source.type)

        index_name = self._ssa_name_getter.get_ssa_name(op.index)
        index_type = self._get_egg_type(op.index.type)

        out_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        out_type = self._get_egg_type(op.results[0].type)

        return Tensor.dim(
            SSA(source_name, source_type),
            SSA(index_name, index_type),
            SSA(out_name, out_type),
        )

    def _process_empty_op(self, op: EmptyOp) -> Operation:
        op_type = self._get_egg_type(op.tensor.type)
        op_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        argsList: List[SSA] = []

        for operand in op.operands:
            argsList.append(SSA(operand.name_hint, self._get_egg_type(operand.type)))

        args_vec = Vec[SSA](*argsList) if argsList else Vec[SSA].empty()
        return Tensor.empty(args_vec, SSA(op_name, op_type))

    def _process_fill_op(self, op: FillOp) -> Operation:
        input_name = self._ssa_name_getter.get_ssa_name(op.inputs[0])
        input_type = self._get_egg_type(op.inputs[0].type)

        output_name = self._ssa_name_getter.get_ssa_name(op.outputs[0])
        output_type = self._get_egg_type(op.outputs[0].type)

        ret_val_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        ret_val_type = self._get_egg_type(op.results[0].type)

        return Linalg.fill(
            SSA(input_name, input_type),
            SSA(output_name, output_type),
            SSA(ret_val_name, ret_val_type),
        )

    def _process_matmul_op(self, op: MatmulOp) -> Operation:
        inputs = op.inputs
        outputs = op.outputs

        egg_ins: List[SSA] = []
        egg_outs: List[SSA] = []

        for input in inputs:
            input_name = self._ssa_name_getter.get_ssa_name(input)
            egg_ins.append(SSA(input_name, self._get_egg_type(input.type)))

        for output in outputs:
            output_name = self._ssa_name_getter.get_ssa_name(output)
            egg_outs.append(SSA(output_name, self._get_egg_type(output.type)))

        matmul_output_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        matmul_output_type = self._get_egg_type(op.results[0].type)
        matmul_return_val = SSA(matmul_output_name, matmul_output_type)

        return Linalg.matmul(egg_ins[0], egg_ins[1], egg_outs[0], matmul_return_val)

    def _process_func_op(self, func_op: FuncOp) -> Operation:
        function_name = func_op.properties["sym_name"].data

        function_return_type = self._get_egg_type(func_op.function_type.outputs.data[0])

        argsVec: List[SSA] = []

        for arg in func_op.args:
            arg_name = self._ssa_name_getter.get_ssa_name(arg)
            arg_type = self._get_egg_type(arg.type)

            argsVec.append(SSA(arg_name, arg_type))

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

    def _process_call_op(self, op: CallOp) -> Operation:
        callee = str(op.callee)

        args: List[SSA] = []

        for arg in op.arguments:
            arg_name = arg.name_hint
            arg_type = self._get_egg_type(arg.type)

            args.append(SSA(arg_name, arg_type))

        out_name = self._ssa_name_getter.get_ssa_name(op.results[0])
        out_type = self._get_egg_type(op.results[0].type)

        return Func.call(callee, Vec[SSA](*args), SSA(out_name, out_type))
