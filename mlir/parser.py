from dataclasses import dataclass, field
from typing import List, Tuple

from xdsl.dialects.builtin import (
    ModuleOp,
    Float64Type,
    IntegerType,
    IndexType,
    MemRefType,
    ShapedType,
    TensorType,
)
from xdsl.dialects.arith import AddiOp, ConstantOp, DivSIOp, MuliOp
from xdsl.dialects.func import FuncOp, ReturnOp, CallOp
from xdsl.dialects.linalg import AddOp, FillOp, MatmulOp
from xdsl.dialects.memref import AllocOp, DeallocOp
from xdsl.dialects.tensor import EmptyOp, DimOp, CastOp
from xdsl.dialects.printf import PrintFormatOp
from xdsl.irdl.operations import IRDLOperation
from xdsl.ir import SSAValue

from egglog import Vec

from eggie.enodes.base import SSAType, SSALiteral, SSA, Region, Block
from eggie.enodes.arith import Arith
from eggie.enodes.func import Func
from eggie.enodes.linalg import Linalg
from eggie.enodes.memref import Memref
from eggie.enodes.tensor import Tensor
from eggie.enodes.printf import Printf


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

    def get_name(self, value: SSAValue):
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


class BlockSSAManager:
    # TODO: I don't like the design of this class or how it's being used, but it'll work for now
    def __init__(self) -> None:
        self._ssa_name_to_val = {}
        self._ssa_is_standalone = {}
        self._ssa_name_getter = SSANameGetter()

    def insert(self, name: str, val: SSA):
        self._ssa_name_to_val[name] = val
        self._ssa_is_standalone[name] = True

    def get(self, val) -> Tuple[str, SSA]:
        # TODO: I don't like that this returns a tuple
        name = self._ssa_name_getter.get_name(val)
        type = self.get_egg_type(val.type)

        if name in self._ssa_name_to_val:
            self._ssa_is_standalone[name] = False
            return name, self._ssa_name_to_val[name]

        return name, SSALiteral.value(name, type)

    def get_standalone_ops(self) -> SSA:
        res: List[SSA] = []
        for name, is_standalone in self._ssa_is_standalone.items():
            if is_standalone:
                res.append(self._ssa_name_to_val[name])

        return res

    def get_name(self, val: SSA) -> str:
        return self._ssa_name_getter.get_name(val)

    def get_egg_type(self, mlir_type) -> SSAType:
        if isinstance(mlir_type, ShapedType):
            return self._to_egg_shaped_type(mlir_type)

        if isinstance(mlir_type, IntegerType):
            return SSAType.integer(mlir_type.width.data)

        if isinstance(mlir_type, IndexType):
            return SSAType.index()

        raise ValueError(f"Unsupported mlir type provided: {mlir_type}")

    def _to_egg_shaped_type(self, ranked_type: TensorType | MemRefType) -> SSAType:
        shape = ranked_type.get_shape()
        element_type = ranked_type.get_element_type()

        if isinstance(element_type, IntegerType):
            element_type = f"i{element_type.width.data}"
        elif isinstance(element_type, Float64Type):
            element_type = f"f64"
        else:
            raise ValueError(f"Unsupported element type: {element_type}")

        return (
            SSAType.tensor(shape[0], shape[1], element_type)
            if isinstance(ranked_type, TensorType)
            else SSAType.memref(shape[0], shape[1], element_type)
        )

    def reset(self):
        self._ssa_name_to_val.clear()
        self._ssa_name_getter.reset()


# TODO: Clean up this class; What can I move to separate classes?
class MLIRParser:
    def __init__(self, module_op: ModuleOp) -> None:
        self._module_op = module_op
        self._block_ssa_manager = None

    def parse(self) -> Region:
        blocks: List[Block] = []

        region = self._module_op.body
        current_block = region.blocks.first

        while current_block is not None:
            ops: List[SSA] = []
            for op in current_block.ops:
                ops.append(self._process_op(op))

            blocks.append(Block(Vec[SSA].empty(), Vec[SSA](*ops)))
            self._block_ssa_manager.reset()
            current_block = current_block.next_block

        return Region(Vec[Block](*blocks))

    def _process_op(self, op: IRDLOperation) -> SSA:
        match op.dialect_name():
            case "arith":
                return self._process_arith(op)
            case "func":
                return self._process_func(op)
            case "linalg":
                return self._process_linalg(op)
            case "memref":
                return self._process_memref(op)
            case "tensor":
                return self._process_tensor(op)
            case "printf":
                return self._process_print(op)
            case _:
                raise ValueError(f"Unsupported dialect for operation: {op}")

    def _process_arith(self, op: IRDLOperation) -> SSA:
        match op.name:
            case ConstantOp.name:
                val = op.value.value.data
                name, out = self._block_ssa_manager.get(op.results[0])

                res = Arith.constant(val, out)
                self._block_ssa_manager.insert(name, res)
                return res
            case AddiOp.name:
                return self._process_arith_addi(op)
            case DivSIOp.name:
                return self._process_arith_divsi(op)
            case MuliOp.name:
                return self._process_arith_muli(op)
            case _:
                raise ValueError(f"Unsupported arith operation: {op}")

    def _process_func(self, op: IRDLOperation) -> SSA:
        match op.name:
            case CallOp.name:
                return self._process_call_op(op)
            case FuncOp.name:
                return self._process_func_op(op)
            case ReturnOp.name:
                return self._process_return_op(op)
            case _:
                raise ValueError(f"Unsupported func operation: {op}")

    def _process_linalg(self, op: IRDLOperation) -> SSA:
        match op.name:
            case AddOp.name:
                return self._process_linalg_add_op(op)
            case FillOp.name:
                return self._process_fill_op(op)
            case MatmulOp.name:
                return self._process_matmul_op(op)
            case _:
                raise ValueError(f"Unsupported linalg operation: {op}")

    def _process_memref(self, op: IRDLOperation) -> SSA:
        match op.name:
            case AllocOp.name:
                return self._process_alloc_op(op)
            case DeallocOp.name:
                return self._process_dealloc_op(op)
            case _:
                raise ValueError(f"Unsupported memref operation: {op}")

    def _process_tensor(self, op: IRDLOperation) -> SSA:
        match op.name:
            case EmptyOp.name:
                return self._process_empty_op(op)
            case DimOp.name:
                return self._process_dim_op(op)
            case CastOp.name:
                return self._process_cast_op(op)
            case _:
                raise ValueError(f"Unsupported tensor operation: {op}")

    def _process_print(self, op: IRDLOperation) -> SSA:
        match op.name:
            case PrintFormatOp.name:
                format_str = str(op.format_str).strip('"')
                vals_list = []
                for op in op.operands:
                    vals_list.append(self._block_ssa_manager.get(op)[1])

                vals_vec = Vec[SSA](*vals_list) if vals_list else Vec[SSA].empty()

                return Printf.print_format(format_str, vals_vec)
            case _:
                raise ValueError(f"Unsupported printf operation: {op}")

    # TODO: Would be good to move these to dialect-specific classes
    def _process_arith_addi(self, op: AddiOp) -> SSA:
        op1 = self._block_ssa_manager.get(op.operands[0])[1]
        op2 = self._block_ssa_manager.get(op.operands[1])[1]
        name, out = self._block_ssa_manager.get(op.results[0])
        res = Arith.addi(op1, op2, out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_arith_divsi(self, op: AddiOp) -> SSA:
        op1 = self._block_ssa_manager.get(op.operands[0])[1]
        op2 = self._block_ssa_manager.get(op.operands[1])[1]
        name, out = self._block_ssa_manager.get(op.results[0])
        res = Arith.divsi(op1, op2, out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_arith_muli(self, op: AddiOp) -> SSA:
        op1 = self._block_ssa_manager.get(op.operands[0])[1]
        op2 = self._block_ssa_manager.get(op.operands[1])[1]
        name, out = self._block_ssa_manager.get(op.results[0])
        res = Arith.muli(op1, op2, out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_cast_op(self, op: CastOp) -> SSA:
        _, source = self._block_ssa_manager.get(op.source)

        dest: SSAType = self._block_ssa_manager.get_egg_type(op.dest.type)

        name, out = self._block_ssa_manager.get(op.results[0])
        res = Tensor.cast(source, dest, out)

        self._block_ssa_manager.insert(name, res)
        return res

    def _process_dim_op(self, op: DimOp) -> SSA:
        _, source = self._block_ssa_manager.get(op.source)
        _, index = self._block_ssa_manager.get(op.index)
        name, out = self._block_ssa_manager.get(op.results[0])

        res = Tensor.dim(
            source,
            index,
            out,
        )

        self._block_ssa_manager.insert(name, res)
        return res

    def _process_empty_op(self, op: EmptyOp) -> SSA:
        argsList: List[SSA] = []
        for operand in op.operands:
            argsList.append(self._block_ssa_manager.get(operand)[1])

        args_vec = Vec[SSA](*argsList) if argsList else Vec[SSA].empty()
        name, out = self._block_ssa_manager.get(op.results[0])

        res = Tensor.empty(args_vec, out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_linalg_add_op(self, op: MatmulOp) -> SSA:
        inputs = op.inputs
        outputs = op.outputs

        egg_ins: List[SSA] = []
        egg_outs: List[SSA] = []

        for input in inputs:
            egg_ins.append(self._block_ssa_manager.get(input)[1])

        for output in outputs:
            egg_outs.append(self._block_ssa_manager.get(output)[1])

        if not op.results:  # No return types when operating on memref types
            name, out = (
                self._block_ssa_manager.get_name(op.outputs[0]),
                SSALiteral.value("", SSAType.none()),
            )
        else:
            name, out = self._block_ssa_manager.get(op.results[0])

        res = Linalg.add(egg_ins[0], egg_ins[1], egg_outs[0], out)

        self._block_ssa_manager.insert(name, res)
        return res

    def _process_fill_op(self, op: FillOp) -> SSA:
        _, input = self._block_ssa_manager.get(op.inputs[0])

        _, dest = self._block_ssa_manager.get(op.outputs[0])

        name, out = self._block_ssa_manager.get(op.results[0])
        res = Linalg.fill(input, dest, out)

        self._block_ssa_manager.insert(name, res)
        return res

    def _process_matmul_op(self, op: MatmulOp) -> SSA:
        inputs = op.inputs
        outputs = op.outputs

        egg_ins: List[SSA] = []
        egg_outs: List[SSA] = []

        for input in inputs:
            egg_ins.append(self._block_ssa_manager.get(input)[1])

        for output in outputs:
            egg_outs.append(self._block_ssa_manager.get(output)[1])

        if not op.results:  # No return types when operating on memref types
            name, out = (
                self._block_ssa_manager.get_name(op.outputs[0]),
                SSALiteral.value("", SSAType.none()),
            )
        else:
            name, out = self._block_ssa_manager.get(op.results[0])

        res = Linalg.matmul(egg_ins[0], egg_ins[1], egg_outs[0], out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_alloc_op(self, op: AllocOp) -> SSA:
        name, out = self._block_ssa_manager.get(op.results[0])
        res = Memref.alloc(out)
        self._block_ssa_manager.insert(name, res)
        return res

    def _process_dealloc_op(self, op: AllocOp) -> SSA:
        name, arg = self._block_ssa_manager.get(op.operands[0])
        res = Memref.dealloc(arg)
        self._block_ssa_manager.insert(f"dealloc_{name}", res)
        return res

    def _process_func_op(self, func_op: FuncOp) -> SSA:
        self._block_ssa_manager = BlockSSAManager()
        function_name = func_op.properties["sym_name"].data

        function_return_type = (
            self._block_ssa_manager.get_egg_type(func_op.function_type.outputs.data[0])
            if func_op.function_type.outputs.data
            else SSAType.none()
        )

        argsVec: List[SSA] = []

        for arg in func_op.args:
            argsVec.append(self._block_ssa_manager.get(arg)[1])

        # Sample programs are in the form of a region with a single block
        block = func_op.regions[0].blocks.first
        for op in block.ops:
            self._process_op(op)

        opsVec: List[SSA] = self._block_ssa_manager.get_standalone_ops()

        self._block_ssa_manager.reset()
        return Func.func(
            function_name,
            Vec[SSA](*argsVec),
            Vec[SSA](*opsVec),
            function_return_type,
        )

    def _process_call_op(self, op: CallOp) -> SSA:
        callee = str(op.callee)

        args: List[SSA] = []

        for arg in op.arguments:
            args.append(self._block_ssa_manager.get(arg)[1])

        name, out = self._block_ssa_manager.get(op.results[0])
        res = Func.call(callee, Vec[SSA](*args), out)

        self._block_ssa_manager.insert(name, res)

        return res

    def _process_return_op(self, op: ReturnOp) -> SSA:
        _, return_val = self._block_ssa_manager.get(op.arguments[0])
        res = Func.ret(
            return_val,
            self._block_ssa_manager.get_egg_type(op.arguments.types[0]),
        )
        self._block_ssa_manager.insert("func.ret", res)
        return res
