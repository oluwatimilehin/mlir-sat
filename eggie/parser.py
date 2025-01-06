import logging

from .ast import *
from .lexer import *
from eggie.enodes.base import Region

logger = logging.getLogger(__name__)


class EgglogParser:
    def __init__(self, region: Region):
        self.lexer = Lexer(str(region))
        self.vars = {}

    def parse(self) -> RegionAST:
        token = self.lexer.next_token()

        while token.kind != EgglogTokenKind.REGION:
            if token.kind != EgglogTokenKind.VARIABLE_NAME:
                raise ValueError(f"Expected variable name, received {token}")

            variable_name = token.text
            self._validate(self.lexer.next_token(), EgglogTokenKind.EQUALS)

            self.vars[variable_name] = self._parse_token_(self.lexer.next_token())

            token = self.lexer.next_token()

        return self._validate_and_parse(token, EgglogTokenKind.REGION)

    def _validate(self, token: EgglogToken, expected_kind: EgglogTokenKind):
        if (
            expected_kind == EgglogTokenKind.SSA
            and not (
                token.kind == EgglogTokenKind.SSA_LITERAL or Lexer.is_dialect(token)
            )
        ) and token.kind != expected_kind:
            raise ValueError(f"Expected {expected_kind}, received {token}")

    def _validate_and_parse(self, token: EgglogToken, expected_kind: EgglogTokenKind):
        if token.kind == EgglogTokenKind.VARIABLE_NAME:
            return self.vars[token.text]

        self._validate(token, expected_kind)

        return self._parse_token_(token)

    def _parse_token_(self, token: EgglogToken):
        match token.kind:
            case EgglogTokenKind.REGION:
                return self._parse_region()
            case EgglogTokenKind.BLOCK:
                return self._parse_block()
            case EgglogTokenKind.SSA_LITERAL:
                return self._parse_ssa_literal()
            case EgglogTokenKind.ARITH:
                return self._parse_arith()
            case EgglogTokenKind.FUNC:
                return self._parse_function()
            case EgglogTokenKind.MEMREF:
                return self._parse_memref()
            case EgglogTokenKind.SSA_TYPE:
                return self._parse_ssa_type()
            case EgglogTokenKind.TENSOR:
                return self._parse_tensor()
            case EgglogTokenKind.LINALG:
                return self._parse_linalg()
            case EgglogTokenKind.PRINTF:
                return self._parse_printf()
            case EgglogTokenKind.STRING_LITERAL:
                return token.text
            case EgglogTokenKind.INTEGER_LITERAL:
                return int(token.text)
            case EgglogTokenKind.VARIABLE_NAME:
                return self.vars[token.text]
            case EgglogTokenKind.VEC:
                return self._parse_vector()
            case _:
                raise ValueError(f"Unsupported token: {token}")

    def _parse_region(self) -> RegionAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        blocks = self._validate_and_parse(self.lexer.next_token(), EgglogTokenKind.VEC)

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)

        return RegionAST(blocks)

    def _parse_vector(self) -> List[ExprAST]:
        results: List[ExprAST] = []

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_SQUARE_BRACKET)

        token = self.lexer.next_token()
        if not Lexer.is_keyword(token):
            raise ValueError(
                f"Expected a keyword to specify vector type; found: {token}"
            )

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_SQUARE_BRACKET)

        token = self.lexer.next_token()  # Can be a left parenthesis or a dot if empty

        # empty vector handling
        if token.kind == EgglogTokenKind.DOT:
            text = self._validate_and_parse(
                self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
            )
            if text != "empty":
                raise ValueError(f"Expected 'empty' call for Vec; found: {text}")

            # Pop '()'
            self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)
            self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
            return results

        while token.kind != EgglogTokenKind.RIGHT_PARENTHESIS:
            token = self.lexer.next_token()

            if token.kind == EgglogTokenKind.RIGHT_PARENTHESIS:
                break

            results.append(self._parse_token_(token))

        return results

    def _parse_block(self) -> BlockAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        #  In this current form, args is always empty, so we technically don't need it but hey
        args = self._validate_and_parse(self.lexer.next_token(), EgglogTokenKind.VEC)
        ops = self._validate_and_parse(self.lexer.next_token(), EgglogTokenKind.VEC)

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)

        return BlockAST(args, ops)

    # TODO: separate class for type parsing
    def _parse_ssa_type(self) -> ExprTypeAST:
        name = self._get_class_fn()

        match name:
            case "tensor":
                return self._parse_shaped_type(name)
            case "integer":
                return self._parse_integer_type()
            case "index":
                return self._parse_index_type()
            case "memref":
                return self._parse_shaped_type(name)
            case "none":
                return self._parse_none_type()
            case _:
                raise ValueError(f"Unsupported SSA type: {name}")

    def _parse_index_type(self) -> IndexTypeAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)
        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return IndexTypeAST()

    def _parse_integer_type(self) -> IntegerTypeAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        width = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.INTEGER_LITERAL
        )

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return IntegerTypeAST(width)

    def _parse_none_type(self) -> IndexTypeAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)
        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return NoneTypeAST()

    def _parse_shaped_type(self, name) -> ShapedTypeAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        i = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.INTEGER_LITERAL
        )

        j = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.INTEGER_LITERAL
        )

        type = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
        )

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return (
            TensorTypeAST(i, j, type) if name == "tensor" else MemrefTypeAST(i, j, type)
        )

    def _parse_ssa_literal(self) -> SSAExprAST:
        op = self._get_class_fn()
        if not op == "value":
            raise ValueError("Expected value keyword for SSALiteral")

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        name = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
        )

        type = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.SSA_TYPE
        )

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)

        return SSAExprAST(name, type)

    # Dialects
    def _parse_arith(self) -> OperationAST:
        op = self._get_class_fn()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
            case "constant":
                res = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.INTEGER_LITERAL
                )

                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = ArithConstantAst(res, SSAExprAST(out.name, out.type))
            case "addi":
                x = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                y = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = ArithAddiAst(x, y, out)
            case "muli":
                x = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                y = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = ArithMuliAst(x, y, out)         
            case _:
                raise ValueError(f"Unsupported Arith operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_tensor(self) -> OperationAST:
        op = self._get_class_fn()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
            case "cast":
                source = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                dest = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA_TYPE
                )
                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = TensorCastAST(source, dest, out)

            case "dim":
                source = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                index = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = TensorDimAST(source, index, out)
            case "empty":
                args = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.VEC
                )

                result = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = TensorEmptyAST(args, result)
            case _:
                raise ValueError(f"Unsupported Tensor operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_printf(self) -> OperationAST:
        op = self._get_class_fn()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None

        match op:
            case "print_format":
                format_str = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
                )
                format_vals = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.VEC
                )

                val = PrintFormatAST(format_str, format_vals)
            case _:
                raise ValueError(f"Unsupported Printf operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_linalg(self) -> OperationAST:
        op = self._get_class_fn()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
            case "add":
                x = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                y = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                return_val = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = LinalgAddAST(x, y, out, return_val)
            case "fill":
                scalar = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                return_val = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = LinalgFillAST(scalar, out, return_val)
            case "matmul":
                x = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                y = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                return_val = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = LinalgMatmulAST(x, y, out, return_val)
            case _:
                raise ValueError(f"Unsupported Linalg operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_memref(self) -> OperationAST:
        op = self._get_class_fn()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None

        match op:
            case "alloc":
                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = MemrefAllocAST(out)
            case "dealloc":
                arg = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = MemrefDeallocAST(arg)
            case _:
                raise ValueError(f"Unsupported memref operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_function(self) -> OperationAST:
        op = self._get_class_fn()

        # pop left parenthesis
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
            case "call":
                callee = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
                )

                args = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.VEC
                )

                out = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )

                val = FuncCallAST(callee, args, out)

            case "func":
                name = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
                )

                args = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.VEC
                )

                ops = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.VEC
                )

                type = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA_TYPE
                )
                val = FuncAST(name, args, ops, type)
            case "ret":
                val = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                type = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA_TYPE
                )
                val = FuncReturnAST(val, type)
            case _:
                raise ValueError(f"Unsupported Function operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _get_class_fn(self) -> str:
        self._validate(self.lexer.next_token(), EgglogTokenKind.DOT)

        return self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
        )
