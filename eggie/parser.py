from .ast import *
from .lexer import *
from eggie.enode.base import Region
import logging

"""
This Egglog parser is for specific inputs that have the form:
Region(
    Vec[Block](
        Block(
            Vec[SSA].empty(),
            Vec[Operation](
                Function.func(
                    "_2mm_small",
                    Vec[SSA](SSA("arg0", TensorT(73, 77, "i32")), SSA("arg1", TensorT(77, 79, "i32"))),
                    Vec[Operation](
                        Tensor.empty(SSA("0", TensorT(73, 79, "i32"))),
                        Linalg.matmul(SSA("arg0", TensorT(73, 77, "i32")), SSA("arg1", TensorT(77, 79, "i32")),SSA("0", TensorT(73, 79, "i32"), SSA("1", TensorT(73, 79, "i32"))),
                        Function.ret(SSA("1", TensorT(73, 79, "i32"))),
                    ),
                    TensorT(73, 79, "i32"),
                )
            ),
        )
    )
)

TODO:
- make it work for more complex MLIR<->Egglog programs
- replace assertions with proper if/else/exceptions
"""

logger = logging.getLogger(__name__)


class EgglogParser:
    def __init__(self, region: Region):
        self.lexer = Lexer(str(region))

    def parse(self) -> RegionAST:
        token = self.lexer.next_token()

        return self._validate_and_parse(token, EgglogTokenKind.REGION)

    def _validate(self, token: EgglogToken, expected_kind: EgglogTokenKind):
        if token.kind != expected_kind:
            raise ValueError(f"Expected {expected_kind}, received {token}")

    def _validate_and_parse(self, token: EgglogToken, expected_kind: EgglogTokenKind):
        self._validate(token, expected_kind)

        return self._parse_token_(token)

    def _parse_token_(self, token: EgglogToken):
        match token.kind:
            case EgglogTokenKind.REGION:
                return self._parse_region()
            case EgglogTokenKind.BLOCK:
                return self._parse_block()
            case EgglogTokenKind.SSA:
                return self._parse_ssa()
            case EgglogTokenKind.TENSOR:
                return self._parse_tensor()
            case EgglogTokenKind.TENSOR_TYPE:
                return self._parse_tensor_type()
            case EgglogTokenKind.LINALG:
                return self._parse_linalg()
            case EgglogTokenKind.FUNCTION:
                return self._parse_function()
            case EgglogTokenKind.STRING_LITERAL:
                return token.text
            case EgglogTokenKind.INTEGER_LITERAL:
                return int(token.text)
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
            assert self.lexer.next_token().text == "empty"

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

    def _parse_tensor_type(self) -> TensorTypeAST:
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
        return TensorTypeAST(i, j, type)

    def _parse_ssa(self) -> SSAExprAST:
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        name = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
        )

        type = self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.TENSOR_TYPE
        )

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)

        return SSAExprAST(name, type)

    def _parse_tensor(self) -> OperationAST:
        op = self._get_dialect_operation()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
            case "empty":
                ssa = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = TensorEmptyAST(ssa)
            case _:
                raise ValueError(f"Unsupported Tensor operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _parse_linalg(self) -> OperationAST:
        op = self._get_dialect_operation()

        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
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

    def _parse_function(self) -> OperationAST:
        op = self._get_dialect_operation()

        # pop left parenthesis
        self._validate(self.lexer.next_token(), EgglogTokenKind.LEFT_PARENTHESIS)

        val: OperationAST = None
        match op:
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
                    self.lexer.next_token(), EgglogTokenKind.TENSOR_TYPE
                )
                val = FuncAST(name, args, ops, type)
            case "ret":
                type = self._validate_and_parse(
                    self.lexer.next_token(), EgglogTokenKind.SSA
                )
                val = FuncReturnAST(type)
            case _:
                raise ValueError(f"Unsupported Function operation: {op}")

        self._validate(self.lexer.next_token(), EgglogTokenKind.RIGHT_PARENTHESIS)
        return val

    def _get_dialect_operation(self) -> str:
        self._validate(self.lexer.next_token(), EgglogTokenKind.DOT)

        return self._validate_and_parse(
            self.lexer.next_token(), EgglogTokenKind.STRING_LITERAL
        )
