from .ast import *
from .lexer import *
from eggie.eclasses.base import Region
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
# logging.basicConfig(
#     format="%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in "
#     "function %(funcName)s] %(message)s",
#     datefmt="%Y-%m-%d:%H:%M:%S",
#     level=logging.INFO,
# )

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, region: Region):
        self.lexer = Lexer(str(region))

    def parse(self) -> RegionAST:
        return self.parse_region()

    def parse_region(self) -> RegionAST:
        logger.info("Parsing a region")
        token = self.lexer.next_token()

        if token.kind != EgglogTokenKind.REGION:
            raise ValueError("Expected Region at the start")

        token = self.lexer.next_token()

        if token.kind != EgglogTokenKind.LEFT_PARENTHESIS:
            raise ValueError("Expected ( at start of region")

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.VEC:
            raise ValueError("Expected a vector")

        blocks = self.parse_vector()

        if self.lexer.next_token().kind != EgglogTokenKind.RIGHT_PARENTHESIS:
            raise ValueError("Expected ) at end of region")

        return RegionAST(blocks)

    # todo: maybe remove these checks once I have something working?
    def parse_vector(self) -> List[ExprAST]:
        results: List[ExprAST] = []

        token = self.lexer.next_token()

        if token.kind != EgglogTokenKind.LEFT_SQUARE_BRACKET:
            raise ValueError(f"Expected [ in vec declaration; found: {token}")

        token = self.lexer.next_token()
        if not Lexer.is_keyword(token):
            raise ValueError(
                f"Expected a keyword to specify vector type; found: {token}"
            )

        # pop right square bracket
        token = self.lexer.next_token()

        if token.kind != EgglogTokenKind.RIGHT_SQUARE_BRACKET:
            raise ValueError(f"Expected ] in vec declaration; found: {token}")

        token = self.lexer.next_token()
        if token.kind == EgglogTokenKind.DOT:
            assert self.lexer.next_token().text == "empty"

            # Pop '()'
            self.lexer.next_token()
            self.lexer.next_token()
            return results

        # # pop a left parenthesis
        # self.lexer.next_token()

        while token.kind != EgglogTokenKind.RIGHT_PARENTHESIS:
            token = self.lexer.next_token()
            match token.kind:
                case EgglogTokenKind.BLOCK:
                    results.append(self.parse_block())
                case EgglogTokenKind.SSA:
                    results.append(self.parse_ssa())
                case EgglogTokenKind.TENSOR:
                    results.append(self.parse_tensor())
                case EgglogTokenKind.LINALG:
                    results.append(self.parse_linalg())
                case EgglogTokenKind.FUNCTION:
                    results.append(self.parse_function())
                case EgglogTokenKind.COMMA:
                    continue
                case EgglogTokenKind.RIGHT_PARENTHESIS:
                    continue
                case _:
                    raise ValueError(f"Unknown expression found in Vec: {token}")

        return results

    def parse_block(self) -> BlockAST:
        # pop left parenthesis
        self.lexer.next_token()

        # then we should get two vectors: one for the args and one for the ops.
        #  In this current form, args is always empty, so we technically don't need it but hey
        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.VEC:
            raise ValueError(
                f"Expected a vector as first argument to a Block, received {token}"
            )

        args: List[ExprAST] = self.parse_vector()

        # pop the comma
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.VEC:
            raise ValueError(
                f"Expected a vector as second argument to a Block, received {token}"
            )

        ops: List[ExprAST] = self.parse_vector()

        # pop comma and right parenthesis
        token = self.lexer.next_token()

        if token.kind == EgglogTokenKind.COMMA:
            # then pop right parenthesis; otherwise I assume the right parenthesis is what got popped
            assert self.lexer.next_token().kind == EgglogTokenKind.RIGHT_PARENTHESIS

        return BlockAST(args, ops)

    def parse_tensor_type(self) -> TensorTypeAST:
        # pop left parenthesis
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.NUMBER_LITERAL:
            raise ValueError(
                f"Expected a number literal for tensor type; found: {token}"
            )

        i = int(token.text)

        # pop comma
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.NUMBER_LITERAL:
            raise ValueError(
                f"Expected a number literal for tensor type; found: {token}"
            )

        j = int(token.text)

        # pop comma
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.STRING_LITERAL:
            raise ValueError(
                f"Expected a string literal for tensor type; found: {token}"
            )

        type = token.text

        # pop right parenthesis
        self.lexer.next_token()

        return TensorTypeAST(i, j, type)

    def parse_ssa(self) -> SSAExprAST:
        # pop left parenthesis
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.STRING_LITERAL:
            raise ValueError(f"Expected a string literal for SSA name; found: {token}")

        name = token.text

        # pop comma
        self.lexer.next_token()

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.TENSORT:
            raise ValueError(f"Expected a tensor type for SSA; found: {token}")

        type = self.parse_tensor_type()

        # pop right parenthesis
        self.lexer.next_token()

        return SSAExprAST(name, type)

    def parse_tensor(self) -> OperationAST:
        op = self._get_dialect_operation()

        # pop left parenthesis
        self.lexer.next_token()

        val: OperationAST = None
        match op:
            case "empty":
                self.lexer.next_token()
                ssa = self.parse_ssa()
                val = TensorEmptyAST(ssa)
            case _:
                logger.warn(f"Unregistered operation: {op}")

        # pop right parenthesis
        self.lexer.next_token()
        return val

    def parse_linalg(self) -> OperationAST:
        op = self._get_dialect_operation()

        # pop left parenthesis
        self.lexer.next_token()

        val: OperationAST = None
        match op:
            case "matmul":
                # pop SSA keyword
                self.lexer.next_token()
                x = self.parse_ssa()

                # pop comma and SSA keyword
                self.lexer.next_token()
                self.lexer.next_token()

                y = self.parse_ssa()

                # pop comma and SSA keyword
                self.lexer.next_token()
                self.lexer.next_token()

                out = self.parse_ssa()

                # pop comma and SSA keyword
                self.lexer.next_token()
                self.lexer.next_token()
                return_val = self.parse_ssa()
                val = LinalgMatmulAST(x, y, out, return_val)
            case _:
                logger.warn(f"Unregistered operation: {op}")

        # pop right parenthesis
        self.lexer.next_token()
        return val

    def parse_function(self) -> OperationAST:
        op = self._get_dialect_operation()

        # pop left parenthesis
        self.lexer.next_token()

        val: OperationAST = None
        match op:
            case "func":
                token = self.lexer.next_token()
                if token.kind != EgglogTokenKind.STRING_LITERAL:
                    raise ValueError(
                        "Expected string literal as first function argument"
                    )
                name = token.text

                # pop comma and vec keyword
                self.lexer.next_token()
                self.lexer.next_token()

                args = self.parse_vector()

                # pop comma and vec keyword
                self.lexer.next_token()
                self.lexer.next_token()

                ops = self.parse_vector()

                # pop comma and TensorT keyword
                self.lexer.next_token()
                self.lexer.next_token()

                type = self.parse_tensor_type()
                val = FuncAST(name, args, ops, type)
            case "ret":
                # pop ssa keyword
                self.lexer.next_token()
                type = self.parse_ssa()

                val = FuncReturnAST(type)
            case _:
                logger.warn(f"Unregistered operation: {op}")

        # pop right parenthesis
        self.lexer.next_token()
        return val

    def _get_dialect_operation(self) -> str:
        token = self.lexer.next_token()

        if token.kind != EgglogTokenKind.DOT:
            raise ValueError(f"Expected a dot after dialect name, found: {token}")

        token = self.lexer.next_token()
        if token.kind != EgglogTokenKind.STRING_LITERAL:
            raise ValueError(f"Expected a string literal for dialect, found: {token}")

        return token.text
