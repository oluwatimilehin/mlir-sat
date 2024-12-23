from enum import Enum, auto
from collections import namedtuple


class EgglogTokenKind(Enum):
    ARITH = auto()
    BLOCK = auto()
    COMMA = auto()
    DOT = auto()
    EOF = auto()
    EQUALS = auto()
    FUNC = auto()
    LEFT_PARENTHESIS = auto()
    LEFT_SQUARE_BRACKET = auto()
    LINALG = auto()
    INTEGER_LITERAL = auto()
    OPERATION = auto()
    REGION = auto()
    RIGHT_PARENTHESIS = auto()
    RIGHT_SQUARE_BRACKET = auto()
    SSA = auto()
    STRING_LITERAL = auto()
    TENSOR = auto()
    SSA_TYPE = auto()
    TENSOR_TYPE = auto()
    VARIABLE_NAME = auto()
    VEC = auto()


dialect_to_token = {
    "Arith": EgglogTokenKind.ARITH,
    "Func": EgglogTokenKind.FUNC,
    "Linalg": EgglogTokenKind.LINALG,
    "Tensor": EgglogTokenKind.TENSOR,
}
chars_to_token = {
    "[": EgglogTokenKind.LEFT_SQUARE_BRACKET,
    "]": EgglogTokenKind.RIGHT_SQUARE_BRACKET,
    "(": EgglogTokenKind.LEFT_PARENTHESIS,
    ")": EgglogTokenKind.RIGHT_PARENTHESIS,
    ",": EgglogTokenKind.COMMA,
    ".": EgglogTokenKind.DOT,
    "=": EgglogTokenKind.EQUALS,
}

keyword_to_token = {
    "Region": EgglogTokenKind.REGION,
    "Block": EgglogTokenKind.BLOCK,
    "Vec": EgglogTokenKind.VEC,
    "String": EgglogTokenKind.STRING_LITERAL,
    "SSA": EgglogTokenKind.SSA,
    "Operation": EgglogTokenKind.OPERATION,
    "SSAType": EgglogTokenKind.SSA_TYPE,
    "TensorT": EgglogTokenKind.TENSOR_TYPE,
}


str_to_token = dialect_to_token | chars_to_token | keyword_to_token
EgglogToken = namedtuple("Token", ["kind", "text"])


class Lexer:
    def __init__(self, input: str):
        self.input = input
        self.index = 0

    @classmethod
    def is_keyword(cls, token: EgglogToken) -> bool:
        return keyword_to_token.get(token.text) != None

    @classmethod
    def is_dialect(cls, token: EgglogToken) -> bool:
        return dialect_to_token.get(token.text) != None

    def next_token(self) -> EgglogToken:
        if self.index >= len(self.input):
            return EgglogToken(EgglogTokenKind.EOF, "")

        if self.input[self.index] == ",":
            # don't return a comma
            self.index += 1

        char = self.input[self.index]

        while char.isspace():
            self.index += 1
            if self.index >= len(self.input):
                return EgglogToken(EgglogTokenKind.EOF, "")
            char = self.input[self.index]

        if char.isalnum():
            token: str = ""
            while char.isalnum():
                token += char
                self.index += 1
                char = self.input[self.index]

            if token in str_to_token:
                return EgglogToken(str_to_token[token], token)

            if token.isdigit():
                return EgglogToken(EgglogTokenKind.INTEGER_LITERAL, token)

            return EgglogToken(EgglogTokenKind.STRING_LITERAL, token)

        if char == '"':
            token = ""
            self.index += 1

            char = self.input[self.index]
            while char != '"':
                token += char
                self.index += 1
                char = self.input[self.index]

            self.index += 1
            return EgglogToken(EgglogTokenKind.STRING_LITERAL, token)

        # Parse a variable name;
        if char == "_":
            token = char
            self.index += 1

            char = self.input[self.index]
            while char.isalnum() or char == "_":
                token += char
                self.index += 1
                char = self.input[self.index]

            # self.index += 1
            return EgglogToken(EgglogTokenKind.VARIABLE_NAME, token)

        self.index += 1
        if char not in str_to_token:
            raise ValueError(f"Unknown character: {char}")

        return EgglogToken(str_to_token[char], char)
