from enum import Enum, auto
import logging
from collections import namedtuple


class EgglogTokenKind(Enum):
    REGION = auto()
    BLOCK = auto()
    FUNCTION = auto()
    SSA = auto()
    TENSOR = auto()
    LINALG = auto()
    TENSORT = auto()
    OPERATION = auto()
    VEC = auto()
    STRING_LITERAL = auto()
    NUMBER_LITERAL = auto()
    COMMA = auto()
    LEFT_SQUARE_BRACKET = auto()
    RIGHT_SQUARE_BRACKET = auto()
    LEFT_PARENTHESIS = auto()
    RIGHT_PARENTHESIS = auto()
    DOT = auto()
    EOF = auto()


dialect_to_token = {
    "Tensor": EgglogTokenKind.TENSOR,
    "Linalg": EgglogTokenKind.LINALG,
    "Function": EgglogTokenKind.FUNCTION,
}
chars_to_token = {
    "[": EgglogTokenKind.LEFT_SQUARE_BRACKET,
    "]": EgglogTokenKind.RIGHT_SQUARE_BRACKET,
    "(": EgglogTokenKind.LEFT_PARENTHESIS,
    ")": EgglogTokenKind.RIGHT_PARENTHESIS,
    ",": EgglogTokenKind.COMMA,
    ".": EgglogTokenKind.DOT,
}

keyword_to_token = {
    "Region": EgglogTokenKind.REGION,
    "Block": EgglogTokenKind.BLOCK,
    "Vec": EgglogTokenKind.VEC,
    "SSA": EgglogTokenKind.SSA,
    "Operation": EgglogTokenKind.OPERATION,
    "TensorT": EgglogTokenKind.TENSORT,
}


str_to_token = dialect_to_token | chars_to_token | keyword_to_token
Token = namedtuple("Token", ["kind", "text"])


class Lexer:
    def __init__(self, input: str):
        self.input = input
        self.index = 0

    @classmethod
    def is_keyword(cls, token: Token) -> bool:
        return keyword_to_token.get(token.text) != None

    @classmethod
    def is_dialect(cls, token: Token) -> bool:
        return dialect_to_token.get(token.text) != None

    def next_token(self) -> Token:
        if self.index >= len(self.input):
            return Token(EgglogTokenKind.EOF, "")

        char = self.input[self.index]

        while char.isspace():
            self.index += 1
            if self.index >= len(self.input):
                return Token(EgglogTokenKind.EOF, "")
            char = self.input[self.index]

        if char.isalnum():
            token: str = ""
            while char.isalnum():
                token += char
                self.index += 1
                char = self.input[self.index]

            if token in str_to_token:
                return Token(str_to_token[token], token)

            if token.isdigit():
                return Token(EgglogTokenKind.NUMBER_LITERAL, token)

            # todo: this string literal mechanism is wrong
            # print(f"returning literal: {token}")
            return Token(EgglogTokenKind.STRING_LITERAL, token)

        if char == '"':
            token = ""
            self.index += 1

            char = self.input[self.index]
            while char != '"':
                token += char
                self.index += 1
                char = self.input[self.index]

            self.index += 1
            print(f"returning literal: {token}")
            return Token(EgglogTokenKind.STRING_LITERAL, token)

        self.index += 1
        if char not in str_to_token:
            logging.warn(f"Unseen character: {char}")
            return

        return Token(str_to_token[char], char)
