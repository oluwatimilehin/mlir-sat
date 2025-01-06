from __future__ import annotations
from egglog import *
from eggie.enodes.base import SSA
from eggie.enodes.arith import Arith
from eggie.enodes.linalg import Linalg

rewrites_ruleset = ruleset(name="rewrites")


@rewrites_ruleset.register
def commutativity(op1: SSA, op2: SSA, out: SSA):
    yield rewrite(Arith.addi(op1, op2, out)).to(Arith.addi(op2, op1, out))
    yield rewrite(Arith.muli(op1, op2, out)).to(Arith.muli(op2, op1, out))


@rewrites_ruleset.register
def distributive(op1: SSA, op2: SSA, op3: SSA, out1: SSA, out2: SSA, out3: SSA):
    yield rewrite(
        Arith.addi(Arith.muli(op1, op2, out1), Arith.muli(op3, op2, out2), out3)
    ).to(Arith.muli(op2, Arith.addi(op1, op3, out2), out3))


# def is_power_of_2(num: i64Like):
#     if num <= 0:
#         return False, None

#     is_power = (num & (num - 1)) == 0

#     if is_power:
#         exponent = num.bit_length() - 1
#         return True, exponent
#     else:
#         return False, None


# @rewrites_ruleset.register
# def mul_to_left_shift(const: i64Like, op: SSA, const_out: SSA, mul_out: SSA):
#     yield rewrite(Arith.muli(Arith.constant(const, const_out), op, mul_out)).to(
#         Arith.shli(op, Arith.constant(is_power_of_2(const)[1], const_out), mul_out),
#         is_power_of_2(const)[0],
#     )


@rewrites_ruleset.register
def constant_folding(
    const_one: i64,
    const_two: i64,
    const_one_out: SSA,
    const_two_out: SSA,
    mul_op: SSA,
    mul_one_out: SSA,
    mul_two_out: SSA,
):
    # new_const: i64 = i64(const_one) * const_two
    yield rewrite(
        Arith.muli(
            Arith.muli(mul_op, Arith.constant(const_one, const_one_out), mul_one_out),
            Arith.constant(const_two, const_two_out),
            mul_two_out,
        )
    ).to(
        Arith.muli(
            Arith.constant(const_one * const_two, const_two_out), mul_op, mul_two_out
        )
    )
