from __future__ import annotations
from egglog import *
from eggie.enodes.base import SSA
from eggie.enodes.arith import Arith
from eggie.enodes.linalg import Linalg

rewrites_ruleset = ruleset(name="rewrites")


@rewrites_ruleset.register
def arith_commutative(op1: SSA, op2: SSA, out: SSA):
    yield rewrite(Arith.addi(op1, op2, out)).to(Arith.addi(op2, op1, out))
    yield rewrite(Arith.muli(op1, op2, out)).to(Arith.muli(op2, op1, out))


@rewrites_ruleset.register
def arith_distributive(op1: SSA, op2: SSA, op3: SSA, out1: SSA, out2: SSA, out3: SSA):
    yield rewrite(
        Arith.addi(Arith.muli(op1, op2, out1), Arith.muli(op3, op2, out2), out3)
    ).to(Arith.muli(op2, Arith.addi(op1, op3, out2), out3))


# def is_power_of_2(num: i64Like):
#     # egraph = EGraph()
#     # print(f"num type: {num.eval()}")
#     # num = egraph.eval(num.eval())
#     # print(f"num: {num}")
#     egraph = EGraph()
#     num = egraph.eval(PyObject.from_int(num))

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
#     yield rewrite(Arith.muli(op, Arith.constant(const, const_out), mul_out)).to(
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


@rewrites_ruleset.register
def linalg_commutative(op1: SSA, op2: SSA, out: SSA, return_val: SSA):
    yield rewrite(Linalg.add(op1, op2, out, return_val)).to(
        Linalg.add(op2, op1, out, return_val)
    )
    # yield rewrite(Linalg.matmul(op1, op2, out, return_val)).to(
    #     Linalg.matmul(op2, op1, out, return_val)
    # )


@rewrites_ruleset.register
def linalg_distributive(
    rhs: SSA,
    matmul_one_op1: SSA,
    matmul_two_op1: SSA,
    matmul_op2: SSA,
    matmul_one_out: SSA,
    matmul_two_out: SSA,
    matmul_one_ret: SSA,
    matmul_two_ret: SSA,
    add_out: SSA,
    add_ret_val: SSA,
):
    op = Linalg.add(
        Linalg.matmul(matmul_one_op1, matmul_op2, matmul_one_out, matmul_one_ret),
        Linalg.matmul(matmul_two_op1, matmul_op2, matmul_two_out, matmul_two_ret),
        add_out,
        add_ret_val,
    )

    # yield rule(eq(op).to(rhs)).then(
    #     set_(op).to(
    #         Linalg.matmul(
    #             matmul_op2,
    #             Linalg.add(
    #                 matmul_one_op1, matmul_two_op1, matmul_two_out, matmul_two_ret
    #             ),
    #             add_out,
    #             add_ret_val,
    #         )
    #     ),
    #     delete(
    #         Linalg.matmul(matmul_one_op1, matmul_op2, matmul_one_out, matmul_one_ret),
    #     ),
    #     delete(Linalg.matmul(matmul_two_op1, matmul_op2, matmul_two_out, matmul_two_ret))
    # )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(matmul_one_op1, matmul_op2, matmul_one_out, matmul_one_ret),
            Linalg.matmul(matmul_two_op1, matmul_op2, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            matmul_op2,
            Linalg.add(matmul_one_op1, matmul_two_op1, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(matmul_op2, matmul_one_op1, matmul_one_out, matmul_one_ret),
            Linalg.matmul(matmul_two_op1, matmul_op2, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            matmul_op2,
            Linalg.add(matmul_one_op1, matmul_two_op1, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(matmul_op2, matmul_one_op1, matmul_one_out, matmul_one_ret),
            Linalg.matmul(matmul_op2, matmul_two_op1, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            matmul_op2,
            Linalg.add(matmul_one_op1, matmul_two_op1, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(matmul_op2, matmul_one_op1, matmul_one_out, matmul_one_ret),
            Linalg.matmul(matmul_two_op1, matmul_op2, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            matmul_op2,
            Linalg.add(matmul_one_op1, matmul_two_op1, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    # yield rewrite(
    #     Arith.addi(Arith.muli(op1, op2, matmul_one_out), Arith.muli(op3, op2, matmul_two_out), out3)
    # ).to(Arith.muli(op2, Arith.addi(op1, op3, matmul_two_out), out3))
