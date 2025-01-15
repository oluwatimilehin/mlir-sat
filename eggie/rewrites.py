from __future__ import annotations
from egglog import *
from eggie.enodes.base import SSA
from eggie.enodes.arith import Arith
from eggie.enodes.linalg import Linalg

rewrites_ruleset = ruleset(name="rewrites")


@rewrites_ruleset.register
def arith_commutative(x: SSA, y: SSA, out: SSA):
    yield rewrite(Arith.addi(x, y, out)).to(Arith.addi(y, x, out))
    yield rewrite(Arith.muli(x, y, out)).to(Arith.muli(y, x, out))


@rewrites_ruleset.register
def arith_distributive(x: SSA, y: SSA, z: SSA, out1: SSA, out2: SSA, out3: SSA):
    yield rewrite(
        Arith.addi(Arith.muli(x, y, out1), Arith.muli(z, y, out2), out3)
    ).to(Arith.muli(y, Arith.addi(x, z, out2), out3))


@rewrites_ruleset.register
def mul_to_left_shift(op: SSA, const_out: SSA, mul_out: SSA):
    for i in range(1, 10):
        power_of_2 = 2**i
        shift_amount = i
        yield rewrite(
            Arith.muli(op, Arith.constant(power_of_2, const_out), mul_out)
        ).to(Arith.shli(op, Arith.constant(shift_amount, const_out), mul_out))


@rewrites_ruleset.register
def constant_folding(
    x: i64,
    y: i64,
    x_out: SSA,
    y_out: SSA,
    z: SSA,
    mul_one_out: SSA,
    mul_two_out: SSA,
):
    yield rewrite(
        Arith.muli(
            Arith.muli(z, Arith.constant(x, x_out), mul_one_out),
            Arith.constant(y, y_out),
            mul_two_out,
        )
    ).to(
        Arith.muli(
            Arith.constant(x * y, y_out), z, mul_two_out
        )
    )


@rewrites_ruleset.register
def linalg_commutative(x: SSA, y: SSA, out: SSA, return_val: SSA):
    yield rewrite(Linalg.add(x, y, out, return_val)).to(
        Linalg.add(y, x, out, return_val)
    )


@rewrites_ruleset.register
def linalg_distributive(
    x: SSA,
    y: SSA,
    z: SSA,
    matmul_one_out: SSA,
    matmul_two_out: SSA,
    matmul_one_ret: SSA,
    matmul_two_ret: SSA,
    add_out: SSA,
    add_ret_val: SSA,
):

    yield rewrite(
        Linalg.add(
            Linalg.matmul(x, z, matmul_one_out, matmul_one_ret),
            Linalg.matmul(y, z, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            z,
            Linalg.add(x, y, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(z, x, matmul_one_out, matmul_one_ret),
            Linalg.matmul(y, z, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            z,
            Linalg.add(x, y, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(z, x, matmul_one_out, matmul_one_ret),
            Linalg.matmul(z, y, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            z,
            Linalg.add(x, y, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )

    yield rewrite(
        Linalg.add(
            Linalg.matmul(z, x, matmul_one_out, matmul_one_ret),
            Linalg.matmul(y, z, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    ).to(
        Linalg.matmul(
            z,
            Linalg.add(x, y, matmul_two_out, matmul_two_ret),
            add_out,
            add_ret_val,
        )
    )
