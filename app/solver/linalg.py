"""精确有理数线性代数。

为什么要 Fraction 而不是 float：这一层的产出要拿来**跟 ngspice 的浮点结果对账**。
如果自己也是浮点，两边的误差就混在一起，没法判断"差 1e-6"是算法问题还是浮点问题。
用精确有理数，等式 `自己 - ngspice` 的偏差就干净地只剩 ngspice 的浮点误差，
对账表才有意义（对账阈值定在 1e-10 也是因为这个）。

顺带一个实际好处：教科书的数字都是配好的，精确解的分母会统一（比如整道题
全是 563），一眼就能看出读图和建模都对上了。浮点解没有这个自检能力。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from ..ir.model import CircuitError


Num = Fraction | int | float | str


def to_frac(x: Num) -> Fraction:
    """把数字转成精确有理数。

    ★ float 按**十进制最短表示**转，而不是按二进制真值 —— 这一条是刻意的：

    ``Fraction(0.1)`` 得到的是 3602879701896397/36028797018963968（一个分母为 2⁵⁵ 的怪分数），
    因为 0.1 在二进制里不精确。用户的意图显然是 1/10，而不是"最接近 0.1 的那个 double"。
    用 ``Fraction(str(0.1))`` 就得到干净的 1/10。

    这不只是好看。技能文档里有一条很有用的自检：
    **"若结果的分母统一（如本题全是 563），说明读图和建模都对了。"**
    如果元件值一进来就被二进制毛刺污染（1µF 变成分母 2⁵³ 的分数），
    整张电路的解的分母会全是天文数字，这条自检直接失效。

    反过来说：若某个值真的是算出来的（如 ngspice 回来的浮点），
    str() 会给出 17 位有效数字的十进制串，转成分数依然与它一一对应，不会丢信息。
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise CircuitError("布尔值不是合法的电路参数")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise CircuitError(f"非有限数值 {x!r} 不能进方程")
        if x == int(x) and abs(x) < 1e16:
            return Fraction(int(x))          # 10.0 -> 10，别写成 10/1 以外的东西
        return Fraction(str(x))
    if isinstance(x, str):
        return Fraction(x)          # 支持 "3/7" 这种写法
    raise CircuitError(f"无法把 {x!r} 转成有理数")


def solve_linear(
    A: Sequence[Sequence[Num]], b: Sequence[Num], *, what: str = "线性方程组"
) -> list[Fraction]:
    """高斯-约当消元，全 Fraction 精确运算。

    奇异时抛 CircuitError 并说明**为什么可能是电路画错了** ——
    矩阵奇异在本项目里几乎总是"KCL/KVL 少列了一条、或列了线性相关的回路"
    （技能文档第 4 条校核表里点名的就是这个）。
    """
    n = len(A)
    if n == 0:
        return []
    if any(len(row) != n for row in A):
        raise CircuitError(f"{what}：系数矩阵不是方阵")
    if len(b) != n:
        raise CircuitError(f"{what}：右端项长度 {len(b)} 与系数矩阵阶数 {n} 不一致")

    M = [[to_frac(v) for v in row] + [to_frac(b[i])] for i, row in enumerate(A)]

    pivot_row = 0
    for col in range(n):
        piv = None
        for r in range(pivot_row, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        M[pivot_row], M[piv] = M[piv], M[pivot_row]
        pv = M[pivot_row][col]
        M[pivot_row] = [v / pv for v in M[pivot_row]]
        for r in range(n):
            if r != pivot_row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[pivot_row][c] for c in range(n + 1)]
        pivot_row += 1

    if pivot_row < n:
        # 找出自由列，报出来便于定位
        free = [
            c for c in range(n)
            if all(M[r][c] == 0 for r in range(n))
        ]
        raise CircuitError(
            f"{what}的系数矩阵奇异（秩 {pivot_row} < {n}）。"
            "常见原因：KCL/KVL 少列一条或列了线性相关的回路；"
            "电路里有电压源直接并联、电流源悬空串联、或存在全电阻构成的闭环独立分量。"
            f"自由变量可能的列号：{free}"
        )

    return [M[r][n] for r in range(n)]


def frac_str(x: Fraction) -> str:
    """分数转人类可读串：整数就写整数，否则写 p/q。"""
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def frac_float(x: Fraction) -> float:
    return x.numerator / x.denominator
