"""节点电压法（MNA，精确有理数）。

未知量 = 各节点电压（除参考点） + **每个电压源支路的电流**。
方程 = 每个非参考节点的 KCL + 每个电压源的电压约束。

电压源支路电流的参考方向取**电源内部由 − 流向 +**（见 base.declared_direction），
于是"电流为正"就等于"这个电源在供电"，功率符号不用再换算。

★ 关于"理想电压源把未知量减掉"：两个串联的电压源一夹，中间那个节点的电压
就是常数，理论上可以把它从未知量里删掉，缩小方程规模。本实现**不删**，
依然把它当未知量 —— 因为删变量要额外写一套简化逻辑，而多出来的一两阶
对本题规模毫无影响，却会引入新的出错面。少写代码 = 少犯错。
矩阵规模靠"节点数 + 电压源数"自然收敛。
"""

from __future__ import annotations

from fractions import Fraction

from ..ir.model import Circuit, CircuitError
from .base import Solution, declared_direction, reject_non_dc
from .linalg import solve_linear, to_frac


def node_voltage_method(circuit: Circuit) -> Solution:
    """节点电压法主解。返回全精确的有理数解。"""
    circuit.validate()
    reject_non_dc(circuit)

    hot = circuit.hot_nodes
    vsources = [c for c in circuit.components if c.kind == "V"]

    idx = {n: i for i, n in enumerate(hot)}
    n_hot = len(hot)
    n_tot = n_hot + len(vsources)
    #: 电压源位号 -> 该源电流未知量的列号（也是它的电压约束行号）
    #: 用位号而不是 dataclass 相等性来定位，避免两个结构相同的元件互相顶替
    vslot = {c.ref: n_hot + k for k, c in enumerate(vsources)}

    if n_tot == 0:
        raise CircuitError("没有任何未知量（既没有非参考节点也没有电压源）")

    A: list[list[Fraction]] = [[Fraction(0)] * n_tot for _ in range(n_tot)]
    rhs: list[Fraction] = [Fraction(0)] * n_tot

    def row_of(node: str) -> int | None:
        """节点对应的 KCL 行号；参考节点的 KCL 不列（它由其余方程自动满足）。"""
        return idx.get(node)

    # ---- KCL：对每个非参考节点，令"流出该节点的电流之和 = 0"
    for c in circuit.components:
        val = to_frac(c.value) if c.value is not None else None
        f, t = declared_direction(c)

        if c.kind == "R":
            if val is None or val <= 0:
                raise CircuitError(f"{c.ref}: 电阻必须为正，收到 {c.value!r}")
            g = Fraction(1) / val
            # 电阻按 nodes[0] -> nodes[1] 的参考方向，(V_a − V_b)/R 流出 a
            a, b = c.nodes
            ra, rb = row_of(a), row_of(b)
            if ra is not None:
                A[ra][idx[a]] += g
                if b in idx:
                    A[ra][idx[b]] -= g
            if rb is not None:
                A[rb][idx[b]] += g
                if a in idx:
                    A[rb][idx[a]] -= g

        elif c.kind == "I":
            if val is None:
                raise CircuitError(f"{c.ref}: 电流源缺数值")
            # 电流在源内部由 a 流向 b，外部由 b 端流出
            # 于是"流出节点 a 进入支路"的电流 = +I，"流出节点 b 进入支路" = −I
            ra, rb = row_of(f), row_of(t)
            if ra is not None:
                rhs[ra] -= val
            if rb is not None:
                rhs[rb] += val

        elif c.kind == "V":
            k = vslot[c.ref]
            p, n = c.nodes                      # p = + 端
            rp, rn = row_of(p), row_of(n)
            # 电流在源内部由 n 流向 p：流出节点 p 进入支路的电流 = −i
            if rp is not None:
                A[rp][k] -= 1
            # 流出节点 n 进入支路的电流 = +i
            if rn is not None:
                A[rn][k] += 1
            # 电压约束行：V_p − V_n = E
            if p in idx:
                A[k][idx[p]] += 1
            if n in idx:
                A[k][idx[n]] -= 1
            if val is None:
                raise CircuitError(f"{c.ref}: 电压源缺数值")
            rhs[k] += val

        else:
            raise CircuitError(f"{c.ref}: 节点电压法暂不支持元件类型 {c.kind}")

    sol_vec = solve_linear(A, rhs, what="节点电压法")

    node_v: dict[str, Fraction] = {circuit.ref_node: Fraction(0)}
    for n, i in idx.items():
        node_v[n] = sol_vec[i]
    src_current: dict[str, Fraction] = {
        c.ref: sol_vec[n_hot + k] for k, c in enumerate(vsources)
    }

    # ---- 由解反推支路电流与压降
    currents: dict[str, Fraction] = {}
    drops: dict[str, Fraction] = {}
    for c in circuit.components:
        f, t = declared_direction(c)
        vf, vt = node_v.get(f, Fraction(0)), node_v.get(t, Fraction(0))
        drops[c.ref] = vf - vt
        if c.kind == "R":
            currents[c.ref] = drops[c.ref] / to_frac(c.value)
        elif c.kind == "I":
            currents[c.ref] = to_frac(c.value)
        elif c.kind == "V":
            currents[c.ref] = src_current[c.ref]
        else:
            raise CircuitError(f"{c.ref}: 未知元件类型 {c.kind}")

    return Solution(
        method="节点电压法(MNA/精确有理数)",
        node_voltages=node_v,
        branch_currents=currents,
        branch_drops=drops,
        exact=True,
        detail={
            "unknowns": [*hot, *(f"i({c.ref})" for c in vsources)],
            "matrix_size": n_tot,
            "source_currents": {c.ref: str(src_current[c.ref]) for c in vsources},
        },
    )
