"""节点电压法（MNA，精确有理数）。

未知量 = 各节点电压（除参考点） + **每个电压输出元件的支路电流**
（理想电压源 ``V`` 与电压型受控源 ``E``/``H`` —— 它们都靠"给一条电压约束"
来工作，都得配一个自己的电流未知量）。

方程 = 每个非参考节点的 KCL + 每个电压输出元件的电压约束。

电压输出元件支路电流的参考方向取**电源内部由 − 流向 +**（见 base.declared_direction），
于是"电流为正"就等于"这个元件在输出功率"，功率符号不用再换算。

★ 关于"理想电压源把未知量减掉"：两个串联的电压源一夹，中间那个节点的电压
就是常数，理论上可以把它从未知量里删掉，缩小方程规模。本实现**不删**，
依然把它当未知量 —— 因为删变量要额外写一套简化逻辑，而多出来的一两阶
对本题规模毫无影响，却会引入新的出错面。少写代码 = 少犯错。
矩阵规模靠"节点数 + 电压输出元件数"自然收敛。

## 受控源在这里怎么进方程

* ``E``/``H``（输出为电压）：照样占一个电流未知量列，只是**约束行右边不再是常数**，
  改成控制量的线性表达式 ``V_p − V_n − 控制量 = 0``。
* ``G``/``F``（输出为电流）：不占未知量列，它们的电流直接进 KCL 的左边
  （在理想电流源那里，电流是常数、要移到右边；而受控源的电流含未知量，
  是左边的一部分）。

两种情形都通过 ``controlled.control_terms()`` 取到统一的
``(节点系数, 源电流系数, 常数)``，**标准形与用户自定义表达式共用这一条出口** ——
不然"标准形写一处、表达式写另一处"，迟早有一处把符号写反，
而三条路径会一致地错成同一个答案，互校拦不住。
"""

from __future__ import annotations

from fractions import Fraction

from ..ir.model import Circuit, CircuitError
from .base import Solution, declared_direction, reject_non_dc
from .controlled import MnaCtx, control_terms, ensure_sense_sources, terms_value
from .linalg import solve_linear, to_frac


def node_voltage_method(circuit: Circuit) -> Solution:
    """节点电压法主解。返回全精确的有理数解。"""
    # ★ 电流控制型受控源要先在采样支路里插 0V 探针源（幂等）。
    #   插进来的元件**只存在于求解这张副本里**，不写回会话 ——
    #   否则界面上的接线与叠图坐标会跟着错。
    circuit, sense = ensure_sense_sources(circuit)
    circuit.validate()
    reject_non_dc(circuit)

    hot = circuit.hot_nodes
    vsources = [c for c in circuit.components if c.outputs_voltage]

    idx = {n: i for i, n in enumerate(hot)}
    n_hot = len(hot)
    n_tot = n_hot + len(vsources)
    #: 电压输出元件位号 -> 该元件电流未知量的列号（也是它的电压约束行号）
    #: 用位号而不是 dataclass 相等性来定位，避免两个结构相同的元件互相顶替
    vslot = {c.ref: n_hot + k for k, c in enumerate(vsources)}
    ctx = MnaCtx(node_col=idx, vslot=vslot)

    if n_tot == 0:
        raise CircuitError("没有任何未知量（既没有非参考节点也没有电压输出元件）")

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

        elif c.outputs_voltage:
            # ---- 电压输出元件（V / E / H）：占一个电流未知量列
            k = vslot[c.ref]
            p, n = c.nodes                      # p = + 端
            rp, rn = row_of(p), row_of(n)
            # 电流在源内部由 n 流向 p：流出节点 p 进入支路的电流 = −i
            if rp is not None:
                A[rp][k] -= 1
            # 流出节点 n 进入支路的电流 = +i
            if rn is not None:
                A[rn][k] += 1
            # 电压约束行：V_p − V_n = 右边
            if p in idx:
                A[k][idx[p]] += 1
            if n in idx:
                A[k][idx[n]] -= 1
            if c.kind == "V":
                if val is None:
                    raise CircuitError(f"{c.ref}: 电压源缺数值")
                rhs[k] += val
            else:
                # 受控源：右边是控制量的线性表达式。
                # 方程 V_p − V_n − 控制量 = 0  ⇒  控制量移到左边、常数移到右边
                nc, vc, cons = control_terms(c, circuit, ctx)
                for col, co in nc.items():
                    A[k][col] -= co
                for col, co in vc.items():
                    A[k][col] -= co
                rhs[k] += cons

        elif c.kind in ("G", "F"):
            # ---- 电流输出受控源：不占未知量列，电流直接进 KCL 左边
            #      （理想电流源的电流是常数、要移到右边；这里含未知量，是左边的一部分）
            nc, vc, cons = control_terms(c, circuit, ctx)
            a, b = c.nodes                      # 参考方向 a -> b
            ra, rb = row_of(a), row_of(b)
            if ra is not None:                  # 流出 a 的电流 = +i
                for col, co in nc.items():
                    A[ra][col] += co
                for col, co in vc.items():
                    A[ra][col] += co
                rhs[ra] -= cons
            if rb is not None:                  # 流出 b 的电流 = −i
                for col, co in nc.items():
                    A[rb][col] -= co
                for col, co in vc.items():
                    A[rb][col] -= co
                rhs[rb] += cons

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
        elif c.outputs_voltage:
            currents[c.ref] = src_current[c.ref]
        elif c.kind in ("G", "F"):
            # ★ 受控源的电流不是未知量，必须按它的控制关系**回代**算出来。
            #   漏这一步的话，KCL 校验与功率守恒都会把它当 0 ——
            #   而功率"守恒"反而更容易成立（少算了它那部分），
            #   于是看起来一切正常。属于典型的静默缺值。
            nc, vc, cons = control_terms(c, circuit, ctx)
            currents[c.ref] = terms_value(nc, vc, cons, node_v, hot,
                                          src_current, vsources)
        else:
            raise CircuitError(f"{c.ref}: 未知元件类型 {c.kind}")

    detail = {
        "unknowns": [*hot, *(f"i({c.ref})" for c in vsources)],
        "matrix_size": n_tot,
        "source_currents": {c.ref: str(src_current[c.ref]) for c in vsources},
    }
    if sense.get("applied") or circuit.controlled:
        detail["controlled"] = _controlled_detail(circuit, currents, drops)
    if sense.get("inserted"):
        detail["sense_probes"] = sense["inserted"]

    return Solution(
        method="节点电压法(MNA/精确有理数)",
        node_voltages=node_v,
        branch_currents=currents,
        branch_drops=drops,
        exact=True,
        detail=detail,
    )


def _controlled_detail(circuit: Circuit, currents: dict[str, Fraction],
                       drops: dict[str, Fraction]) -> list[dict[str, str]]:
    """受控源的受控关系与实际取值，供报告逐条列出。"""
    out: list[dict[str, str]] = []
    for c in circuit.controlled:
        std = "" if (c.ctrl and c.ctrl.expr) else f"{c.value}×控制量"
        out.append({
            "ref": c.ref, "kind": c.kind,
            "control": c.control_text(),
            "expr": (c.ctrl.expr if c.ctrl else ""),
            "mode": "自定义表达式" if (c.ctrl and c.ctrl.expr) else "标准形",
            "standard_form": std,
            "i": str(currents.get(c.ref, "")),
            "u": str(drops.get(c.ref, "")),
            "sense": (c.ctrl.sampling if c.ctrl else ""),
            "probe": (c.ctrl.sense_ref if c.ctrl else ""),
        })
    return out
