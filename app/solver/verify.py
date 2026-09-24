"""收口校核：KCL/KVL 逐个回代 + 功率守恒。

技能文档说得很直接：**功率守恒是最强的单条判据**。拓扑抄错的解往往也能让
KCL/KVL 自洽（因为错的拓扑配错的方程，看着一样"干净"），但能量一定不平。
所以报告里先给 ΣP，再给别的。

★ 一个统一的功率公式：**所有二端元件的吸收功率都等于
「沿参考方向的压降 × 沿参考方向的电流」**，即 ``P = drops[ref] · i[ref]``。
代入三种元件：
- 电阻：``P = (R·i)·i = R·i²`` ≥ 0
- 电压源：``P = (−E)·i``，正是技能文档里的 ``P = −E·(从 + 端流出的电流)``
- 电流源：``P = (V_a − V_b)·i``

一条公式覆盖三类元件，就不会出现"电阻用 R·i²、电源用另一套符号、两处写反一处"
的经典事故。符号约定：**吸收为正、提供为负**。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..ir.model import Circuit, CircuitError
from .base import Solution, declared_direction
from .branch import _build_edges, fundamental_loops
from .linalg import frac_float, frac_str


def _elem(x: Any) -> Fraction | float:
    """统一转成可比较的精确量：Fraction 保持精确，float 保持原样。"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise CircuitError("布尔值不能参与校核")
    return Fraction(x) if isinstance(x, int) else x


def _is_zero(x: Any, tol_abs: float = 1e-9) -> bool:
    if isinstance(x, Fraction):
        return x == 0
    return abs(float(x)) < tol_abs


def _mag(x: Any) -> float:
    if isinstance(x, Fraction):
        return abs(frac_float(x))
    return abs(float(x))


def _fmt(x: Any, digits: int = 10) -> str:
    if isinstance(x, Fraction):
        return frac_str(x)
    if x != x:                                   # NaN
        return "NaN"
    return f"{float(x):.{digits}g}"


# ---------------------------------------------------------------- KCL


def check_kcl(circuit: Circuit, sol: Solution) -> dict[str, Any]:
    """逐个节点回代 KCL：流出该节点的电流之和应为 0。

    抓的是：支路电流抄错、参考方向写反。
    """
    rows = []
    all_ok = True
    for node in circuit.nodes:
        total: Any = Fraction(0) if sol.exact else 0.0
        parts = []
        for c in circuit.components:
            f, t = declared_direction(c)
            if node == f:
                sign, other = +1, t
            elif node == t:
                sign, other = -1, f
            else:
                continue
            i = sol.get_current(c.ref)
            if i is None:
                continue
            total = total + sign * _elem(i)
            parts.append(f"{'+' if sign > 0 else '-'}{c.ref}({other})")
        ok = _is_zero(total)
        all_ok = all_ok and ok
        rows.append({
            "node": node, "residual": _fmt(total), "residual_float": _mag(total),
            "ok": ok, "terms": parts,
        })
    return {
        "name": "KCL 逐结点回代（流出之和 = 0）",
        "ok": all_ok, "rows": rows,
        "max_residual": max((r["residual_float"] for r in rows), default=0.0),
    }


# ---------------------------------------------------------------- KVL


def check_kvl(circuit: Circuit, sol: Solution) -> dict[str, Any]:
    """逐个基本回路回代 KVL：沿回路压降之和应为 0。

    抓的是：电源极性看反、回路列重、回路间线性相关。
    回路用生成树导出，与支路电流法同一套构造 —— 但**输入是已经解好的电压/压降**，
    所以这是"回代校核"，不是"再解一遍"。
    """
    nodes = circuit.nodes
    edges = _build_edges(circuit)
    # 回路统一由 fundamental_loops() 出 —— 那里带"自证闭合"，不会给出断开的回路

    rows = []
    all_ok = True
    known = {c.ref: c for c in circuit.components}
    for ci, (chord_ref, walk) in enumerate(fundamental_loops(circuit)):
        total: Any = Fraction(0) if sol.exact else 0.0
        terms = []
        for (x, y, ei) in walk:
            e_u, e_v, e_ref = edges[ei]
            c = known[e_ref]
            f, t = declared_direction(c)
            drop = sol.get_drop(c.ref)
            if drop is None:
                drop = sol.get_node(f) - sol.get_node(t)
            # 沿 x->y 行进时，压降的符号取决于是否顺着参考方向。
            # ★ 这里必须拿 **edge 存的下标** 比 **walk 里的下标**。
            #   早先写成 `x == f`（下标 vs 节点名字符串，如 2 vs "2"）——
            #   比较恒为 False，于是每一项都被取反，三个解法的 KVL 残差
            #   整齐地同时报同一个错值。**校核层自己写错，会让正确的解背锅**，
            #   而且看起来特别像"三法都错在同一个地方"。
            forward = (x == e_u and y == e_v)
            total = total + (_elem(drop) if forward else -_elem(drop))
            terms.append(f"{'⇢' if forward else '⇠'}{c.ref}")
        ok = _is_zero(total)
        all_ok = all_ok and ok
        rows.append({
            "loop": f"回路{ci+1}",
            "chord": chord_ref,
            "path": " ".join(terms),
            "residual": _fmt(total), "residual_float": _mag(total), "ok": ok,
        })
    return {
        "name": "KVL 逐基本回路回代（压降之和 = 0）",
        "ok": all_ok, "rows": rows,
        "max_residual": max((r["residual_float"] for r in rows), default=0.0),
    }


# ---------------------------------------------------------------- 功率守恒


def check_power(circuit: Circuit, sol: Solution) -> dict[str, Any]:
    """功率表 + 守恒结论。**报告里的第一条判据。**"""
    rows = []
    total: Any = Fraction(0) if sol.exact else 0.0
    supplied = 0.0
    consumed = 0.0

    for c in circuit.components:
        i = sol.get_current(c.ref)
        drop = sol.get_drop(c.ref)
        if drop is None:
            f, t = declared_direction(c)
            drop = sol.get_node(f) - sol.get_node(t)
        if i is None:
            continue
        p = _elem(drop) * _elem(i)               # 唯一公式，见模块头部
        total = total + p
        pf = frac_float(p) if isinstance(p, Fraction) else float(p)
        if pf >= 0:
            consumed += pf
        else:
            supplied += -pf
        role = "消耗" if pf > 0 else ("提供" if pf < 0 else "不参与")
        rows.append({
            "ref": c.ref, "kind": c.kind,
            "current": _fmt(i), "drop": _fmt(drop),
            "power": _fmt(p), "power_float": pf, "role": role,
        })

    ok = _is_zero(total, tol_abs=1e-7)
    return {
        "name": "功率守恒（ΣP = 0，吸收为正）",
        "ok": ok, "rows": rows,
        "total": _fmt(total), "total_float": _mag(total),
        "supplied": supplied, "consumed": consumed,
        "conclusion": (
            f"电源共提供 {supplied:.10g} W，元件共消耗 {consumed:.10g} W，"
            f"净残差 {_mag(total):.3g} W —— "
            + ("守恒，读图与建模一致。" if ok else
               "**不守恒**，几乎一定某处极性或拓扑读错了，先回查电源极性。")
        ),
    }


# ---------------------------------------------------------------- 汇总


def verify_all(circuit: Circuit, sol: Solution) -> dict[str, Any]:
    """对一个解跑完整套校核。"""
    checks = [check_power(circuit, sol), check_kcl(circuit, sol), check_kvl(circuit, sol)]
    return {
        "method": sol.method,
        "exact": sol.exact,
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
        "warnings": sol.warnings,
    }
