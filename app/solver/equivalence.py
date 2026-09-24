"""等效电源（戴维南/诺顿）与叠加原理。

这两样既是《电路》课程的常考题型，也是本项目**第四条独立校验路径**：
叠加原理靠线性性把响应分解，数学路线与"解一个完整方程组"完全不同
（前者是 N 次小规模求解后相加，后者是一次大规模求解），
所以它抓共性错误的能力和前三条是互补的。

★ 一个必须写进报告的陷阱：**叠加只对电压/电流成立，对功率不成立**。
因为 P = R·i² 是二次的，把各电源单独作用时的功率相加是错的。
报告里会显式警告这一条 —— 这是学生最常踩的坑之一。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..ir.model import Circuit, CircuitError, Component, Evidence
from .base import Solution, declared_direction
from .linalg import frac_float, frac_str
from .mna import node_voltage_method


# ---------------------------------------------------------------- 源置零


def kill_sources(circuit: Circuit, keep: set[str] | None = None) -> Circuit:
    """把独立源置零，得到"无源网络"。

    规则（与教材完全一致）：
    - 电压源置零 = **短路** → 保留元件、数值改 0 V（不去改拓扑，
      这样不会引入意外的并联电压源，也不会让节点编号漂移）
    - 电流源置零 = **开路** → 保留元件、数值改 0 A

    ``keep`` 非空时，只对不在 keep 里的源置零（叠加原理用）。
    """
    out = circuit.copy()
    for c in out.components:
        if c.kind not in ("V", "I"):
            continue
        if keep is not None and c.ref in keep:
            continue
        c.value = 0.0
        c.evidence = Evidence(source="exact", confidence=1.0,
                              detail="由 kill_sources 置零（电压源->短路，电流源->开路）")
    return out


# ---------------------------------------------------------------- 叠加


def superposition(circuit: Circuit) -> dict[str, Any]:
    """叠加原理：各独立源单独作用的结果相加。

    返回逐源贡献表 + 与完整解的偏差 + 适用性说明。
    """
    circuit.validate()
    sources = [c for c in circuit.components if c.kind in ("V", "I") and c.value != 0]
    if not sources:
        raise CircuitError("电路里没有独立源，叠加原理无从谈起")
    if len(sources) == 1:
        return {
            "applicable": True,
            "note": "只有一个独立源，叠加退化为它自身，无额外信息量。",
            "contributions": [], "rows": [],
            "max_deviation": 0.0, "ok": True,
        }

    full = node_voltage_method(circuit)
    contribs: dict[str, dict[str, Any]] = {}
    for s in sources:
        sub = kill_sources(circuit, keep={s.ref})
        part = node_voltage_method(sub)
        contribs[s.ref] = {
            "node_voltages": part.node_voltages,
            "branch_currents": part.branch_currents,
        }

    # 相加
    nodes = circuit.nodes
    summed_v = {n: Fraction(0) for n in nodes}
    for s in sources:
        for n in nodes:
            summed_v[n] += contribs[s.ref]["node_voltages"].get(n, Fraction(0))

    summed_i: dict[str, Fraction] = {}
    for c in circuit.components:
        acc = Fraction(0)
        for s in sources:
            iv = contribs[s.ref]["branch_currents"].get(c.ref)
            acc += iv if iv is not None else Fraction(0)
        summed_i[c.ref] = acc

    # 与完整解对账
    rows = []
    max_dev = 0.0
    for n in nodes:
        exact = full.node_voltages.get(n, Fraction(0))
        sup = summed_v[n]
        dev = abs(frac_float(exact - sup))
        max_dev = max(max_dev, dev)
        rows.append({
            "quantity": f"V({n})",
            "full": frac_str(exact), "superposed": frac_str(sup),
            "deviation": dev,
            "per_source": {s.ref: frac_str(contribs[s.ref]["node_voltages"].get(n, Fraction(0)))
                           for s in sources},
        })

    ok = max_dev == 0.0        # 全精确有理数，应当严格为 0
    return {
        "applicable": True,
        "sources": [s.ref for s in sources],
        "rows": rows,
        "max_deviation": max_dev,
        "ok": ok,
        "branch_currents": {k: frac_str(v) for k, v in summed_i.items()},
        "note": (
            "叠加原理基于线性性，**只对电压与电流成立**。"
            "功率是二次量（P = R·i²），把各源单独作用时的功率直接相加是错的 —— "
            "必须先叠加得到总电流，再由总电流算功率。"
        ),
        "conclusion": (
            f"逐源叠加结果与完整解最大偏差 {max_dev:.3g} V，"
            + ("严格为 0，线性性自洽。" if ok else
               "**不为 0**，说明某处建模含非线性项或解的符号约定不一致。")
        ),
    }


# ---------------------------------------------------------------- 戴维南/诺顿


def thevenin_norton(
    circuit: Circuit, node_p: str, node_q: str, *, test_current: float = 1.0
) -> dict[str, Any]:
    """求端口 (node_p, node_q) 的戴维南/诺顿等效。

    做法（全程精确有理数，不用任何数值拟合）：
    1. **开路电压** ``Voc = V_p − V_q``：直接解原电路。
    2. **等效电阻** ``Req``：把所有独立源置零（V→短路，I→开路），
       再从端口注入测试电流 ``I_t``（注入 node_p、抽出 node_q），
       ``Req = (V_p − V_q) / I_t``。
    3. **短路电流** ``Isc = Voc / Req``（Req=0 或 ∞ 时单独说明）。

    ★ 注入方向容易搞反：电流源的参考方向是"源内部由 from 流向 to"，
    所以要让电流**注入 node_p**，测试源必须写成 ``nodes=(q, p)``
    （内部 q→p，于是在 p 端流出，即注入 p）。写反会得到负电阻。
    """
    circuit.validate()
    if node_p not in circuit.nodes or node_q not in circuit.nodes:
        raise CircuitError(f"端口节点 {node_p!r}/{node_q!r} 不在电路里")
    if node_p == node_q:
        raise CircuitError("端口两端不能是同一个节点")

    full = node_voltage_method(circuit)
    voc = full.get_node(node_p) - full.get_node(node_q)

    # ---- Req
    passive = kill_sources(circuit)
    t = to_frac_safe(test_current)
    test_ref = Circuit.auto_ref("I", passive.refs())
    passive.components.append(Component(
        ref=test_ref, kind="I", nodes=(node_q, node_p), value=test_current,
        evidence=Evidence(source="exact", confidence=1.0, detail="戴维南等效测试电流源"),
    ))

    req: Fraction | None = None
    req_note = ""
    try:
        probe = node_voltage_method(passive)
        vp = probe.get_node(node_p)
        vq = probe.get_node(node_q)
        req = (vp - vq) / t
        if req < 0:
            req_note = ("算出负等效电阻。含受控源的电路确实可能为负，"
                        "但本版本不支持受控源 —— 请确认端口选择与读图是否正确。")
    except CircuitError as e:
        req = None
        req_note = f"无源网络在端口处开路（置零后端口之间没有直流通路）：{e}"

    isc = None
    if req is not None and req != 0:
        isc = voc / req

    return {
        "port": [node_p, node_q],
        "voc": frac_str(voc), "voc_float": frac_float(voc),
        "req": frac_str(req) if req is not None else None,
        "req_float": frac_float(req) if req is not None else None,
        "isc": frac_str(isc) if isc is not None else None,
        "isc_float": frac_float(isc) if isc is not None else None,
        "test_current": test_current,
        "note": req_note or "等效电阻由「源置零 + 端口注入测试电流」精确求得。",
        "norton_current": frac_str(isc) if isc is not None else None,
    }


def to_frac_safe(x: Any) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x)
