"""把**参数名写的线性表达式**渲染成 ngspice 能读的文本。

它为什么住在 IR 层：**两处都要用它** ——

* 求解层的 ngspice 路径要用它写行为源（``B``）的表达式；
* 网表层（``ir/spice.py``）的 ``to_spice()`` 要用它把受控源落盘。

放在求解层，网表层就得反过来 import 求解层，要么成环、要么逼着
"网表那边再写一份"。而两份渲染器的分叉表现是最难查的那种：
**网表看着没问题，ngspice 算出来却是另一个电路**。

支持范围是**刻意的**，只覆盖"能一对一写成 SPICE 表达式"的量：
节点电压、支路电压、独立源与受控源的电流、电阻上的电流
（展开成 ``V(a,b)/R``）、以及元件值。别的一律报错 ——
宁可不支持，也不要生成一条 ngspice 看不懂的表达式，
让它甩一句带行号的错出来。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable

from .model import (Circuit, Component, CircuitError, declared_direction)
from .params import MAX_EXPR_DEPTH, LinearExpr

#: ★★ IR 的"电压源支路电流"与 SPICE 的 ``I(Vx)`` **互为相反数**。只此一处定义。
#:
#:   * IR：``i > 0`` 表示该支路**向外供出功率**（参考方向取元件内部 −→+）。
#:   * SPICE：``I(Vx)`` 正号 = 电流由 **+ 端流入**（元件内部 +→−）。
#:
#: 凡是要把"某条电压源支路的电流"交给 ngspice，都必须乘上它：
#:
#:   1. 表达式里的 ``I(...)`` 片段 —— 见本模块 :func:`_term`；
#:   2. ``H``/``F`` 卡的控制增益 —— **求解层与网表层各有一处**，
#:      两边都调下面那两个函数，不各写一个负号。
#:
#: 为什么要拧成一股：实测过一次"标准形 H 卡差号、走自定义表达式的同一个
#: 电路却不差号"——因为两处各写了一个负号，其中一处漏了。同题两答案，
#: 且三法互校拦不住（ngspice 那一路自己内部就不一致）。
SPICE_SOURCE_CURRENT_SIGN = -1


def spice_gain_value(gain: Any) -> float:
    """IR 增益 → SPICE ``H``/``F`` 卡上该写的数（给 PySpice 用的浮点）。"""
    return SPICE_SOURCE_CURRENT_SIGN * float(gain)


def spice_gain_text(gain: Any) -> str:
    """同上，但给文本网表用（保持 ``-2000`` 而不是 ``-2000.0``）。"""
    if isinstance(gain, Fraction):
        return num_str(gain * SPICE_SOURCE_CURRENT_SIGN)
    return num_str(Fraction(gain) * SPICE_SOURCE_CURRENT_SIGN)


def num_str(v: Any) -> str:
    """把数值写成 ngspice 认的字面量。

    ★ 用 ``repr(float)`` 而不是工程记法：``1e-05`` 一定被 ngspice 认成
    "1 乘 10 的 −5 次"，而 ``10u`` 这类后缀在**表达式内部**的解析规则
    与元件卡上并不完全相同（历史上有过 ``m`` 的歧义）。表达式的正确性
    比可读性重要，所以这里一律走纯数字。
    """
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator)
        return repr(v.numerator / v.denominator)
    f = float(v)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _table(circuit: Circuit):
    # 参数表是"名字 ↔ 绑定"的派生层。表达式用它把名字翻回 ref / 节点名。
    # 万一表还没同步（表里一个名字都没有），表达式就会被判成"未知符号" ——
    # 所以这里先补一次同步，代价是一次 O(n) 遍历。
    if not circuit.params.items:
        circuit.sync_params()
    return circuit.params


def _term(comp: Component, circuit: Circuit, *,
          netname: Callable[[Component], str], depth: int) -> str:
    """把"一条支路的电流"写成 ngspice 片段。"""
    if comp.outputs_voltage:
        # 实测：ngspice 支持 I(V1)，也支持 I(E1)（受控源同样是电压源性质）。
        # 取到的是"流入 + 端的电流"，与 IR 的"内部 −→+"差一个符号 ——
        # 这里的片段是给**表达式**用的，必须补回这个符号，
        # 否则自定义表达式会整体差号，而三条路径会一致地差同一个号，
        # 三法互校根本发现不了。
        #
        # ★ 同一处符号修正也在 ngspice.py 的 H/F 卡增益上（那里取反）。
        #   两处必须一起改：改一处会让"自定义表达式"与"标准形"分家，
        #   而分家的表现是同一个电路两种写法算出不同答案。
        return f"(-1)*I({netname(comp)})"
    if comp.kind == "R":
        if comp.value is None:
            raise CircuitError(f"{comp.ref}: 电阻还没有数值，写不出电流表达式")
        f, t = declared_direction(comp)
        x = "0" if f == circuit.ref_node else f
        y = "0" if t == circuit.ref_node else t
        return f"V({x},{y})/({num_str(comp.value)})"
    if comp.kind == "I":
        if comp.value is None:
            raise CircuitError(f"{comp.ref}: 电流源还没有数值，写不出电流表达式")
        return num_str(comp.value)
    if comp.is_controlled:
        ctrl = comp.ctrl
        if ctrl is not None and ctrl.expr:
            lin = _table(circuit).resolve_expression(ctrl.expr)
            return f"({spice_expression(lin, circuit, netname=netname, depth=depth)})"
        raise CircuitError(
            f"{comp.ref}: 要在表达式里引用另一个受控源的电流，"
            "那个受控源自己也得写成表达式 —— 这种情况本版本不支持"
            "（会出现代数环）。请把它的控制关系直接展开写进表达式。")
    raise CircuitError(f"{comp.ref}: 元件类型 {comp.kind} 的电流写不成 SPICE 表达式")


def spice_expression(lin: LinearExpr, circuit: Circuit, *,
                     netname: Callable[[Component], str],
                     depth: int = 0) -> str:
    """把一个线性表达式写成 **ngspice 能读的文本**。

    ``netname`` 负责"IR 位号 → 网表里的器件名"（两者通常相同，
    但自定义表达式的受控源会被写成行为源 ``B…``，名字会变）。
    """
    if depth > MAX_EXPR_DEPTH:
        raise CircuitError("受控源表达式互相引用成了环，无法写成 SPICE 表达式")

    parts: list[str] = []
    for sym, k in lin.coeffs.items():
        p = _table(circuit).by_symbol(sym)
        if p is None:
            raise CircuitError(f"表达式里的 {sym!r} 不在当前参数表里")
        kind, key = p.binder

        if kind == "node_u":
            frag = "0" if key == circuit.ref_node else f"V({key})"
        elif kind == "branch_u":
            comp = circuit.by_ref(key)
            if comp.outputs_voltage and comp.is_controlled:
                raise CircuitError(
                    f"{sym!r} 是受控源的输出电压，无法直接写成 SPICE 表达式"
                    "（它的值与控制量构成代数环）。请改用节点电压来写。")
            f, t = declared_direction(comp)
            x = "0" if f == circuit.ref_node else f
            y = "0" if t == circuit.ref_node else t
            frag = f"V({x},{y})"
        elif kind == "branch_i":
            frag = _term(circuit.by_ref(key), circuit,
                         netname=netname, depth=depth + 1)
        elif kind == "value":
            comp = circuit.by_ref(key)
            if comp.value is None:
                raise CircuitError(f"{sym!r} 对应的元件 {key} 还没有数值")
            frag = num_str(comp.value)
        else:
            raise CircuitError(f"未知的绑定种类 {kind!r}")

        coef = ""
        if k != 1:
            coef = "-" if k == -1 else f"{num_str(k)}*"
        parts.append(f"{coef}({frag})" if coef else frag)

    out = " + ".join(parts) if parts else ""
    if lin.const != 0:
        c = num_str(lin.const)
        out = f"{out} + {c}" if out else c
    return out or "0"
