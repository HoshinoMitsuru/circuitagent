"""三种解法共用的结果类型。

**为什么先定公共结果类型**：三法要能逐项对账，前提是三者对"同一个量"给出
同一种定义。如果各解法自己解释符号，对账就变成对三套约定，毫无意义。

电流符号的全局约定（与 ir/model.py 完全一致，不再重复推导）：

- 电阻 R：参考方向 ``nodes[0] -> nodes[1]``
- 电流源 I：参考方向 ``nodes[0] -> nodes[1]``（即箭头方向）
- 电压源 V：参考方向 ``nodes[1] -> nodes[0]``
  （电源内部由 − 流向 +，所以为正 = 该电源在供电）

``declared_direction()`` 是唯一出口，任何地方都不许绕过它自己算方向。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..ir.model import (Circuit, Component, CircuitError, SOLVABLE_KINDS,
                        VOLTAGE_OUTPUT_KINDS,
                        declared_direction as _declared_direction)


def reject_non_dc(circuit: Circuit) -> None:
    """拦住"还没化简就硬解"的调用。

    没有这道闸门时，含 C 的电路会掉进求解器的兜底分支，报出
    "暂不支持元件类型 C" 这种让人一头雾水的错 —— 真正的原因其实是
    "你没先做直流稳态化简"。
    """
    bad = sorted({c.ref for c in circuit.components if c.kind not in SOLVABLE_KINDS})
    if bad:
        raise CircuitError(
            f"{bad} 含 C/L 这类非直流元件。求解前必须先做直流稳态化简"
            "（C 视为开路、L 视为短路）：调用 solver.dc_reduce.reduce_to_dc()，"
            "或直接用 solver.reconcile.run_all()，它会自动化简并把过程写进报告。"
        )


def declared_direction(c: Component) -> tuple[str, str]:  # noqa: D103
    """**转发**到 :func:`app.ir.model.declared_direction`，此处只作别名。

    搬家的理由：网表层（``ir/spice.py``）渲染受控源表达式时也要用它，
    而 IR 不该反过来依赖求解层。保留这个同名转发是为了不动散落各处的
    ``from .base import declared_direction``。**改方向约定请改 IR 层那一份。**
    """
    return _declared_direction(c)


@dataclass
class Solution:
    """单条解法路径的完整解。"""

    method: str
    #: 节点电压，键为 IR 节点名
    node_voltages: dict[str, Any] = field(default_factory=dict)
    #: 支路电流，正方向见 declared_direction()
    branch_currents: dict[str, Any] = field(default_factory=dict)
    #: 支路压降 V_from − V_to，from/to 同样取 declared_direction()
    branch_drops: dict[str, Any] = field(default_factory=dict)
    #: 是否为精确有理数解
    exact: bool = False
    #: 解法特有的中间量（如电压源供电电流、网孔电流）
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_float(self) -> dict[str, dict[str, float]]:
        def f(v: Any) -> float:
            if isinstance(v, Fraction):
                return v.numerator / v.denominator
            return float(v)

        return {
            "node_voltages": {k: f(v) for k, v in self.node_voltages.items()},
            "branch_currents": {k: f(v) for k, v in self.branch_currents.items()},
            "branch_drops": {k: f(v) for k, v in self.branch_drops.items()},
        }

    def get_node(self, node: str) -> Any:
        return self.node_voltages.get(node, 0)

    def get_current(self, ref: str) -> Any:
        return self.branch_currents.get(ref)

    def get_drop(self, ref: str) -> Any:
        return self.branch_drops.get(ref)


def branch_drop_from_solution(c: Component, sol: Solution) -> Any:
    """按 IR 约定算这条支路的压降（V_from − V_to，from/to = declared_direction）。"""
    d = sol.branch_drops.get(c.ref)
    if d is not None:
        return d
    f, t = declared_direction(c)
    return sol.get_node(f) - sol.get_node(t)


def build_node_voltages_from_drops(circuit: Circuit, drops: dict[str, Any]) -> dict[str, Any]:
    """给定每条支路的压降，用生成树从参考节点推全节点电压。

    作为独立复算路径的一部分：不依赖任何矩阵解，纯靠"沿着树累加压降"。
    """
    nodes = circuit.nodes
    adj: dict[str, list[tuple[str, Component, str]]] = {n: [] for n in nodes}
    for c in circuit.components:
        f, t = declared_direction(c)
        adj[f].append((t, c, f))
        adj[t].append((f, c, t))

    volt: dict[str, Any] = {circuit.ref_node: Fraction(0)}
    stack = [circuit.ref_node]
    while stack:
        x = stack.pop()
        for y, c, who in adj[x]:
            if y in volt:
                continue
            d = drops[c.ref]                    # = V_f − V_t
            f, t = declared_direction(c)
            volt[y] = volt[x] - d if who == f else volt[x] + d
            stack.append(y)
    return volt
