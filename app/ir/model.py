"""规范中间表示（IR）—— 全项目唯一的电路公共语言。

设计原则（对齐 circuit-photo-to-solution 技能的两条禁令）：

1. **不许靠肉眼定连接** —— 所以 IR 里每一处拓扑判断都必须带 `source`（谁判的）
   与 `confidence`（多大把握）。凡是从照片/位图猜出来的连接，都必须能被
   人工在 WebUI 上核对并改写；凡是从 SVG / KiCad 网表精确解析出来的，
   置信度恒为 1.0 并标记 `source=exact`。
2. **不许只算一遍** —— 所以 IR 只负责"电路长什么样"，求解交给三条互相独立的
   代码路径（节点电压法 / 支路电流法 / ngspice），IR 本身不预设任何解法。

关键约定（与技能文档一字不差地对齐，符号错一个后面全错）：

- 参考节点恒为 ``REF_NODE``（字符串 "0"），对应 SPICE 内部节点 0。
- 电阻 ``R``：参考电流方向 = 图中箭头方向 = ``a → b``，且 ``V_a - V_b = R·i``。
- 电压源 ``V``：``a`` 为 + 端、``b`` 为 − 端，``V_a - V_b = E``；
  支路电流 ``i`` 的参考方向取**电源内部由 − 流向 +**（即外部由 b→a 经 a 流出），
  于是 ``i > 0`` 直接读作"这个电源在供电"，功率符号不需要事后心算。
- 电流源 ``I``：电流参考方向 ``a → b``（箭头），即外部电流由 b 端流出。
- **每条支路的参考方向都按图中所画写，绝不在此处替用户换方向。**

功率符号：吸收为正、提供为负。
- 电阻 ``P = R·i²``
- 电压源 ``P = −E·i``  （``i`` 即"从 + 端流出的电流"）
- 电流源 ``P = (V_a − V_b)·i``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Literal

# ---------------------------------------------------------------- 常量

REF_NODE = "0"

#: 元件种类 -> SPICE 首字母
KIND_LETTER: dict[str, str] = {
    "R": "R",
    "L": "L",
    "C": "C",
    "V": "V",
    "I": "I",
}

#: IR 层面接受的全部元件种类
ALLOWED_KINDS: frozenset[str] = frozenset({"R", "V", "I", "C", "L"})

#: 直流求解器**直接**能解的元件种类。
#: C/L 不在其中 —— 它们必须先经 solver/dc_reduce.py 做直流稳态化简
#: （C 开路、L 短路），化简后 IR 里就只剩 R/V/I 了。
SOLVABLE_KINDS: frozenset[str] = frozenset({"R", "V", "I"})

#: 兼容旧名
DC_SUPPORTED = SOLVABLE_KINDS

SourceKind = Literal["exact", "vision", "cv", "vlm", "manual"]

#: 置信度阈值：低于此值的判断必须走人工确认闸门
CONFIDENCE_GATE = 0.85


class CircuitError(ValueError):
    """IR 层面的结构性错误（拓扑不连通、元件类型不支持等）。"""


class CircuitUnsatisfiable(CircuitError):
    """约束集**自相矛盾** —— 这张电路在直流稳态下无解。

    与 ``CircuitError`` 分成两类，是因为它们要告诉用户完全不同的事：

    - ``CircuitError``：**输入有问题**（漏画线、缺数值、类型不支持）→ 请改图；
    - ``CircuitUnsatisfiable``：**输入没问题，但题目的理想化模型自相矛盾** → 请改题。

    典型的后者：电感在直流下是理想短路，若它与理想电压源并联，就同时要求
    ``V_a − V_b = E`` 和 ``V_a = V_b``。这在教科书里是"该理想电路无解"，
    而不是"程序算不出来"。

    ★ 之所以单开一个类型，是因为本项目**踩过一次**：化简层把"两端等电位的元件"
    一律当作"被短接、等价于消失"删掉，于是矛盾凭空消失，剩下的电路照常可解，
    三条代码路径一致地给出全零解，功率守恒平凡成立（ΣP = 0），
    最后报告 `overall_pass = True` 并告诉学生"**这张电路的读图与建模可以采信**"。
    静默降级到这种程度，必须有一个能被显式抛出的类型来兜住。

    ``contradictions`` 里每条都说明"谁和谁矛盾、矛盾在哪、建议怎么办"，
    供报告逐条呈现，而不是只丢一句"无解"。
    """

    def __init__(self, message: str, contradictions: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.contradictions: list[dict[str, Any]] = list(contradictions or [])


# ---------------------------------------------------------------- 数据类


@dataclass
class Evidence:
    """一处判断的出处。**这是本项目安全性的核心结构**：没有出处的判断不许进 IR。"""

    source: SourceKind = "manual"
    confidence: float = 1.0
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise CircuitError(f"confidence 必须在 [0,1]，收到 {self.confidence}")

    @property
    def needs_human(self) -> bool:
        return self.source != "exact" and self.confidence < CONFIDENCE_GATE


@dataclass
class Component:
    """一条支路（一个二端元件）。

    ``nodes`` 的次序就是参考方向的次序，语义见模块头部约定。
    """

    ref: str                     # 位号，如 "R1"
    kind: str                    # "R" / "V" / "I" / ...
    nodes: tuple[str, str]       # (a, b)：参考方向 a -> b
    value: float | None = None   # R[Ω] / V[V] / I[A]
    evidence: Evidence = field(default_factory=Evidence)
    #: 视觉坐标（仅在从图像解析时有意义），用于回绘与叠图对账
    geom: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        self.ref = str(self.ref).strip()
        self.kind = str(self.kind).strip().upper()
        if not self.ref:
            raise CircuitError("元件位号不能为空")
        if len(self.nodes) != 2:
            raise CircuitError(f"{self.ref}: 目前只支持二端元件，收到 {self.nodes}")
        if self.nodes[0] == self.nodes[1]:
            raise CircuitError(f"{self.ref}: 两端短接到同一个节点 {self.nodes[0]}")

    @property
    def other(self) -> tuple[str, str]:
        return (self.nodes[1], self.nodes[0])

    def value_str(self) -> str:
        if self.value is None:
            return ""
        v = self.value
        return str(int(v)) if float(v).is_integer() else repr(v)

    def to_spice(self) -> str:
        if self.value is None:
            raise CircuitError(f"{self.ref}: 缺少数值，无法生成网表")
        return f"{self.ref} {self.nodes[0]} {self.nodes[1]} {self.value_str()}"


@dataclass
class Probe:
    """要在报告里显式出现的支路电流/电压。

    ``direction`` 必须与图中箭头一致 —— 报告里出现的符号才等于答案的符号。
    """

    kind: Literal["i", "v"] = "i"
    name: str = ""
    ref: str = ""
    direction: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("i", "v"):
            raise CircuitError(f"探针类型只能是 i 或 v，收到 {self.kind}")
        if self.kind == "i" and not self.direction:
            raise CircuitError(f"电流探针 {self.name} 必须给出参考方向")
        if not self.name:
            self.name = f"{self.kind}({self.ref})"


@dataclass
class Circuit:
    """一张电路的规范表示。"""

    name: str = "circuit"
    components: list[Component] = field(default_factory=list)
    ref_node: str = REF_NODE
    probes: list[Probe] = field(default_factory=list)
    #: 从导入过程带出来的诊断信息（读图依据），一并写进报告
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    #: 溯源：这张图从哪来、怎么来的
    origin: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------ 基本查询

    @property
    def nodes(self) -> list[str]:
        """全部节点，参考节点排第一，其余按首次出现次序。"""
        seen: list[str] = []
        for c in self.components:
            for n in c.nodes:
                if n not in seen:
                    seen.append(n)
        if self.ref_node in seen:
            seen.remove(self.ref_node)
        return [self.ref_node] + seen

    @property
    def hot_nodes(self) -> list[str]:
        """除参考节点外的节点（MNA 未知量对应的节点）。"""
        return [n for n in self.nodes if n != self.ref_node]

    def by_ref(self, ref: str) -> Component:
        for c in self.components:
            if c.ref == ref:
                return c
        raise CircuitError(f"找不到位号 {ref}")

    def refs(self) -> list[str]:
        return [c.ref for c in self.components]

    def degree(self, node: str) -> int:
        return sum(1 for c in self.components for n in c.nodes if n == node)

    # ------------------------------------------------ 校验

    def validate(self, allow_incomplete: bool = False) -> list[str]:
        """检查 IR 自洽性。返回警告列表；结构性错误直接抛 CircuitError。

        ``allow_incomplete`` 为 True 时允许缺数值（编辑中途状态）。
        """
        warnings: list[str] = []

        if not self.components:
            raise CircuitError("电路里没有任何元件")

        refs = self.refs()
        dup = {r for r in refs if refs.count(r) > 1}
        if dup:
            raise CircuitError(f"位号重复：{sorted(dup)}")

        if self.ref_node not in self.nodes:
            raise CircuitError(
                f"参考节点 {self.ref_node!r} 没有出现在任何元件端点上，"
                "网表会缺地，无法求解"
            )

        for c in self.components:
            if c.kind not in ALLOWED_KINDS:
                raise CircuitError(
                    f"{c.ref}: 元件类型 {c.kind!r} 不在支持范围 "
                    f"{sorted(ALLOWED_KINDS)}"
                )
            if c.kind in SOLVABLE_KINDS and c.value is None:
                msg = f"{c.ref}: 数值缺失"
                if allow_incomplete:
                    warnings.append(msg)
                else:
                    raise CircuitError(msg + "，请先在结构确认面板填写")
            # C/L 不需要数值：直流稳态化简只用它的"存在"（开路/短路），不用容值感值
            if c.kind in ("C", "L") and c.value is None:
                warnings.append(
                    f"{c.ref}: 缺容值/感值。直流稳态下不需要它（C 开路、L 短路），"
                    "但若以后要算暂态就必须补上"
                )

        # 连通性：位图解析最容易漏线，这里必须先拦住，否则后面矩阵必然奇异
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for c in self.components:
            adj[c.nodes[0]].add(c.nodes[1])
            adj[c.nodes[1]].add(c.nodes[0])
        seen = {self.ref_node}
        stack = [self.ref_node]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        island = [n for n in self.nodes if n not in seen]
        if island:
            raise CircuitError(
                f"电路不连通，以下节点与参考节点 {self.ref_node} 之间没有通路："
                f"{island}。位图解析漏线时最常出现这个错误。"
            )

        # 悬空端点（只接了一个元件）——不致命，但必须提示
        for n in self.nodes:
            if n != self.ref_node and self.degree(n) < 2:
                warnings.append(f"节点 {n} 只接了 {self.degree(n)} 个元件，疑似悬空端点")

        # 电压源直并 / 电流源直串 —— 会让方程奇异，提前拦住
        pair_v: dict[tuple[str, str], list[str]] = {}
        for c in self.components:
            if c.kind == "V":
                key = tuple(sorted(c.nodes))
                pair_v.setdefault(key, []).append(c.ref)
        for key, group in pair_v.items():
            if len(group) > 1:
                warnings.append(f"{group} 是并接在节点 {key} 上的多个电压源，可能导致方程无解")

        return warnings

    def unmet_needs(self) -> list[dict[str, Any]]:
        """列出所有"需要人工确认"的判断，供 WebUI 弹出确认闸门。"""
        out: list[dict[str, Any]] = []
        for c in self.components:
            if c.evidence.needs_human:
                out.append({
                    "ref": c.ref, "kind": c.kind, "nodes": list(c.nodes),
                    "value": c.value, "source": c.evidence.source,
                    "confidence": c.evidence.confidence,
                    "detail": c.evidence.detail,
                })
            elif c.value is None:
                out.append({
                    "ref": c.ref, "kind": c.kind, "nodes": list(c.nodes),
                    "value": None, "source": c.evidence.source,
                    "confidence": c.evidence.confidence,
                    "detail": "缺少数值，需人工填写",
                })
        return out

    # ------------------------------------------------ 序列化

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["components"] = [
            {**asdict(c), "nodes": list(c.nodes)} for c in self.components
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Circuit":
        comps = []
        for raw in d.get("components", []):
            ev = raw.get("evidence") or {}
            comps.append(Component(
                ref=raw["ref"], kind=raw["kind"],
                nodes=tuple(raw["nodes"]),
                value=raw.get("value"),
                evidence=Evidence(**ev) if isinstance(ev, dict) else ev,
                geom=raw.get("geom") or {},
                note=raw.get("note", ""),
            ))
        probes = []
        for raw in d.get("probes", []):
            direction = raw.get("direction")
            probes.append(Probe(
                kind=raw.get("kind", "i"), name=raw.get("name", ""),
                ref=raw.get("ref", ""),
                direction=tuple(direction) if direction else None,
            ))
        return cls(
            name=d.get("name", "circuit"), components=comps,
            ref_node=d.get("ref_node", REF_NODE), probes=probes,
            diagnostics=d.get("diagnostics", []),
            origin=d.get("origin", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "Circuit":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------ 生成辅助

    @staticmethod
    def auto_ref(kind: str, existing: Iterable[str]) -> str:
        """自动分配不冲突的位号。"""
        used = set(existing)
        i = 1
        while f"{kind}{i}" in used:
            i += 1
        return f"{kind}{i}"

    def copy(self) -> "Circuit":
        return Circuit.from_dict(self.to_dict())
