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
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

# ★ 错误类型住在 errors.py（见那里的模块注释：拆出去是为了打断
#   model ↔ params 的循环导入）。这里 re-export，既有的
#   ``from .model import CircuitError`` 一处都不用改。
from .errors import CircuitError, CircuitUnsatisfiable
# ★ 依赖方向是一条线：errors ← params ← model。
#   params 只在 TYPE_CHECKING 下引用 model，所以这里是安全的运行时导入。
from .params import CONTROL_MODE, CONTROL_NOTE, GAIN_SYMBOL, ParamTable

if TYPE_CHECKING:                                # 只有类型检查期需要
    pass

# ---------------------------------------------------------------- 常量

REF_NODE = "0"

#: 元件种类 -> SPICE 首字母
KIND_LETTER: dict[str, str] = {
    "R": "R",
    "L": "L",
    "C": "C",
    "V": "V",
    "I": "I",
    "E": "E",
    "G": "G",
    "H": "H",
    "F": "F",
}

#: 独立源：数值由题目给定，不依赖电路里别处的量。
INDEPENDENT_KINDS: frozenset[str] = frozenset({"R", "V", "I", "C", "L"})

#: ★ 受控源（dependent / controlled source）—— 四类，正是 SPICE 的四张标准卡。
#:
#: | kind | 名称 | 输出 | 控制量 | 增益符号 | SPICE 卡 |
#: |---|---|---|---|---|---|
#: | ``E`` | 电压控制电压源 VCVS | 电压 | 电压 | μ | ``E out+ out- c+ c- μ`` |
#: | ``G`` | 电压控制电流源 VCCS | 电流 | 电压 | gm | ``G out+ out- c+ c- gm`` |
#: | ``H`` | 电流控制电压源 CCVS | 电压 | 电流 | rm | ``H out+ out- Vsense rm`` |
#: | ``F`` | 电流控制电流源 CCCS | 电流 | 电流 | α | ``F out+ out- Vsense α`` |
#:
#: **为什么要区分"受控源 / 非受控源"**：受控源不是二端元件，它有一条**额外的
#: 控制支路**。拓扑上它只往电路里接两个端子，但那两个顶点的电位差（或支路电流）
#: 会反过来决定它自己的输出。所以：
#:   * 画图时它要额外声明"控制量取自哪里"；
#:   * 求解时它给不出独立的支路方程，要额外引入控制项；
#:   * 报告里它的"数值"其实是**增益**，单位与 R/L/C 完全不同
#:     （V/V、Ω、S、A/A，见 params.VALUE_UNIT）。
#: 把两者混为一谈，就会在"它到底是个电阻还是条方程"上出错。
CONTROLLED_KINDS: frozenset[str] = frozenset({"E", "G", "H", "F"})

#: 输出**电压**的元件种类：像理想电压源那样占一个支路电流未知量。
#: ★ 这个集合是"支路电流参考方向"与"被当成电压源处理"的唯一出口 ——
#:   base.declared_direction / mna / branch / spice 都只认它，不再各自写
#:   ``c.kind == "V"``。散着写正是"漏改一处就静默算错"的经典成因。
VOLTAGE_OUTPUT_KINDS: frozenset[str] = frozenset({"V", "E", "H"})

#: 输出**电流**的元件种类：像理想电流源那样电流由自己（或控制量）决定。
CURRENT_OUTPUT_KINDS: frozenset[str] = frozenset({"I", "G", "F"})

#: IR 层面接受的全部元件种类
ALLOWED_KINDS: frozenset[str] = INDEPENDENT_KINDS | CONTROLLED_KINDS

#: 直流求解器**直接**能解的元件种类。
#: C/L 不在其中 —— 它们必须先经 solver/dc_reduce.py 做直流稳态化简
#: （C 开路、L 短路），化简后 IR 里就只剩 R/V/I 了。
#: 受控源本身是直流可解的（它是代数约束，不含储能元件）。
SOLVABLE_KINDS: frozenset[str] = frozenset({"R", "V", "I"}) | CONTROLLED_KINDS

#: 兼容旧名
DC_SUPPORTED = SOLVABLE_KINDS

#: ★ **位图通道（照片/截图）能认出形状的符号**。目前只有独立源那一套。
#:
#: 为什么要有这个常量、而不是让视觉层继续"遍及 ``ALLOWED_KINDS``"：
#: 这两种范围的**含义不同**——``ALLOWED_KINDS`` 是"IR 装得下的类型"，
#: ``BITMAP_KINDS`` 是"模板库里有画法、能靠形状匹配出来的类型"。
#: 混用会以两种方式出错，两种都实测过：
#:
#: * 视觉层的模板自检去枚举 ``ALLOWED_KINDS`` → 撞上受控源 →
#:   ``ValueError: 没有 'E' 这个符号的画法`` → ``/api/health`` 直接 500；
#: * 更像隐患的那种：VLM 返回 ``kind="E"``，而"在支持范围内"这句判断
#:   用的是 ``ALLOWED_KINDS``，于是它被当成一个**普通二端元件**建出来 ——
#:   受控源的控制关系凭空消失，电路却照样算得动。
#:
#: 受控源要进位图通道，得先有符号画法（``render.GLYPHS``）与模板，
#: 再把它加进来。在那之前，位图通道遇到受控源的做法是**明确报出来**，
#: 而不是假装认识。
BITMAP_KINDS: frozenset[str] = INDEPENDENT_KINDS

# GAIN_SYMBOL / CONTROL_NOTE 从 .params 转出（那里是 IR 的最底层）

SourceKind = Literal["exact", "vision", "cv", "vlm", "manual"]

#: 置信度阈值：低于此值的判断必须走人工确认闸门
CONFIDENCE_GATE = 0.85


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
class Control:
    """受控源的**控制支路** —— 这才是它与二端元件的真正区别。

    普通元件"自己的方程只含自己"：``R`` 是 ``u = R·i``、``V`` 是 ``u = E``。
    受控源不是：它的输出由一个**别处的量**决定，所以它必须额外声明"取自哪里"。

    两种取法（正好对应 SPICE 的四张标准卡）：

    * ``mode="V"``（电压控制）—— 控制量是 ``nodes`` 两端的电位差
      ``V(nodes[0]) − V(nodes[1])``。它**不消耗**这两个端点上的电流，
      纯粹是"看一眼"。对应 ``E``（VCVS）与 ``G``（VCCS）。
    * ``mode="I"``（电流控制）—— 控制量是某条支路的电流。对应
      ``H``（CCVS）与 ``F``（CCCS）。

    ★ **``mode="I"`` 是这一层最容易出错的地方。** SPICE 的 ``H``/``F`` 卡
    只能引用**电压源**的电流（``H1 out+ out- Vsense rm``），写不了
    ``H1 out+ out- R1 rm``；而教科书题目常说的是"受 ``R1`` 上的电流控制"。
    本项目的处理是**在那条支路里插一个 0V 电压源当电流探针**，使它变成
    可以取电流的支路：

      * 题目给出的 ``ref = "R1"``（保留原意，报告里要能对上题目说法）；
      * 系统插入 ``sense_ref = "Vsense_E1"``（0V 源，与 ``R1`` 仍是串联关系，
        电路的**电学行为完全不变**）；
      * 三条求解路径统一把它当电压源处理，控制量就是那个 0V 源的电流。

    插进来的 0V 源会在参数表里留一行、在报告里标明"**测量探针引入、
    不是题目元件**"—— 它会让节点数/支路数各多一个，必须让人看得见，
    否则就是"题目里没有的东西偷偷进了答案"。
    """

    mode: str = "V"                              # "V" 电压控制 / "I" 电流控制
    #: ``mode="V"``：控制电压的两端 ``(+, −)``，与元件自身端子无关
    nodes: tuple[str, str] | None = None
    #: ``mode="I"``：**题目里说的**那条被采样支路的位号（如 ``R1``）
    ref: str = ""
    #: 可选的自定义表达式（用参数名书写）。**为空 = 标准形**
    #: ``输出 = 增益 × 控制量``——此时求解完全不需要参数表参与。
    expr: str = ""
    #: ``mode="I"``：为取电流而**自动插入**的 0V 探针源位号。
    #: 被采样支路本身已经是电压源时留空（直接引用它即可）。
    sense_ref: str = ""

    def __post_init__(self) -> None:
        self.mode = str(self.mode or "V").strip().upper()
        if self.mode not in ("V", "I"):
            raise CircuitError(
                f"受控源的控制量只能是 V（电压）或 I（电流），收到 {self.mode!r}")
        self.ref = str(self.ref or "").strip()
        self.sense_ref = str(self.sense_ref or "").strip()
        self.expr = str(self.expr or "").strip()
        if self.nodes is not None:
            self.nodes = (str(self.nodes[0]), str(self.nodes[1]))
        if self.mode == "V":
            if not self.nodes or len(self.nodes) != 2:
                raise CircuitError("电压控制型受控源必须给出控制端对 (+, −)")
            if self.nodes[0] == self.nodes[1]:
                raise CircuitError(
                    f"电压控制型受控源的控制端是同一个节点 {self.nodes[0]!r}，"
                    "控制量恒为 0，等于一个值为 0 的独立源 —— 多半是画错了")
        else:
            if not self.ref:
                raise CircuitError("电流控制型受控源必须给出被采样支路的位号")

    @property
    def sampling(self) -> str:
        """实际用来取电流的那条支路位号（优先探针源）。"""
        return self.sense_ref or self.ref

    def describe(self, kind: str, value: float | None) -> str:
        """人读的控制关系，如 ``2·(V(3) − V(0))``。

        ★ 电流控制 + 插了探针时，显示的是**题目里说的那条支路**，
        并在括号里点明"经 0V 探针取出"。不这么写就会出现一个很难看懂的局面：
        报告上写着 ``2000·i(Vsense_R1)``，而用户翻遍题目也没有 ``Vsense_R1``
        这个元件 —— 那是求解器为了取电流自己串进去的。
        **名字必须指向用户认得的东西，机制另说。**
        """
        gain = "?" if value is None else (
            str(int(value)) if float(value).is_integer() else repr(value))
        if self.mode == "V" and self.nodes:
            ctrl = f"(V({self.nodes[0]}) − V({self.nodes[1]}))"
        elif self.sense_ref and self.sense_ref != self.ref:
            ctrl = f"i({self.ref})（经 0V 探针 {self.sense_ref} 取出，不是题目元件）"
        else:
            ctrl = f"i({self.sampling})"
        if self.expr:
            return f"{self.expr}   ← 自定义表达式（标准形为 {gain}·{ctrl}）"
        return f"{gain}·{ctrl}"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode,
                "nodes": list(self.nodes) if self.nodes else None,
                "ref": self.ref, "expr": self.expr, "sense_ref": self.sense_ref}

    @classmethod
    def from_dict(cls, d: "Mapping[str, Any] | None") -> "Control | None":
        if not d:
            return None
        n = d.get("nodes")
        return cls(mode=d.get("mode", "V"),
                   nodes=tuple(n) if n else None,
                   ref=d.get("ref", ""), expr=d.get("expr", ""),
                   sense_ref=d.get("sense_ref", ""))


@dataclass
class Component:
    """一条支路（一个二端元件）。

    ``nodes`` 的次序就是参考方向的次序，语义见模块头部约定。
    """

    ref: str                     # 位号，如 "R1"
    kind: str                    # "R" / "V" / "I" / ... / "E" / "G" / "H" / "F"
    nodes: tuple[str, str]       # (a, b)：参考方向 a -> b
    #: R[Ω] / V[V] / I[A] / 受控源增益（单位由 kind 决定，见 params.VALUE_UNIT）
    value: float | None = None
    evidence: Evidence = field(default_factory=Evidence)
    #: 视觉坐标（仅在从图像解析时有意义），用于回绘与叠图对账
    geom: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    #: 受控源的控制支路。**非受控源必须为 None** —— 见 __post_init__ 的互斥校验。
    ctrl: "Control | None" = None

    def __post_init__(self) -> None:
        self.ref = str(self.ref).strip()
        self.kind = str(self.kind).strip().upper()
        if not self.ref:
            raise CircuitError("元件位号不能为空")
        if len(self.nodes) != 2:
            raise CircuitError(f"{self.ref}: 目前只支持二端元件，收到 {self.nodes}")
        if self.nodes[0] == self.nodes[1]:
            raise CircuitError(f"{self.ref}: 两端短接到同一个节点 {self.nodes[0]}")
        if isinstance(self.ctrl, dict):
            self.ctrl = Control.from_dict(self.ctrl)
        # ★ 受控源与非受控源必须泾渭分明。两者混淆不会当场报错，
        #   而会在很久以后的求解里表现为"某个方程莫名其妙"——
        #   所以在这里就把话说死。
        if self.kind in CONTROLLED_KINDS:
            if self.ctrl is None:
                raise CircuitError(
                    f"{self.ref}: {self.kind} 是受控源，必须给出控制支路"
                    "（电压控制要控制端对，电流控制要被采样支路位号）")
        elif self.ctrl is not None:
            raise CircuitError(
                f"{self.ref}: {self.kind} 不是受控源，不该带控制支路 —— "
                "带上了说明上游把受控源和普通元件搞混了")

    @property
    def is_controlled(self) -> bool:
        return self.kind in CONTROLLED_KINDS

    @property
    def outputs_voltage(self) -> bool:
        """它像电压源那样给"电压约束"，还是像电流源那样给"电流"。"""
        return self.kind in VOLTAGE_OUTPUT_KINDS

    @property
    def control_terminals(self) -> tuple[str, str] | None:
        """电压控制时控制端对所涉及的节点（报告与拓扑检查要用）。"""
        if self.ctrl is None or self.ctrl.mode != "V":
            return None
        return self.ctrl.nodes

    def control_text(self) -> str:
        """控制关系的可读描述（受控源才有）。"""
        if self.ctrl is None:
            return ""
        return self.ctrl.describe(self.kind, self.value)

    @property
    def other(self) -> tuple[str, str]:
        return (self.nodes[1], self.nodes[0])

    def value_str(self) -> str:
        if self.value is None:
            return ""
        v = self.value
        return str(int(v)) if float(v).is_integer() else repr(v)

    def to_spice(self, circuit: "Circuit | None" = None) -> str:
        """单张元件卡。

        ★ 受控源的卡**形状与二端元件完全不同**（多一列控制量），
        所以这里直接交给 :mod:`app.ir.spice` 里的统一实现，绝不在这里
        再写一份 —— 两份写法迟早会分叉，而分叉的表现是"网表看着没问题、
        ngspice 算出来是另一个电路"。

        ``circuit`` 不是必需的，但**受控源必须要**：控制端节点名要映射成
        SPICE 的地写法，电流控制型还要去查采样支路、自定义表达式还要按
        参数表展开。缺了它这里只能报错，不能猜。
        """
        if self.kind in CONTROLLED_KINDS:
            if circuit is None:
                raise CircuitError(
                    f"{self.ref}: 受控源的网表卡需要电路上下文（控制端节点名要映射、"
                    "电流控制型要查采样支路、自定义表达式要按参数表展开）。"
                    "请用 app.ir.spice.to_spice(circuit)，不要单独取这一张卡。")
            from .spice import controlled_card
            return controlled_card(self, self.nodes[0], self.nodes[1], circuit)
        if self.value is None:
            raise CircuitError(f"{self.ref}: 缺少数值，无法生成网表")
        return f"{self.ref} {self.nodes[0]} {self.nodes[1]} {self.value_str()}"


def declared_direction(c: Component) -> tuple[str, str]:
    """该支路电流的参考方向（**IR 的符号约定，唯一出口**）。

    **电压输出元件特殊**（理想电压源 ``V``，以及电压型受控源 ``E``/``H``）：
    IR 里 ``nodes[0]`` 是 + 端，但电流参考方向取元件**内部**由 − 流向 +，
    也就是 ``nodes[1] -> nodes[0]``。这样"i > 0"直接读作"这个元件在输出功率"，
    功率符号不用事后心算。其余元件一律 ``nodes[0] -> nodes[1]``。

    ★ 判据必须是 ``VOLTAGE_OUTPUT_KINDS`` 而**不是** ``kind == "V"``。
    这里踩过一次：加了受控源之后只改了 IR 层的常量，忘了改这一处，
    于是 ``E``/``H`` 被当作普通元件。症状是"节点电压法对、支路电流法把
    输出电压的符号搞反"—— 两条路径本该逐项严格相等，于是对账立刻报差异。
    这次错得很显眼是运气；换个不参与输出符号的场合就是静默错值。

    ★ 它住在 IR 层而不是求解层：**网表层渲染受控源表达式时也要用它**
    （把 ``V(a,b)`` 写成 ngspice 认的节点顺序）。放在求解层就会逼出
    "网表那边再写一份方向判据" —— 两份判据迟早分叉，而分叉的表现是
    "网表看着没问题、ngspice 算出来是另一个电路"。
    """
    if c.kind in VOLTAGE_OUTPUT_KINDS:
        return (c.nodes[1], c.nodes[0])
    return (c.nodes[0], c.nodes[1])


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
    #: ★ 参数表（名字 ↔ 绑定）。见 app/ir/params.py。
    #:   它是**派生层**：即使整张表被清空，电路与求解一切照旧
    #:   （sync 会按 ref/结点名重新派生一遍默认名）。
    #:   反过来说，它也**绝不**参与矩阵装配 —— 求解路径只认 ref 与结点名。
    params: ParamTable = field(default_factory=ParamTable)

    # ------------------------------------------------ 基本查询

    @property
    def nodes(self) -> list[str]:
        """全部**电路节点**，参考节点排第一，其余按首次出现次序。

        ★ 只收元件端子，**不含受控源的控制端**。理由：控制端是"看一眼"，
        它在电路里不接任何东西，也不引入新的未知量。把它算进来会让
        "节点数"与 MNA 未知量数对不上，进而让报告里的 b−n+1 等口径失真。
        控制端节点的存在性由 :meth:`validate` 单独校验（见 ``control_nodes``）。
        """
        seen: list[str] = []
        for c in self.components:
            for n in c.nodes:
                if n not in seen:
                    seen.append(n)
        if self.ref_node in seen:
            seen.remove(self.ref_node)
        return [self.ref_node] + seen

    @property
    def control_nodes(self) -> list[str]:
        """全部受控源**控制端**所引用的节点（含重复，去重后返回）。"""
        out: list[str] = []
        for c in self.components:
            for n in (c.control_terminals or ()):
                if n not in out:
                    out.append(n)
        return out

    @property
    def controlled(self) -> list[Component]:
        """受控源（E/G/H/F）。报告与 UI 靠它把两类元件分开摆。"""
        return [c for c in self.components if c.is_controlled]

    def sense_sources(self) -> list[Component]:
        """为取电流而自动插进来的 0V 探针源。

        ★ 单独列出来是因为它们**不是题目元件**：报告里必须与真实元件分开呈现，
        否则"题目里没画的东西出现在答案里"就成了静默降级。
        """
        refs = {c.ctrl.sense_ref for c in self.controlled
                if c.ctrl is not None and c.ctrl.sense_ref}
        return [c for c in self.components if c.ref in refs]

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

        # ★ 先把参数表同步到当前电路，**再**校验受控源的表达式。
        #   不同步就校验的话，参数表里一个名字都没有，于是任何一个表达式
        #   都会被判成"未知符号"—— 报错长这样：「可用的名字有：。」
        #   而电路其实完全正常。sync 是幂等的、且不改变电路语义
        #   （它只维护"名字 ↔ 绑定"这张派生表），所以放在校验里是安全的。
        self.sync_params()

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
            # ★ 受控源可以"用表达式代替增益"：`ctrl.expr` 非空时它自己就是
            #   完整的受控关系，`value` 只是标准形参考值，缺了不算错。
            has_expr = c.ctrl is not None and bool(c.ctrl.expr)
            if c.kind in SOLVABLE_KINDS and c.value is None and not has_expr:
                msg = f"{c.ref}: 数值缺失"
                if c.is_controlled:
                    msg = (f"{c.ref}: 受控源缺少增益（{GAIN_SYMBOL.get(c.kind, '')}）"
                           "，也没有自定义表达式")
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
            # ---- 受控源专有校验
            if c.is_controlled:
                warnings.extend(self._validate_controlled(c, allow_incomplete))

        # ★ 受控源的控制端必须指向**电路里真实存在**的节点。
        #   控制端是"看一眼"，不接任何东西，所以它不会出现在元件端子表里，
        #   也就不会被连通性检查兜住。而一旦它指向一个不存在的节点，
        #   控制量会**静默取 0**（两侧都查不到电位）：方程照常解出来、
        #   三条路径还会一致地给出同一个错答案 —— 这正是三法互校
        #   拦不住的那一类。所以在这里拦死。
        known_nodes = set(self.nodes)
        for c in self.components:
            for n in (c.control_terminals or ()):
                if n not in known_nodes:
                    raise CircuitError(
                        f"{c.ref}: 控制端引用了节点 {n!r}，但电路里没有这个节点"
                        f"（现有节点：{sorted(known_nodes)}）。受控源的控制端必须接在"
                        "真实节点上 —— 否则控制量会静默取 0，方程照样解得出来、"
                        "而答案全错。")

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

        # 电压输出元件直并 —— 会让方程奇异，提前拦住。
        # ★ 判据是 outputs_voltage（V / E / H），不再是 ``kind == "V"``：
        #   受控源 E/H 与理想电压源并接同样是过约束，只认 V 就会漏掉它，
        #   而漏掉的后果是"矩阵奇异"这种指不清原因的报错。
        pair_v: dict[tuple[str, str], list[str]] = {}
        for c in self.components:
            if c.outputs_voltage:
                key = tuple(sorted(c.nodes))
                pair_v.setdefault(key, []).append(c.ref)
        for key, group in pair_v.items():
            if len(group) > 1:
                warnings.append(
                    f"{group} 是并接在节点 {key} 上的多个电压输出元件"
                    "（理想电压源或电压型受控源），可能导致方程无解")

        return warnings

    def _validate_controlled(self, c: Component,
                             allow_incomplete: bool = False) -> list[str]:
        """受控源专有的自洽检查。返回警告；结构性问题直接抛。"""
        warns: list[str] = []
        ctrl = c.ctrl
        assert ctrl is not None                      # Component.__post_init__ 已保证

        by_ref = {x.ref: x for x in self.components}

        if ctrl.mode == "I":
            if ctrl.ref not in by_ref:
                raise CircuitError(
                    f"{c.ref}: 电流控制型受控源要采样的支路 {ctrl.ref!r} 在电路里"
                    f"不存在（现有位号：{sorted(by_ref)}）。")
            target = by_ref[ctrl.ref]
            if target is c:
                raise CircuitError(
                    f"{c.ref}: 电流控制型受控源不能采样自己的电流（自指）—— "
                    "这会给出一条自己等于自己的方程。")
            if ctrl.sense_ref and ctrl.sense_ref not in by_ref:
                raise CircuitError(
                    f"{c.ref}: 记录的 0V 探针源 {ctrl.sense_ref!r} 在电路里不存在。"
                    "（这一条通常说明电路被手工改过而探针没跟着走）")
            if ctrl.expr:
                # 表达式自己就能按参数名取到电流，不需要探针源 ——
                # 所以这里**不能**预告"会插探针"，否则用户会去找一个不存在的元件。
                pass
            elif not ctrl.sense_ref and not target.outputs_voltage:
                # 不是错误：求解前会自动插入探针。但必须**预告**，
                # 因为用户会看到拓扑里多出一个自己没画的元件。
                warns.append(
                    f"{c.ref}: 要取 {ctrl.ref} 的电流，但 SPICE 只能引用电压源的电流 —— "
                    f"求解时会自动在 {ctrl.ref} 支路里串一个 0V 电压源做电流探针"
                    "（电学行为不变，但会多出一个节点与一条支路，参数表里也会有它一行）")
            elif ctrl.sense_ref and target.outputs_voltage:
                # 被采样支路本来就是电压源，却还插了探针 —— 白多一个元件
                warns.append(
                    f"{c.ref}: 被采样的 {ctrl.ref} 本身就是电压源，可以直接取它的电流，"
                    f"却又记了一个 0V 探针源 {ctrl.sense_ref!r}（多余）")

        # 自定义表达式必须能解析出来
        if ctrl.expr:
            try:
                self.params.resolve_expression(ctrl.expr)
            except CircuitError as e:
                raise CircuitError(f"{c.ref}: 自定义表达式有问题 —— {e}") from None
            if not allow_incomplete:
                std = self.params.resolve_expression(ctrl.expr).is_standard_single()
                if std is None:
                    # 不是标准形并**不等于错**：它的物理含义只有 ngspice 那一路
                    # 能原样接受（B 源），两条精确路径要靠线性展开。
                    warns.append(
                        f"{c.ref}: 自定义表达式不是「单个控制量 × 系数」的标准形，"
                        "本工具会按线性组合展开交给两条精确路径 —— "
                        "展开结果会在报告里列出，请核对它与题目是否一致")

        return warns

    def sync_params(self) -> dict[str, Any]:
        """把参数表对齐到当前电路（保留旧命名、补新绑定、标孤儿）。"""
        return self.params.sync(self)

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
                ctrl=Control.from_dict(raw.get("ctrl")),
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
            params=ParamTable.from_dict(d.get("params")),
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
