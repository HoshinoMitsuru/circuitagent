"""IR <-> SPICE 网表序列化。

网表是本项目的第二公共语言：ngspice 直接吃它，KiCad 能导它，
人也能一眼看懂。**每条支路的参考方向按 IR 原样落盘**，不做任何自动换向。

★ 唯一的例外是 ``H``/``F`` 卡（电流控制型受控源）的增益要带一个负号。
那不是"自动换向"，是两个约定本身的差：IR 的电源电流 ``i > 0`` = 供出功率
（元件内部 −→+），SPICE 的 ``I(Vx)`` 正号 = 由 + 端流入。换算因子只在
:data:`app.ir.spice_expr.SPICE_SOURCE_CURRENT_SIGN` 定义一次，
读回来时同样要反一次（所以 ``from_spice(to_spice(c))`` 依然无损）。
网表里会对这种卡写一行注释说明，不让人以为是我们抄错了数。
"""

from __future__ import annotations

import re
from typing import Any

from .model import (Circuit, Component, CircuitError, Control, Evidence,
                    CONTROLLED_KINDS, CONTROL_NOTE, GAIN_SYMBOL)
from .params import CONTROL_MODE as _CONTROL_MODE
from .probes import ensure_sense_sources
from .spice_expr import num_str, spice_expression, spice_gain_text
# 复用工程记法解析器（1k / 4k7 / 1meg / 10u，含 M/m 约定冲突警告）。
# 直接引模块而非 ingest 包，避免将来 ingest 包 aggregate 导入时绕回本文件形成环。
from ..ingest.values import parse_engineering

# SPICE 首字母 -> 我们的 kind。注意 V 在 KiCad 里可能是 VDD 之类，交给上游判定。
LETTER_KIND = {"R": "R", "L": "L", "C": "C", "V": "V", "I": "I",
               "E": "E", "G": "G", "H": "H", "F": "F"}

#: 受控源首字母 -> 控制量的种类："V" 由两支点电压控制、"I" 由一条支路电流控制。
#: 这不是我们自己定的，是 SPICE 卡的形状决定的：``E``/``G`` 后面跟 **两个节点**，
#: ``H``/``F`` 后面跟 **一个电压源位号**。
#: ★ 定义已上移到 ``params.CONTROL_MODE``（它是**元件类型的固有属性**，不是网表格式的事），
#:   这里转发一下只是为了 `spice.CONTROL_MODE` 这个旧名字还能用。
CONTROL_MODE = _CONTROL_MODE

#: 机器可读的受控源语义标记（写在网表注释行上）。
#:
#: ★ 为什么非要有它：``H``/``F`` 卡只能引用**电压源**的电流，所以电流控制型
#: 受控源在网表里引用的是**系统插入的 0V 探针源**（``Vsense_R1``），
#: 而不是题目里说的那条支路（``R1``）。光看卡片，读回来的人**无法分辨**
#: 这两个名字 —— 于是往返一次之后 ``ctrl.ref`` 就从 ``R1`` 变成了 ``Vsense_R1``：
#: 电学上完全等价（``sampling`` 取的都是探针），但**参数指向已经错了** ——
#: 界面上「被采样支路」会显示成一个系统探针，用户看到的控制关系
#: 从 ``i(R1)`` 变成 ``i(Vsense_R1)``。
#:
#: 这与回绘 SVG 用 ``data-ca-*`` 是同一套办法：**卡片/几何仍是权威**
#: （谁接谁、增益多少只认卡片），标记只补"身份"这一类卡片本身表达不了的信息。
#: 行首是 ``*``，ngspice 当注释忽略；外来网表没有这一行就原样不动。
CTRL_MARK = "* ca-ctrl"

#: 语义标记行的解析式。命名空间 ``ca-ctrl`` 是我们自己的，不会误吃外来的注释。
_CTRL_MARK_RE = re.compile(r"^\*\s*ca-ctrl\s+(?P<ref>\S+)\s*(?P<fields>.*)$",
                           re.IGNORECASE)


def _ctrl_mark(comp: Component) -> str:
    """这条受控源的语义标记行；**没有信息会丢时返回空串**。

    只有一种情况需要它：电流控制、且实际采样的是**系统插入的探针**。
    其余情况卡片本身已经写全了 ——

    * ``E``/``G``：控制端两个节点就写在卡上；
    * 被采样支路本身就是电压源时，卡上引用的就是那条支路自己，读回来无损；
    * 自定义表达式落成行为源 ``B`` 卡，而 ``B`` 卡读回来会**明确报错**，
      不存在"读回来悄悄丢了东西"的余地。
    """
    ctrl = comp.ctrl
    if ctrl is None or ctrl.mode != "I":
        return ""
    if not ctrl.sense_ref or ctrl.sense_ref == ctrl.ref:
        return ""
    return (f"{CTRL_MARK} {comp.ref} mode=I ref={ctrl.ref} sense={ctrl.sense_ref}")



def _netname(c: Component) -> str:
    """这条支路在**网表里**的器件名。

    与位号通常一致（``R1`` → ``R1``）；唯一例外是用了自定义表达式的受控源：
    它在 SPICE 里只能写成行为源，器件名会变成 ``B`` + 位号。
    凡是"按名字取这条支路的电流"的地方（表达式、``H``/``F`` 控制端）
    都必须走这里，否则会去取一个不存在的器件。
    """
    if c.is_controlled and c.ctrl is not None and c.ctrl.expr:
        return "B" + c.ref
    return c.ref


def _params(circuit: Circuit):
    if not circuit.params.items:
        circuit.sync_params()
    return circuit.params


def _two_terminal_card(c: Component, a: str, b: str) -> str:
    if c.value is None:
        raise CircuitError(f"{c.ref} 缺数值，拒绝生成网表（宁可不给，也不给错）")
    # 电压源显式写 DC，避免 ngspice 在 .op 下自作主张
    if c.kind == "V":
        return f"{c.ref} {a} {b} DC {c.value_str()}"
    return f"{c.ref} {a} {b} {c.value_str()}"


def controlled_card(comp: Component, a: str, b: str,
                    circuit: Circuit | None = None) -> str:
    """受控源的**一张** SPICE 卡。

    * ``E``/``G``：``Exx a b cx cy 增益`` —— 控制量是两个节点之间的电压。
    * ``H``/``F``：``Hxx a b 电压源位号 增益`` —— 控制量是那条电压源的电流。
      卡上的增益是 **IR 增益取反**，见 :data:`spice_expr.SPICE_SOURCE_CURRENT_SIGN`。
    * 自定义表达式：写成行为源 ``Bxx a b V = <表达式>`` / ``I = …``。
      这时网表器件名是 ``B`` + 位号，与位号不同 —— 由 :func:`_netname` 映射。
    """
    ctrl = comp.ctrl
    if ctrl is None:
        raise CircuitError(
            f"{comp.ref}: 受控源缺少控制支路，写不出网表卡（{CONTROL_NOTE.get(comp.kind, '')}）")

    # ---- 自定义表达式 → 行为源
    if ctrl.expr:
        if circuit is None:
            raise CircuitError(
                f"{comp.ref}: 自定义表达式要写成 SPICE 文本，必须先知道电路 —— "
                "表达式里的参数名要翻回位号与节点名，器件名也要按网表口径映射。"
                "请用 spice.to_spice(circuit) 生成整张网表，不要单独取这一张卡。")
        lin = _params(circuit).resolve_expression(ctrl.expr)
        text = spice_expression(lin, circuit, netname=_netname)
        field = "V" if comp.outputs_voltage else "I"
        return f"B{comp.ref} {a} {b} {field} = {text}"

    if comp.value is None:
        raise CircuitError(
            f"{comp.ref}: 受控源既缺增益（{GAIN_SYMBOL.get(comp.kind, '')}）"
            "也没有自定义表达式，拒绝生成网表")

    if ctrl.mode == "V":
        if not ctrl.nodes:
            raise CircuitError(f"{comp.ref}: 电压控制型受控源缺少控制端节点对")
        x, y = ctrl.nodes
        if circuit is not None:
            x = "0" if x == circuit.ref_node else x
            y = "0" if y == circuit.ref_node else y
        return f"{comp.ref} {a} {b} {x} {y} {comp.value_str()}"

    # 电流控制型：卡上要写的是**先导电压源的名字**
    sense_ref = ctrl.sampling
    if circuit is not None:
        try:
            sense_c = circuit.by_ref(sense_ref)
        except CircuitError:
            sense_c = None
        if sense_c is not None:
            if not sense_c.outputs_voltage:
                raise CircuitError(
                    f"{comp.ref}: 它的控制量是 {sense_ref} 的电流，"
                    "而 SPICE 的 H/F 卡只能引用**电压源**的电流。"
                    "请先用 app.ir.probes.ensure_sense_sources() 展开电路"
                    "（to_spice 默认会做），或在电路里直接以电压源作为采样支路。")
            sense_ref = _netname(sense_c)
    return f"{comp.ref} {a} {b} {sense_ref} {spice_gain_text(comp.value)}"


def controlled_note(comp: Component) -> str:
    """受控源这一行**要额外说明什么**（网表注释与界面提示共用一份文字）。"""
    ctrl = comp.ctrl
    if ctrl is None:
        return ""
    if ctrl.expr:
        return (f"自定义表达式，网表里写成行为源 —— "
                f"**网表器件名是 {_netname(comp)}，与位号 {comp.ref} 不同**，"
                "按名字取它的电流要用前者")
    if ctrl.mode == "V" and ctrl.nodes:
        return (f"{CONTROL_NOTE.get(comp.kind, '')}；由 "
                f"V({ctrl.nodes[0]}) − V({ctrl.nodes[1]}) 控制")
    if comp.kind in ("H", "F"):
        ir_val = comp.value_str()
        card_val = spice_gain_text(comp.value) if comp.value is not None else "?"
        # ★ 说的必须是**题目里那条支路**（``ctrl.ref``），不是探针位号：
        #   网表注释是给人看的，而人翻遍题目也找不到 Vsense_R1。
        #   探针另说一句，并且明说它不是题目元件。
        if ctrl.sense_ref and ctrl.sense_ref != ctrl.ref:
            who = (f"由 {ctrl.ref} 支路的电流控制（该电流经自动插入的 0V 探针 "
                   f"{ctrl.sense_ref} 取出，它**不是题目元件**）")
        else:
            who = f"由 {ctrl.sampling} 支路的电流控制"
        return (f"{CONTROL_NOTE.get(comp.kind, '')}；{who}。"
                f"★ 卡上增益写成 {card_val}（IR 增益是 {ir_val}）："
                "IR 的电源电流 i>0 表示供出功率（元件内部 −→+），"
                "而 SPICE 的 H/F 卡取的是 I(先导源)（正号 = 由 + 端流入），两者相反。"
                "读回来时会再反一次，所以往返无损。")
    return CONTROL_NOTE.get(comp.kind, "")


def to_spice(circuit: Circuit, title: str | None = None, *,
             expand_sense: bool = True) -> str:
    """生成 SPICE 网表。参考节点写成 0 —— ngspice 内部地节点只能是 0。

    ``expand_sense``（默认开）只影响电流控制型受控源（``H``/``F``）：
    它们的控制量是"某条支路的电流"，而 SPICE 只能引用**电压源**的电流，
    所以要先在采样支路里串一个 0V 探针源。这一步走的是
    :func:`app.ir.probes.ensure_sense_sources` —— **与三条求解路径同一个函数**，
    于是"网表里的探针"与"求解时插的探针"必然是同一个。
    自动插入的探针会在网表里单独写注释标明它不是题目元件。

    ★ 因此往返契约是：``from_spice(to_spice(c))`` ≡ **展开后**的电路
    （``ensure_sense_sources(c)[0]``），而不是 ``c`` 本身 ——
    只有当``c`` 里没有需要插探针的 ``H``/``F`` 时两者才相同。
    这是有意的：界面上显示、落盘、喂给 ngspice 的都必须是**同一张**电路，
    否则用户看到的网表和真正被求解的网表就分家了。
    想拿"原样"的文本用 ``expand_sense=False``（此时控制端不是电压源会直接报错）。

    ★ 展开带来一个问题：``H``/``F`` 卡上引用的是**探针**（``Vsense_R1``），
    而题目里说的是 ``R1``。光看卡片这两个名字分不开，往返一次就会让
    ``ctrl.ref`` 从 ``R1`` 变成 ``Vsense_R1``（电学等价，但**参数指向错了**）。
    所以每张电流控制型受控源旁边会多写一行机器可读的注释
    ``* ca-ctrl H1 mode=I ref=R1 sense=Vsense_R1``，
    ``from_spice`` 读它并把名字还原 —— 见 :data:`CTRL_MARK`。
    """
    work = circuit
    sense: dict[str, Any] = {"applied": False, "inserted": [], "notes": []}
    if expand_sense:
        work, sense = ensure_sense_sources(circuit)
    inserted = {x["ref"]: x for x in sense.get("inserted", [])}

    lines: list[str] = [
        f"* {title or circuit.name}",
        f"* 由 circuit_agent IR 生成；参考节点 = {circuit.ref_node}",
    ]
    if inserted:
        lines += [
            "*",
            "* ★ 下面标了「测量探针」的电压源是**自动插入**的，不是题目元件：",
            "*   SPICE 的 H/F 卡只能引用电压源的电流，为取到某条支路的电流",
            "*   而在这条支路里串了一个 0V 源。电学行为不变（0V 源对回路 KVL",
            "*   的贡献恒为 0），但它们让节点数与支路数各多。",
        ]

    for c in work.components:
        a = "0" if c.nodes[0] == work.ref_node else c.nodes[0]
        b = "0" if c.nodes[1] == work.ref_node else c.nodes[1]

        if c.ref in inserted:
            lines.append(f"* ↓ 测量探针（自动插入）：{inserted[c.ref]['why']}")
        elif c.kind in CONTROLLED_KINDS:
            # ★ 语义标记必须紧贴它描述的那张卡，且在人读注释**之前** ——
            #   将来若要支持"从注释里认回被采样支路"，靠的就是它跟卡片的邻接关系。
            mark = _ctrl_mark(c)
            if mark:
                lines.append(mark)
            note = controlled_note(c)
            if note:
                lines.append(f"* ↓ {note}")

        if c.kind in CONTROLLED_KINDS:
            lines.append(controlled_card(c, a, b, work))
        else:
            lines.append(_two_terminal_card(c, a, b))

    lines.append(".op")
    lines.append(".end")
    return "\n".join(lines) + "\n"


#: 匹配一行元件卡：REF A B <数值字段...>
#: 数值字段整体捕获（不再只抓一个 token）—— 因为 ``DC 12`` / ``DC=12`` 是**两个**
#: token 的写法，而 ``V1 1 0 DC 12`` 恰恰是本文件 to_spice() 自己的输出格式。
#: 只抓单个 token 的旧写法会让「自家写出的网表自家读不回」，实测就是这样。
_CARD_RE = re.compile(
    r"^\s*(?P<ref>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?P<a>[^\s]+)\s+(?P<b>[^\s]+)\s+"
    r"(?P<rest>.+?)\s*$"
)

#: 非直流激励关键字。出现在数值位置时，说明这张卡描述的根本不是直流工作点。
_EXCITATION_KEYWORDS: dict[str, str] = {
    "AC": "交流幅值/相位（给 .ac 用）",
    "SIN": "正弦瞬态源", "SINE": "正弦瞬态源",
    "PULSE": "脉冲瞬态源",
    "EXP": "指数瞬态源",
    "PWL": "分段线性瞬态源",
    "SFFM": "单频调频瞬态源",
    "AM": "调幅瞬态源",
    "TRNOISE": "瞬态噪声源",
    "TRRANDOM": "瞬态随机源",
    "NOISE": "噪声源",
}

_DC_EQ_RE = re.compile(r"^DC\s*=\s*", re.IGNORECASE)

#: 尾部这些字段是元件参数（不是数值），被忽略时要留痕
_PARAM_HINT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*=")


def _raise_excitation(ref: str, kw: str, raw: str) -> None:
    """非直流激励：给**精准**的错，而不是笼统的「不是字面数字」。

    笼统报错会让人以为是自己格式写错了，去反复折腾格式；
    实际问题是「这张图不是直流工作点题」，改格式永远改不对。
    """
    what = _EXCITATION_KEYWORDS[kw]
    if kw == "AC":
        why = ("而且它没有直流分量可读：按 SPICE 约定 .op 下该源等于 0V/0A，"
               "解出来会是一张全零的图 —— 看着像“算通了”，其实是无意义的答案。")
    else:
        why = ("瞬态源的直流分量取决于激励波形本身（如 SIN 的偏置量），"
               "本版本不解析波形表达式，猜一个数进来就等于篡改题目。")
    raise CircuitError(
        f"{ref} 的数值字段 {raw!r} 是**非直流激励**（{what}）。本工具只解直流工作点"
        f"（.op），不做 .ac / .tran 分析。{why}"
        "请改写为直流值（例如 `DC 12`）；确实需要交流或瞬态分析，请直接用 ngspice。"
    )


def _value_field(ref: str, rest: str) -> tuple[str, list[str]]:
    """从元件卡尾部字段里取出**直流**数值文本。

    返回 ``(数值文本, 警告列表)``。遇到纯交流/瞬态激励直接抛 —— 不猜。
    """
    tokens = rest.split()
    if not tokens:
        raise CircuitError(f"{ref} 缺数值")
    head = tokens[0]
    up = head.upper()

    # ---- 整张卡就是非直流激励：拒绝，并说清为什么
    kw = up.split("(")[0].rstrip("=")
    if kw in _EXCITATION_KEYWORDS:
        _raise_excitation(ref, kw, rest)

    warns: list[str] = []

    # ---- DC 前缀三态：``DC 12`` / ``DC=12`` / ``DC = 12``
    if up.startswith("DC"):
        m = _DC_EQ_RE.match(head)
        if m:
            val, tail = head[m.end():], tokens[1:]
        else:
            val, tail = "", tokens[1:]
        # 手写网表里 ``DC = 12``（等号单独成一个 token）也会出现，顺手容忍掉：
        # 不认它的话会一路走到 float('=') 才炸，报错完全指不到真正的问题。
        if not val and tail and tail[0] == "=":
            val, tail = "", tail[1:]
        if not val:
            if not tail:
                raise CircuitError(f"{ref} 写了 DC 但后面没有值（{rest!r}），无法确定直流工作点。")
            val, tail = tail[0], tail[1:]
    else:
        val, tail = head, tokens[1:]

    # ---- 直流值后面还挂着别的字段：取直流，但必须留痕
    for idx, t in enumerate(tail):
        tkw = t.upper().split("(")[0].rstrip("=")
        if tkw in _EXCITATION_KEYWORDS:
            # 命中即收工：后面的 token 是这条激励**自己的参数**（AC 1 的 '1'、
            # SIN(0 12 50) 的 '12'/'50'），逐条报「未使用附加项」纯属刷屏噪音。
            warns.append(
                f"{ref} 的数值字段 {rest!r} 里同时带了非直流激励 —— "
                f"已只取直流分量 {val!r}，{' '.join(tail[idx:])!r} 这部分被忽略。"
            )
            return val, warns
        warns.append(
            f"{ref} 的数值字段 {rest!r} 里有未使用的附加项 {t!r}，已忽略。"
            + ("（这是元件参数写法，与数值无关）" if _PARAM_HINT_RE.match(t) else "")
        )
    return val, warns


def _map_node(x: str) -> str:
    """SPICE 的地节点写法 → IR 的参考节点名（IR 里一律叫 ``0``）。"""
    return "0" if x in ("0", "gnd", "GND") else x


def _controlled_from_card(ref: str, kind: str, a: str, b: str,
                          rest: str) -> tuple[Component, list[str]]:
    """读一张 ``E``/``G``/``H``/``F`` 卡。

    ``E``/``G`` 卡形状：``Exx a b cx cy 增益``（控制量 = 两个节点间的电压）
    ``H``/``F`` 卡形状：``Hxx a b 电压源位号 增益``（控制量 = 那条电压源的电流）

    ★ ``H``/``F`` 的增益要**再取一次反**：写网表时 IR→SPICE 反了一次
    （见 :data:`spice_expr.SPICE_SOURCE_CURRENT_SIGN`），读回来必须反回去。
    只反一边的结果是"存下来是对的、读回来差个号"，而 round-trip 测试
    恰恰会抓到这个 —— 这也是它值得测的原因。
    """
    tokens = rest.split()
    mode = CONTROL_MODE[kind]
    warns: list[str] = []

    if mode == "V":
        if len(tokens) < 3:
            raise CircuitError(
                f"{ref}（{CONTROL_NOTE[kind]}）的卡应当有「控制端节点对 + 增益」"
                f"三列，例如 `{ref} {a} {b} 1 0 3`；实际是 {rest!r}。")
        ctrl = Control(mode="V", nodes=(_map_node(tokens[0]), _map_node(tokens[1])))
        vtxt, extra = tokens[2], tokens[3:]
    else:
        if len(tokens) < 2:
            raise CircuitError(
                f"{ref}（{CONTROL_NOTE[kind]}）的卡应当有「先导电压源位号 + 增益」"
                f"两列，例如 `{ref} {a} {b} V1 3`；实际是 {rest!r}。"
                "注意 SPICE 只允许电流控制型受控源引用**电压源**，"
                "不能直接写一个电阻位号。")
        ctrl = Control(mode="I", ref=tokens[0])
        vtxt, extra = tokens[1], tokens[2:]

    value, pwarns = parse_engineering(vtxt, kind=kind)
    if value is None:
        detail = "；".join(pwarns) if pwarns else "无法识别的数值写法"
        raise CircuitError(
            f"{ref} 的增益 {vtxt!r} 解析不出数字（{detail}）。"
            "本版本不支持参数化写法（如 {GAIN} 或 .param 引用），请先展开成数字。")
    warns.extend(f"{ref}: {w}" for w in pwarns)
    for t in extra:
        warns.append(f"{ref} 的卡上有多余字段 {t!r}，已忽略。")

    if kind in ("H", "F"):
        before = value
        value = -value
        warns.append(
            f"{ref}: H/F 卡的控制电流（先导源电流）符号与 IR 相反，"
            f"已把卡上的增益 {before:g} 记为 IR 增益 {value:g}。"
            "这是两个约定本身的差，不是笔误：IR 的电源电流 i>0 表示供出功率"
            "（元件内部 −→+），SPICE 的 I(先导源) 正号表示由 + 端流入。")

    comp = Component(
        ref=ref, kind=kind, nodes=(a, b), value=value, ctrl=ctrl,
        evidence=Evidence(source="exact", confidence=1.0,
                          detail="来自 SPICE 网表文本（受控源卡）"),
    )
    return comp, warns


def from_spice(text: str, name: str = "from_spice") -> Circuit:
    """解析一个"朴素"SPICE 网表（我们自己的输出 + KiCad 导出）。

    认这些元件卡：``R/L/C/V/I``（二端），``E/G/H/F``（受控源标准形）。
    ``.op`` / ``.end`` / 注释 / 指令行一律跳过。

    **不认行为源 ``B``**（含本工具自己为自定义表达式生成的那种）。
    任意 ngspice 表达式语法 → IR 线性表达式这一步只能靠猜，
    而猜错的表现是"网表读进来了、算出来是另一个电路"。所以直接报错，
    并指出改法（写成标准形受控源卡，或在界面的「参数」页里填表达式）。
    这一条是"不许猜"的直接延伸。

    数值字段支持：
      * 裸数字 ``100``
      * 直流前缀 ``DC 12`` / ``DC=12``（**to_spice 自己的输出格式**，必须读得回来）
      * 工程记法 ``1k`` / ``4k7`` / ``1meg`` / ``10u``（复用 :mod:`app.ingest.values`，
        连 ``M``/``m`` 的单位约定冲突都会跟着升级成警告）
    纯交流/瞬态激励（``AC`` / ``SIN(...)`` / ``PULSE(...)``）**拒绝**并给出精准原因 ——
    本工具只解直流工作点，把交流源当直流量读进来是篡改题目。

    另外认一行我们自己写的语义标记 ``* ca-ctrl <ref> mode=I ref=<支路> sense=<探针>``
    （见 :data:`CTRL_MARK`）：它把"题目里说的被采样支路"从自动插入的 0V 探针下
    还原回来。标记与卡片不符时**以卡片为准**并把原因写进 diagnostics，
    绝不按标记去改电路连接。
    """
    comps: list[Component] = []
    warns: list[str] = []
    marks: dict[str, dict[str, str]] = {}

    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("*"):
            # ---- 语义标记（``* ca-ctrl …``）：只在注释里，不影响电路本身
            mm = _CTRL_MARK_RE.match(line)
            if mm:
                fields: dict[str, str] = {}
                for tok in mm.group("fields").split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        fields[k.strip().lower()] = v.strip()
                marks[mm.group("ref")] = fields
            continue
        if line.startswith("."):
            continue  # 指令行

        m = _CARD_RE.match(line)
        if not m:
            warns.append(f"跳过无法解析的行：{line!r}")
            continue

        ref = m.group("ref")
        up0 = ref[0].upper()
        a, b = _map_node(m.group("a")), _map_node(m.group("b"))

        # ---- 行为源：明确拒绝，并说清改法（不许猜表达式语法）
        if up0 == "B":
            raise CircuitError(
                f"元件卡 {line!r} 是**行为源**（B 源）。本工具不解析任意行为表达式 ——"
                "把 ngspice 的表达式语法猜成 IR 的线性表达式，猜错的表现是"
                "「网表读进来了、算出来是另一个电路」。"
                "若它的控制关系是线性的，请改写成标准形受控源卡"
                "（E/G 由电压控制、H/F 由电流控制）；"
                "若你是在本工具里用参数名写的自定义表达式，"
                "请在界面的「参数」页里改，不要靠网表往返。"
            )

        kind = LETTER_KIND.get(up0)
        if kind is None:
            raise CircuitError(
                f"元件卡 {line!r} 的首字母 {ref[0]!r} 不在支持范围 "
                f"{sorted(LETTER_KIND)}；器件卡（D/Q/M/J/X 等）本版本不支持，"
                "请手工改写为受支持的形式。"
            )

        # ---- 受控源：卡形完全不同，走单独一条路
        if kind in CONTROLLED_KINDS:
            comp, cw = _controlled_from_card(ref, kind, a, b, m.group("rest"))
            warns.extend(cw)
            comps.append(comp)
            continue

        vtxt, vwarns = _value_field(ref, m.group("rest"))
        warns.extend(vwarns)

        value, pwarns = parse_engineering(vtxt, kind=kind)
        if value is None:
            detail = "；".join(pwarns) if pwarns else "无法识别的数值写法"
            raise CircuitError(
                f"{ref} 的数值 {vtxt!r} 解析不出数字（{detail}）。"
                "本版本不支持参数化写法（如 {R1} 或 .param 引用），请先展开成数字。"
            )
        # parse_engineering 的警告**必须带上位号**转发出去，不能丢 ——
        # 其中就包括 1M/1m 那个 SPICE=毫、KiCad=兆 的约定冲突，
        # 而三法互校**拦不住**它（三法共用同一份 IR，会一致地给出同一个错答案）。
        warns.extend(f"{ref}: {w}" for w in pwarns)

        comps.append(Component(
            ref=ref, kind=kind, nodes=(a, b), value=value,
            evidence=Evidence(source="exact", confidence=1.0,
                              detail="来自 SPICE 网表文本"),
        ))

    if not comps:
        raise CircuitError("网表里没有解析出任何元件")

    # ---- 读完之后才能查"H/F 的控制端到底是不是电压源"：
    #   逐行解析时后面的元件还没读到，被控制的那条支路可能出现在下一行。
    #   ★ 必须查。``H1 3 0 R1 2000`` 在 ngspice 里根本不成立（H 卡只认电压源），
    #   而我们下游的 ensure_sense_sources 会**很热心地**给 R1 插一个探针、
    #   把这条错网表悄悄"修好" —— 用户以为读进来了，其实读的是另一张电路。
    by_ref = {c.ref: c for c in comps}
    for c in comps:
        ctrl = c.ctrl
        if ctrl is None or ctrl.mode != "I" or ctrl.expr:
            continue
        tgt = by_ref.get(ctrl.ref)
        if tgt is None:
            raise CircuitError(
                f"{c.ref}（{CONTROL_NOTE.get(c.kind, '')}）的控制端写的是 {ctrl.ref!r}，"
                f"但网表里没有这个元件。现有位号：{sorted(by_ref)}。")
        if not tgt.outputs_voltage:
            raise CircuitError(
                f"{c.ref} 的控制端写的是 {ctrl.ref!r}，但它在网表里是 {tgt.kind}，"
                "不是电压输出元件。SPICE 的 H/F 卡只允许引用**电压源**的电流 ——"
                "这条网表在 ngspice 里同样跑不起来。请把控制端改成电压源位号，"
                "或在被测支路里串一个 0V 电压源（本工具求解与导出网表时会自动做这件事）。")

    # ---- 语义标记：把「题目里说的那条被采样支路」还原回来
    #   ★ 顺序很关键：**先**按卡片查完"H/F 控制端是不是电压源"（那是网表自身的
    #     合法性），**再**用标记还原 ref。反过来的话，标记就会把一个本来不合法的
    #     控制端悄悄换成合法的，把一张跑不起来的网表变成"读进来了"。
    for ref, fields in marks.items():
        comp = by_ref.get(ref)

        def _drop(why: str) -> None:
            warns.append(
                f"网表里有语义标记 {CTRL_MARK} {ref} …（{fields}），"
                f"但{why}，标记已忽略 —— 控制关系以元素卡片为准。")

        if comp is None or comp.ctrl is None:
            _drop("网表里没有这个受控源")
            continue
        if comp.ctrl.mode != "I":
            _drop("它不是电流控制型（标记只用于电流控制）")
            continue
        mode = (fields.get("mode") or "I").upper()
        if mode != "I":
            _drop(f"标记里写的是 mode={mode}")
            continue
        orig, sense = fields.get("ref", ""), fields.get("sense", "")
        if not orig or not sense:
            _drop("标记里缺 ref= 或 sense= 字段")
            continue
        if comp.ctrl.ref != sense:
            _drop(f"卡上引用的是 {comp.ctrl.ref!r}，不是标记里的探针 {sense!r}")
            continue
        if sense not in by_ref:
            _drop(f"探针 {sense!r} 不在网表里")
            continue
        if orig not in by_ref:
            _drop(f"被采样支路 {orig!r} 不在网表里")
            continue
        comp.ctrl.sense_ref = sense          # 实际取电流的那条支路
        comp.ctrl.ref = orig                 # ★ 题目里说的那条支路 —— 名字归位
        warns.append(
            f"{ref}: 按网表语义标记把被采样支路还原为 {orig!r}"
            f"（卡上引用的是自动插入的 0V 探针 {sense!r}，它不是题目元件）。"
            "电学行为完全不变，只是「参数指向」回到题目原意 ——"
            "否则界面上会把被采样支路显示成那个探针。")

    c = Circuit(name=name, components=comps, ref_node="0",
                diagnostics=[{"kind": "spice_parse_warning", "text": w} for w in warns])
    return c



def canonical_netlist_view(circuit: Circuit) -> dict:
    """给 WebUI 用的网表视图：拆成结构化行，便于前端逐行高亮与纠错。

    ``expand_sense`` 的语义与 :func:`to_spice` 一致：电流控制型受控源需要
    0V 探针源，这里也先展开 —— 界面上显示的网表必须**就是**被求解的那一张。
    自动插入的探针行带 ``probe=True`` 与 ``note``，让人一眼看出
    "这一行不是我画的"。
    """
    work, sense = ensure_sense_sources(circuit)
    inserted = {x["ref"]: x for x in sense.get("inserted", [])}
    rows = []
    for c in work.components:
        a = "0" if c.nodes[0] == work.ref_node else c.nodes[0]
        b = "0" if c.nodes[1] == work.ref_node else c.nodes[1]
        if c.kind in CONTROLLED_KINDS:
            card = controlled_card(c, a, b, work)
        else:
            card = _two_terminal_card(c, a, b)
        row: dict[str, Any] = {
            "ref": c.ref, "kind": c.kind, "card": card,
            "plus": a, "minus": b, "value": c.value,
            "source": c.evidence.source, "confidence": c.evidence.confidence,
            "net_name": _netname(c), "note": "",
        }
        if c.kind in CONTROLLED_KINDS and c.ctrl is not None:
            row.update({
                "controlled": True,
                "control": c.control_text(),
                "control_mode": c.ctrl.mode,
                "expr": c.ctrl.expr,
                "gain_symbol": GAIN_SYMBOL.get(c.kind, ""),
                "output": "V" if c.outputs_voltage else "A",
            })
            row["note"] = controlled_note(c)
        if c.ref in inserted:
            row["probe"] = True
            row["note"] = "测量探针（自动插入，不是题目元件）：" + inserted[c.ref]["why"]
        rows.append(row)

    return {
        "title": circuit.name, "ref_node": circuit.ref_node, "rows": rows,
        "sense_applied": bool(sense.get("applied")),
        "sense_notes": list(sense.get("notes", [])),
        "probe_row_count": len(inserted),
    }
