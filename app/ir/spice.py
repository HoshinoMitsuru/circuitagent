"""IR <-> SPICE 网表序列化。

网表是本项目的第二公共语言：ngspice 直接吃它，KiCad 能导它，
人也能一眼看懂。**每条支路的参考方向按 IR 原样落盘**，不做任何自动换向。
"""

from __future__ import annotations

import re

from .model import Circuit, Component, CircuitError, Evidence
# 复用工程记法解析器（1k / 4k7 / 1meg / 10u，含 M/m 约定冲突警告）。
# 直接引模块而非 ingest 包，避免将来 ingest 包 aggregate 导入时绕回本文件形成环。
from ..ingest.values import parse_engineering

# SPICE 首字母 -> 我们的 kind。注意 V 在 KiCad 里可能是 VDD 之类，交给上游判定。
LETTER_KIND = {"R": "R", "L": "L", "C": "C", "V": "V", "I": "I"}


def to_spice(circuit: Circuit, title: str | None = None) -> str:
    """生成 SPICE 网表。参考节点写成 0 —— ngspice 内部地节点只能是 0。"""
    lines: list[str] = []
    lines.append(f"* {title or circuit.name}")
    lines.append(f"* 由 circuit_agent IR 生成；参考节点 = {circuit.ref_node}")

    for c in circuit.components:
        if c.value is None:
            raise CircuitError(f"{c.ref} 缺数值，拒绝生成网表（宁可不给，也不给错）")
        a = "0" if c.nodes[0] == circuit.ref_node else c.nodes[0]
        b = "0" if c.nodes[1] == circuit.ref_node else c.nodes[1]
        # 电压源显式写 DC，避免 ngspice 在 .op 下自作主张
        if c.kind == "V":
            lines.append(f"{c.ref} {a} {b} DC {c.value_str()}")
        else:
            lines.append(f"{c.ref} {a} {b} {c.value_str()}")

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


def from_spice(text: str, name: str = "from_spice") -> Circuit:
    """解析一个"朴素"SPICE 网表（我们自己的 three-line-dialect + KiCad 输出）。

    只认最基础的 R/L/C/V/I 两终端元件卡；``.op`` / ``.end`` / 注释 / 指令行一律跳过。
    复合卡（控制源 E/F/G/H、器件 D/Q/M）暂不支持 —— 遇到直接报错，
    而不是猜着解析（禁令 1 的延伸：不许猜）。

    数值字段支持：
      * 裸数字 ``100``
      * 直流前缀 ``DC 12`` / ``DC=12``（**to_spice 自己的输出格式**，必须读得回来）
      * 工程记法 ``1k`` / ``4k7`` / ``1meg`` / ``10u``（复用 :mod:`app.ingest.values`，
        连 ``M``/``m`` 的单位约定冲突都会跟着升级成警告）
    纯交流/瞬态激励（``AC`` / ``SIN(...)`` / ``PULSE(...)``）**拒绝**并给出精准原因 ——
    本工具只解直流工作点，把交流源当直流量读进来是篡改题目。
    """
    comps: list[Component] = []
    warns: list[str] = []

    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("."):
            continue  # 指令行

        m = _CARD_RE.match(line)
        if not m:
            warns.append(f"跳过无法解析的行：{line!r}")
            continue

        ref = m.group("ref")
        kind = LETTER_KIND.get(ref[0].upper())
        if kind is None:
            raise CircuitError(
                f"元件卡 {line!r} 的首字母 {ref[0]!r} 不在支持范围 "
                f"{sorted(LETTER_KIND)}；复合器件卡（E/F/G/H/D/Q/M/J/X）本版本不支持，"
                "请手工改写为受支持的形式。"
            )

        a, b = m.group("a"), m.group("b")
        # SPICE 的地节点 0 映射回 IR 的参考节点
        a = "0" if a in ("0", "gnd", "GND") else a
        b = "0" if b in ("0", "gnd", "GND") else b

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

    c = Circuit(name=name, components=comps, ref_node="0",
                diagnostics=[{"kind": "spice_parse_warning", "text": w} for w in warns])
    return c



def canonical_netlist_view(circuit: Circuit) -> dict:
    """给 WebUI 用的网表视图：拆成结构化行，便于前端逐行高亮与纠错。"""
    rows = []
    for c in circuit.components:
        a = "0" if c.nodes[0] == circuit.ref_node else c.nodes[0]
        b = "0" if c.nodes[1] == circuit.ref_node else c.nodes[1]
        if c.kind == "V":
            card = f"{c.ref} {a} {b} DC {c.value_str()}"
        else:
            card = f"{c.ref} {a} {b} {c.value_str()}"
        rows.append({
            "ref": c.ref, "kind": c.kind, "card": card,
            "plus": a, "minus": b, "value": c.value,
            "source": c.evidence.source, "confidence": c.evidence.confidence,
        })
    return {"title": circuit.name, "ref_node": circuit.ref_node, "rows": rows}
