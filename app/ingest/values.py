"""工程记法数值解析：``1k`` / ``4.7uF`` / ``100n`` / ``2.2MΩ`` / ``10V``。

**单独成模块的理由**：位号旁边那串数字是"读图最后一步"，也是错得最隐蔽的一步。
``1M`` 到底是 1 毫欧还是 1 兆欧，SPICE 与 KiCad 的约定不一致 ——
本模块的立场是：**能确定就转，不能确定就标警告，绝不猜**。
猜错一个 1M，后面三法会一致地给出同一个错答案（因为三法用的是同一个 IR），
整个"三法互校"体系都拦不住。这正是需要人工确认闸门的地方。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

#: SI 前缀。注意顺序：多字母的 meg 必须排在 m 前面，否则 "1meg" 会被当成 "1m"+"eg"
_PREFIXES: list[tuple[str, float]] = [
    ("meg", 1e6),
    ("tera", 1e12),
    ("giga", 1e9),
    ("kilo", 1e3),
    ("milli", 1e-3),
    ("micro", 1e-6),
    ("nano", 1e-9),
    ("pico", 1e-12),
    ("T", 1e12),
    ("G", 1e9),
    ("K", 1e3),
    ("k", 1e3),
    ("M", 1e6),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("μ", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
]

#: 单位后缀（转完数值就丢掉；不做单位换算，那是下一层的事）
_UNITS = ("ohm", "Ω", "ω", "F", "H", "V", "A", "W", "S", "Hz")

_NUM_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$")

#: 英式/SPICE 写法：4k7 = 4.7k，1R5 = 1.5Ω，2M2 = 2.2M
_BRITISH_RE = re.compile(r"^(\d+)([a-zA-Zµμ])(\d+)$")

_BRITISH_MULT = {"r": 1.0, "R": 1.0, "k": 1e3, "K": 1e3, "M": 1e6,
                 "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12}


def _mul(mantissa: float, mult: float) -> float:
    """用 Decimal 做乘法再转回 float。

    ★ 为什么不用 float 直接乘：``10 * 1e-6`` 在二进制里是
    9.999999999999999e-06。这个毛刺会一路传进 MNA ——
    精确有理数把 float 按**二进制真值**转成分数，
    于是 1µF 变成分母为 2^53 的怪分数，
    技能文档里那条很有用的自检（"精确解的分母统一，说明读图建模都对了"）就失效了。
    走一遍 Decimal 能让 1e-05 这种整齐的数保持整齐。
    """
    try:
        return float(Decimal(repr(mantissa)) * Decimal(repr(mult)))
    except (InvalidOperation, ValueError):
        return mantissa * mult


def parse_engineering(text: Any, *, kind: str = "") -> tuple[float | None, list[str]]:
    """把工程记法解析成浮点数。

    返回 ``(数值或 None, 警告列表)``。解析不出来时**不抛异常**，
    而是返回 None + 警告 —— 让上层把它当成"需人工填写"，而不是让整张图解析失败。
    """
    warnings: list[str] = []
    if text is None:
        return None, ["数值为空"]
    if isinstance(text, (int, float)):
        return float(text), []

    s = str(text).strip()
    if not s:
        return None, ["数值为空"]
    if s in ("-", "?", "DNP", "dnp"):
        return None, [f"数值 {s!r} 表示未装配/未知，需人工确认"]

    # 英式写法先试（4k7 / 1R5 / 2M2）—— 这类串普通正则解析不出正确数值，
    # 会退化成"1"或"4"这种错得离谱的结果，且不会有任何报错
    w = []
    mb = _BRITISH_RE.match(s)
    if mb and mb.group(2) in _BRITISH_MULT:
        head, pfx, tail = mb.groups()
        mult = _BRITISH_MULT[pfx]
        v = _mul(float(head) + float(f"0.{tail}"), mult)
        if pfx in ("M", "m"):
            w.append(f"数值 {s!r} 用了 {pfx!r} 前缀（SPICE 里 M=毫、KiCad 里 M=兆），"
                     "已按 "
                     + ("兆(1e6)" if pfx == "M" else "毫(1e-3)")
                     + " 处理，请人工核对量级。")
        return v, w

    m = _NUM_RE.match(s)
    if not m:
        return None, [f"无法解析数值 {s!r}"]
    mantissa = float(m.group(1))
    rest = m.group(2).strip()

    if not rest:
        return mantissa, []

    # 先剥单位后缀（按长度降序，因为 "ohm"/"Hz" 比单字符单位长）
    core = rest
    for u in sorted(_UNITS, key=len, reverse=True):
        if core.endswith(u):
            core = core[: -len(u)]
            break

    if not core:
        return mantissa, []

    # 前缀匹配
    for pfx, mult in _PREFIXES:
        hit = (core == pfx) or core.startswith(pfx)
        if not hit:
            continue
        if core != pfx:
            tail = core[len(pfx):]
            w.append(
                f"数值 {s!r} 里的后缀 {tail!r} 不认识（已按 {pfx} 前缀解析），请人工核对"
            )
        # ★ M/m 是经典陷阱：SPICE 里 M = 毫（mega 要写 MEG），
        #   KiCad 里 M = 兆。同一个字符串差 10⁹ 倍，而且三法互校**拦不住**它
        #   （因为三法共用同一份 IR）。所以这里必须显式警告，逼人工看一眼。
        if pfx in ("M", "m"):
            w.append(
                f"数值 {s!r} 用了 {pfx!r} 前缀。注意约定冲突：SPICE 里 M = 毫、KiCad 里 M = 兆。"
                f"本解析按「{pfx!r} = {'兆(1e6)' if pfx == 'M' else '毫(1e-3)'}」处理，请人工核对量级。"
            )
        return _mul(mantissa, mult), w

    w.append(
        f"数值 {s!r} 的后缀 {core!r} 无法识别，按原数 {mantissa} 处理，请人工核对"
    )
    return mantissa, w


def parse_resistor(text: Any) -> tuple[float | None, list[str]]:
    """电阻专用。

    现在 ``4k7`` / ``1R5`` / ``2M2`` 这些英式写法已经由 parse_engineering 统一处理，
    这个函数保留下来是因为"电阻"在教材里出现的写法最杂（``1k``、``4k7``、``1R5``、
    ``2.2M``、``220``），单独留一个入口便于以后加规则时不影响别的元件。
    """
    return parse_engineering(text, kind="R")


def format_value(value: float | None, kind: str = "") -> str:
    """反方向：数值转成好看的工程写法，供报告显示。"""
    if value is None:
        return "—"
    unit = {"R": "Ω", "V": "V", "I": "A", "C": "F", "L": "H"}.get(kind, "")
    if value == 0:
        return f"0{unit}"
    a = abs(value)
    for mult, pfx in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
                      (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if a >= mult * 0.999:
            v = value / mult
            txt = f"{v:.6g}"
            return f"{txt}{pfx}{unit}"
    return f"{value:.6g}{unit}"
