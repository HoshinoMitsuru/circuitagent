"""参数表 —— IR 之上的一层"人能读、能写、能引用"的命名体系。

## 为什么需要这一层

IR 里每个量都只有**机器身份**：一条支路叫 ``Component``（由 ``ref`` 认），
一个结点叫 ``"3"``（字符串）。这足够求解，却不够**交流**：

* 学生想问的是"``u_3`` 是多少"，不是"节点 3 的电位是多少"；
* 老师给的表达式是 ``u_o = 2·u_1 − 3·u_2``，不是
  ``V(4) = 2·V(1) − 3·V(2)``；
* 同一个量在不同教材里有不同习惯名（``i_L`` / ``i_1`` / ``i(t)``）。

所以这一层专门解决"**名字**"：

::

    IR（ref / 结点名，机器身份）
      ↑↓  参数表：名字 ↔ 绑定
    名字（u_3 / i_R1 / R1，人能读能写）
      ↑↓  线性表达式：用名字书写
    约束（受控源的受控关系）

**参数名只是"显示与引用层"，不是求解层。** 三条求解路径、网表、跨工具对账
一律继续用 ``ref`` / 结点名 —— 这样对账的"两套独立实现喂同一份网表"这条
最强判据完全不受影响（见 MEMORY.md 的符号约定映射表）。

## 三条设计决定（都是踩过坑之后定的）

1. **绑定是键，名字是值。** ``ParamTable.sync()`` 按**绑定键**做匹配，
   所以电路增删元件时，用户起过的名字跟着它的绑定走，不会顺次挪位。
   元件被删掉时那个参数**标成孤儿留在表里**，不静默消失 —— 否则改名记录
   会凭空蒸发，而用户改过的表达式还指着它。

2. **默认名从 ref / 结点名派生**（``i_R1`` / ``u_R1`` / ``R1`` / ``u_3``），
   不用序号。序号会在"删掉中间一个元件"之后整体漂移，而派生名只跟它自己的
   绑定有关 —— 这正是"避免参数指向错误"的落地方式。

3. **改名要连表达式一起改。** 表达式里写的是名字，名字一改而表达式没改，
   就得到一个"指向不存在符号"的表达式。那是**静默错误**：不报错、
   只是算出来不对。所以 ``rename()`` 会把所有引用旧名的表达式原地改写。

## 单位

用户明确要求"UI 需标明各参数所用单位为 SI"。单位由 ``quantity`` + 元件类型
唯一决定，见 ``QUANTITY_UNIT`` / ``VALUE_UNIT``。受控源的"数值"其实是**增益**，
单位与 R/L/C 完全不同（V/V、Ω、S、A/A），所以单列一张表 ——
把增益当成电阻显示，是这种题目里最容易让人发懵的地方。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .errors import CircuitError

if TYPE_CHECKING:                                # 打断 model ↔ params 的循环
    from .model import Circuit, Component

# ---------------------------------------------------------------- 单位（SI）

#: 量的种类 -> SI 单位
QUANTITY_UNIT: dict[str, str] = {
    "u": "V",
    "i": "A",
    "value": "",                                 # 由元件类型决定，见 VALUE_UNIT
}

#: 元件值 / 受控源增益的单位。
#:
#: ★ 受控源那四行的单位是**按定义**推出来的，不是随便标的：
#:   * ``E`` VCVS：输出 V、控制 V ⇒ V/V（无量纲），教材记作 μ；
#:   * ``G`` VCCS：输出 A、控制 V ⇒ A/V = S（西门子），教材记作 gm；
#:   * ``H`` CCVS：输出 V、控制 A ⇒ V/A = Ω（欧姆），教材记作 rm；
#:   * ``F`` CCCS：输出 A、控制 A ⇒ A/A（无量纲），教材记作 α。
#: 这也是判断"受控源画对没有"的一条硬标准：写出来的增益单位必须与卡片自洽。
VALUE_UNIT: dict[str, str] = {
    "R": "Ω", "C": "F", "L": "H", "V": "V", "I": "A",
    "E": "V/V", "G": "S", "H": "Ω", "F": "A/A",
}

#: 无量纲增益的中文说明（UI 上比裸 "V/V" 好读）
#: 每种受控源的增益在该 kind 下叫什么（教材符号）。
#: —— 与 ``GAIN_NOTE`` 一起住在**这一层**（IR 的最底层，不依赖 model）：
#:    报告、参数面板、报错文案都要用它来解释"这个数是什么"，
#:    而它们上下都有。放在 model 里就会逼着下层反向 import。
GAIN_SYMBOL: dict[str, str] = {"E": "μ", "G": "gm", "H": "rm", "F": "α"}

#: 每种受控源的**控制量**是什么。``"V"`` = 控制量是电压（压控），
#: ``"I"`` = 控制量是支路电流（流控）。
#: ★ 这是**元件类型的固有属性**，与"输出成什么 SPICE 卡"无关 ——
#:   所以它住在 IR 层而不是网表层。网表层过去自己定义了一份，
#:   于是界面（`/api/symbols`）想说"这是压控还是流控"就得反向去 import 网表模块。
#:   和 ``GAIN_SYMBOL`` 一样：只有一处定义，别处都是转发。
CONTROL_MODE: dict[str, str] = {"E": "V", "G": "V", "H": "I", "F": "I"}

#: 每种受控源的**物理含义**一句话。报错、界面提示、报告共用同一份文字。
CONTROL_NOTE: dict[str, str] = {
    "E": "电压控制电压源（输出 V，控制量是某两支点间的电压，增益无量纲）",
    "G": "电压控制电流源（输出 A，控制量是某两支点间的电压，增益单位 S）",
    "H": "电流控制电压源（输出 V，控制量是某条支路的电流，增益单位 Ω）",
    "F": "电流控制电流源（输出 A，控制量是某条支路的电流，增益无量纲）",
}

GAIN_NOTE: dict[str, str] = {
    "E": "电压放大倍数（无量纲）",
    "G": "转移电导 gm（西门子）",
    "H": "转移电阻 rm（欧姆）",
    "F": "电流放大倍数（无量纲）",
}

#: 量 -> 中文名（UI 分组与表头用）
QUANTITY_LABEL: dict[str, str] = {
    "u": "电压",
    "i": "电流",
    "value": "元件值 / 增益",
}

#: 表达式展开的最大深度（防"受控源引用受控源"绕成环）。
#: 放在 IR 层是因为**两条路径都要用同一个上限**：求解层展开线性组合、
#: 网表层渲染 SPICE 文本。分成两个常量，一处放宽一处不放，就会出现
#: "三法说成环、网表却老老实实生成了"这种自相矛盾的输出。
MAX_EXPR_DEPTH = 8

#: 工程记法后缀 -> 十进制指数。
#: ★ ``m`` 按 **SPICE 约定**取毫（1e-3）；兆必须写 ``meg``。
#:   这与 ``ingest.values`` 的处理一致：那里对单个 ``M`` 会升级成警告
#:   （SPICE=毫、KiCad=兆），表达式里同样不接受裸 ``M`` 当兆。
_SUFFIXES: dict[str, int] = {
    "t": 12, "g": 9, "meg": 6, "k": 3, "m": -3,
    "u": -6, "n": -9, "p": -12, "f": -15,
}

#: 合法标识符（参数名要能进表达式，所以必须过这一关）
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: 记号：数字（含工程记法） / 标识符 / 运算符
#:
#: ★ 数字那一支必须一次吃下 ``4k7`` 这种**英式写法**（= 4.7k）。
#:   否则它会被切成 ``4k`` + ``7`` 两个记号，报出一句"多了内容"，
#:   而真正的原因是分词没认这个写法。``ingest.values`` 一直支持它，
#:   表达式里没理由不支持。
#:
#: ★ 末尾的 ``(?![A-Za-z0-9_])`` 很关键：没有它，``2u_1``（本意大概是
#:   "2 乘 u_1"、漏了 ``*``）会被吃成 ``2u`` + ``_1``，报错指向一个
#:   根本不存在的符号 ``_1``；有它则整体认不出来，报错落在
#:   "这里有个不认识的写法"上，指得准。
_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?:(?:meg|[tgkmunpf])\d*)?(?![A-Za-z0-9_]))"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>[+\-*/()])"
    r")"
)

#: 标识符**整词**匹配（前后都不能是标识符字符）。
#: 给"按记号改写表达式"用 —— 见 :func:`rewrite_symbol`。
_IDENT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])")

#: 数字字面量：尾数 + 可选后缀 + 可选的"尾数小数位"（英式 4k7）
_LITERAL_RE = re.compile(
    r"^((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(meg|[tgkmunpf])?(\d*)$")


# ---------------------------------------------------------------- 名字

def sanitize(name: str) -> str:
    """把任意结点名 / 位号压成一个**合法标识符**片段。

    结点名可能是 ``out+`` / ``3.3`` / ``n-1`` 这类不能直接进表达式的东西。
    规则：非字母数字下划线一律换成 ``_``，连续下划线压成一个，首尾去下划线。
    返回值可能为空串（例如结点名是 ``++``），调用方负责兜底。
    """
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    s = re.sub(r"_{2,}", "_", s)
    return s


def is_valid_symbol(sym: str) -> bool:
    """参数名是不是一个能写进表达式的合法标识符。"""
    return bool(_IDENT_RE.match(str(sym)))


# ---------------------------------------------------------------- 绑定

#: 绑定键 —— **参数表里唯一的匹配依据**。
#:
#: 形如 ``("branch_i", "R1")`` / ``("node_u", "3")`` / ``("value", "C1")``。
#: 用元组而不是字典：元组可哈希，能当索引键；字典做不到。
Binder = tuple[str, str]

BINDER_KINDS = ("node_u", "value", "branch_i", "branch_u")

#: 绑定种类 -> 该绑定量属于哪种 quantity
BINDER_QUANTITY: dict[str, str] = {
    "node_u": "u",
    "branch_u": "u",
    "branch_i": "i",
    "value": "value",
}

#: 绑定种类 -> 中文说明（表里"这个参数是什么"那一列）
BINDER_LABEL: dict[str, str] = {
    "node_u": "节点电压",
    "branch_u": "支路电压",
    "branch_i": "支路电流",
    "value": "元件值",
}


def auto_symbol(binder: Binder, used: Iterable[str]) -> str:
    """给一个绑定生成默认名字，并避开已占用的名字。

    派生规则（刻意"从身份派生"而不是"按序号排"）：

    ==================  ================  ==========
    绑定                 默认名            例
    ==================  ================  ==========
    ``("value", ref)``  ``ref``           ``R1``、``C1``
    ``("branch_i",ref)`` ``i_<ref>``      ``i_R1``
    ``("branch_u",ref)`` ``u_<ref>``      ``u_R1``
    ``("node_u", node)`` ``u_<node>``     ``u_3``
    ==================  ================  ==========

    元件值直接用位号（``R1`` 同时表示这个电阻和它的阻值），是教科书惯例 ——
    题目里写的就是"``R1 = 10Ω``"。
    """
    kind, key = binder
    frag = sanitize(key)
    if kind == "value":
        base = frag or "val"
    elif kind == "branch_i":
        base = f"i_{frag}" or "i_x"
    else:                                        # branch_u / node_u
        base = f"u_{frag}" or "u_x"

    taken = set(used)
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


# ---------------------------------------------------------------- 参数

@dataclass
class Param:
    """一个名字，以及它绑到电路的哪个量上。"""

    symbol: str
    binder: Binder
    #: 是否仍是**自动生成**的名字（用户没动过）。UI 上要能一眼分清
    #: "这个名字是我自己起的"和"系统给它起的"。
    auto: bool = True
    #: 绑定的对象在当前电路里已经不存在了（元件被删 / 结点改名）。
    #: ★ 不删掉它：用户改过的表达式可能还引用着这个名字，
    #:   悄悄删掉就是让表达式指向空气。留着 + 显式标出来。
    orphan: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).strip()
        self.binder = (str(self.binder[0]), str(self.binder[1]))
        if not is_valid_symbol(self.symbol):
            raise CircuitError(
                f"参数名 {self.symbol!r} 不是合法标识符（只能用字母、数字、下划线，"
                "且不能以数字开头）—— 它要能写进表达式里"
            )
        if self.binder[0] not in BINDER_QUANTITY:
            raise CircuitError(
                f"未知的绑定种类 {self.binder[0]!r}，可选 {list(BINDER_QUANTITY)}")

    @property
    def quantity(self) -> str:
        """``"u"`` / ``"i"`` / ``"value"`` —— 决定单位与分组。"""
        return BINDER_QUANTITY[self.binder[0]]

    @property
    def key(self) -> Binder:
        return self.binder       # 可读别名

    def binder_text(self) -> str:
        """绑定的可读描述，例如 ``i(R1)``、``u(节点 3)``。"""
        kind, ref = self.binder
        if kind == "node_u":
            return f"V({ref})"
        if kind == "value":
            return f"值({ref})"
        if kind == "branch_i":
            return f"i({ref})"
        return f"u({ref})"

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "binder": list(self.binder),
                "auto": self.auto, "orphan": self.orphan, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Param":
        b = d.get("binder") or ()
        if isinstance(b, str):
            b = (b, d.get("target", ""))
        return cls(symbol=d.get("symbol", ""), binder=(b[0], b[1]),
                   auto=bool(d.get("auto", True)),
                   orphan=bool(d.get("orphan", False)),
                   note=d.get("note", ""))


# ---------------------------------------------------------------- 表达式

@dataclass
class LinearExpr:
    """**线性**表达式：系数表 + 常数项。

    只支持一次式，是有意的：受控源在教科书里给的就是线性受控关系
    （μ·u、rm·i、gm·u、α·i），而非线性（``u²``、``u_1·u_2``）会让
    节点电压法/支路电流法这两条精确路径直接失效 —— 那时"两法逐项严格相等"
    这条最强判据就没了。所以遇到就**明确报错**，而不是勉强算出一个只对
    ngspice 成立的答案却让三条路径看起来还能对账。
    """

    coeffs: dict[str, Fraction] = field(default_factory=dict)
    const: Fraction = field(default_factory=Fraction)

    def __post_init__(self) -> None:
        self.coeffs = {k: Fraction(v) for k, v in self.coeffs.items() if v != 0}
        self.const = Fraction(self.const)

    # ---- 基本性质

    @property
    def symbols(self) -> list[str]:
        return sorted(self.coeffs)

    @property
    def is_constant(self) -> bool:
        return not self.coeffs

    @property
    def is_zero(self) -> bool:
        return not self.coeffs and self.const == 0

    def is_standard_single(self) -> tuple[str, Fraction] | None:
        """是不是"单个符号 × 系数"的标准形（常数项必须为 0）。

        是的话就能直接落成 SPICE 的 E/G/H/F 标准卡；不是才需要 B 源兜底。
        返回 ``(符号, 系数)``。
        """
        if self.const != 0 or len(self.coeffs) != 1:
            return None
        sym, k = next(iter(self.coeffs.items()))
        return sym, k

    # ---- 求值

    def eval(self, values: Mapping[str, Any], *, strict: bool = True) -> Any:
        """代入求值。``strict`` 时缺任一符号即报错（宁可报错也不当 0 用）。"""
        acc: Any = self.const
        if isinstance(acc, Fraction):
            acc = Fraction(acc)
        for sym, k in self.coeffs.items():
            if sym not in values or values[sym] is None:
                if strict:
                    raise CircuitError(f"表达式求值缺参数 {sym!r} 的值")
                continue
            v = values[sym]
            acc = acc + (k * v if isinstance(v, (int, Fraction)) else float(k) * v)
        return acc

    # ---- 展示

    def describe(self, fmt: str = "text") -> str:
        """人读形式。``text`` 用 ``·`` 与上标减号，``ascii`` 给纯 ASCII 场合。

        ★ **按书写顺序输出**，不按字母序重排。用户敲进去 ``2*u_2 - u_1``，
        回显就该是 ``2·u_2 − u_1`` —— 顺序被重排会让人怀疑"是不是又被
        替成了另一个式子"。这一点在表达式可编辑的功能里格外要紧。
        """
        if self.is_zero:
            return "0"
        times, minus, plus = (("·", " − ", " + ") if fmt == "text"
                              else ("*", " - ", " + "))

        def coef_str(k: Fraction) -> str:
            if k == 1:
                return ""
            if k == -1:
                return "-"
            s = (str(k.numerator) if k.denominator == 1
                 else f"{k.numerator}/{k.denominator}")
            return s + times

        parts: list[str] = []
        for sym, k in self.coeffs.items():       # 保持书写顺序
            if not parts:
                parts.append(coef_str(k) + sym)
            elif k < 0:
                parts.append(minus + coef_str(-k) + sym)
            else:
                parts.append(plus + coef_str(k) + sym)
        if self.const != 0:
            c = self.const
            bits = (str(c.numerator) if c.denominator == 1
                    else f"{c.numerator}/{c.denominator}")
            if not parts:
                return ("-" if c < 0 else "") + bits.lstrip("-")
            parts.append((minus + bits.lstrip("-")) if c < 0 else (plus + bits))
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"coeffs": {k: [v.numerator, v.denominator]
                           for k, v in self.coeffs.items()},
                "const": [self.const.numerator, self.const.denominator]}


# ---------------------------------------------------------------- 表达式解析

def _number(text: str) -> Fraction:
    """把字面量（含工程记法）解析成**精确**有理数。

    ★ 用 ``Decimal`` 而不是 ``float``：``0.1`` 在二进制浮点里是
    ``0.1000000000000000055…``，而这条路径的下游是"精确有理数"的
    节点电压法。表达式里出现一个 0.1 就把精度污染成浮点，
    两条精确路径"逐项严格相等"就再也比不出来了。

    支持 ``2k``、``4k7``（= 4.7k，英式写法）、``1meg``、``0.5m``。
    """
    m = _LITERAL_RE.match(text)
    if not m:
        raise CircuitError(f"表达式里的 {text!r} 不是合法数字")
    mant, suf, trail = m.group(1), m.group(2), m.group(3)
    if trail:
        # 英式写法：后缀后面的数字是**小数位**（4k7 = 4.7k）。
        # 尾数已经带小数点时这个写法没有意义（4.5k7 = ?），明确报错而不是猜。
        if "." in mant or "e" in mant or "E" in mant:
            raise CircuitError(
                f"表达式里的 {text!r} 写法有歧义：小数点与英式小数位同时出现了")
        mant = f"{mant}.{trail}"
    try:
        dec = Decimal(mant)
    except InvalidOperation:
        raise CircuitError(f"表达式里的 {text!r} 不是合法数字") from None
    if suf:
        dec = dec * (Decimal(10) ** _SUFFIXES[suf])
    return Fraction(dec)


class _Lin:
    """解析过程中的中间值：``(常数项, 系数表)``，带四则运算。

    单独一个类是因为"线性"这件事只能在**运算时**判定：
    两个带符号的量相乘就是二次，此时必须报错，而不是丢掉符号项。
    """

    __slots__ = ("const", "coeffs")

    def __init__(self, const: Fraction = Fraction(0),
                 coeffs: dict[str, Fraction] | None = None) -> None:
        self.const = Fraction(const)
        self.coeffs: dict[str, Fraction] = dict(coeffs or {})

    @property
    def is_const(self) -> bool:
        return not self.coeffs

    def __add__(self, o: "_Lin") -> "_Lin":
        out = dict(self.coeffs)
        for k, v in o.coeffs.items():
            out[k] = out.get(k, Fraction(0)) + v
        return _Lin(self.const + o.const, out)

    def __sub__(self, o: "_Lin") -> "_Lin":
        out = dict(self.coeffs)
        for k, v in o.coeffs.items():
            out[k] = out.get(k, Fraction(0)) - v
        return _Lin(self.const - o.const, out)

    def __neg__(self) -> "_Lin":
        return _Lin(-self.const, {k: -v for k, v in self.coeffs.items()})

    def __mul__(self, o: "_Lin") -> "_Lin":
        if self.is_const:
            return _Lin(self.const * o.const,
                        {k: self.const * v for k, v in o.coeffs.items()})
        if o.is_const:
            return _Lin(self.const * o.const,
                        {k: o.const * v for k, v in self.coeffs.items()})
        raise CircuitError(
            f"表达式里出现了两个含参数的量相乘（{self._fmt()} × {o._fmt()}）。"
            "受控源只支持**线性**受控关系（μ·u、rm·i、gm·u、α·i 这一类）："
            "非线性关系会让节点电压法与支路电流法这两条精确路径失效，"
            "而它们逐项严格相等正是本工具最靠得住的判据。"
            "请改写为线性的形式；确实需要非线性受控源，请直接用 ngspice。"
        )

    def __truediv__(self, o: "_Lin") -> "_Lin":
        if not o.is_const:
            raise CircuitError(
                f"表达式里除以了一个含参数的量（{o._fmt()}）—— "
                "那样一般不再是线性关系。请改写为线性形式。"
            )
        if o.const == 0:
            raise CircuitError("表达式里除了零")
        return _Lin(self.const / o.const,
                    {k: v / o.const for k, v in self.coeffs.items()})

    def _fmt(self) -> str:
        return LinearExpr(self.coeffs, self.const).describe()

    def out(self) -> LinearExpr:
        return LinearExpr(self.coeffs, self.const)


def tokenize(expr: str) -> list[tuple[str, str]]:
    """切成 ``[(类型, 文本), ...]``，类型是 ``num`` / ``ident`` / ``op``。

    暴露出来是给"改名时重写表达式"用的 —— 那里绝不能拿字符串
    ``replace()`` 硬替，否则把 ``u_1`` 改成 ``u_10`` 时，
    表达式里的 ``u_10`` 会被二次替换成 ``u_100``。按记号重写才不会串。
    """
    out: list[tuple[str, str]] = []
    pos = 0
    s = expr or ""
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            raise CircuitError(
                f"表达式 {expr!r} 里有个不认识的字符 {s[pos]!r}（位置 {pos}）。"
                "只支持数字、参数名、+ - * / 和括号。")
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def parse_linear(expr: str, table: "ParamTable | None" = None) -> LinearExpr:
    """解析线性表达式，返回系数表。

    ``table`` 给定时会校验每个符号都**确实在参数表里**；不认识的符号直接报错
    并列出可用名字 —— 这正是"避免参数指向错误"最有效的一招：
    表达式里写错一个名字，宁可当场被拦住，也不要静默当成 0。

    支持：``+ - * /``、括号、工程记法数字（``2k``、``0.5m``、``1meg``）、
    一元负号。**不支持**：函数、``^``、方括号、两个含参量相乘/相除。
    """
    toks = tokenize(expr)
    if not toks:
        raise CircuitError("表达式是空的")

    known = set(table.symbols()) if table is not None else None
    i = 0

    def peek() -> tuple[str, str] | None:
        return toks[i] if i < len(toks) else None

    def take() -> tuple[str, str]:
        nonlocal i
        t = toks[i]
        i += 1
        return t

    def primary() -> _Lin:
        nonlocal i
        t = peek()
        if t is None:
            raise CircuitError(f"表达式 {expr!r} 在应该还有内容的地方结束了")
        kind, text = t
        if kind == "op" and text == "(":
            take()
            v = addsub()
            nxt = peek()
            if nxt is None or nxt[0] != "op" or nxt[1] != ")":
                raise CircuitError(f"表达式 {expr!r} 的括号没配平")
            take()
            return v
        if kind == "num":
            take()
            return _Lin(_number(text))
        if kind == "ident":
            take()
            if known is not None and text not in known:
                near = sorted(known)
                hint = "、".join(near[:12]) + ("…" if len(near) > 12 else "")
                raise CircuitError(
                    f"表达式里的 {text!r} 不是已知参数名。可用的名字有：{hint}。"
                    "（参数名可在「参数」页里改，也可以直接用元件位号与结点名）")
            return _Lin(Fraction(0), {text: Fraction(1)})
        raise CircuitError(f"表达式 {expr!r} 里 {text!r} 出现在不该出现的位置")

    def unary() -> _Lin:
        t = peek()
        if t is not None and t[0] == "op" and t[1] in "+-":
            take()
            v = unary()
            return v if t[1] == "+" else -v
        return primary()

    def muldiv() -> _Lin:
        v = unary()
        while True:
            t = peek()
            if t is None or t[0] != "op" or t[1] not in "*/":
                return v
            take()
            r = unary()
            v = v * r if t[1] == "*" else v / r

    def addsub() -> _Lin:
        v = muldiv()
        while True:
            t = peek()
            if t is None or t[0] != "op" or t[1] not in "+-":
                return v
            take()
            r = muldiv()
            v = v + r if t[1] == "+" else v - r

    out = addsub()
    if i != len(toks):
        raise CircuitError(
            f"表达式 {expr!r} 在 {toks[i][1]!r} 处多了内容 —— 是不是漏了运算符？")
    return out.out()


def rewrite_symbol(expr: str, old: str, new: str) -> str:
    """把表达式里的符号 ``old`` 换成 ``new``（**按记号替换，其余字符一字不动**）。

    ★ 不能拿 ``str.replace``：把 ``u_1`` 改成 ``u_10`` 时，
    表达式里本来就有的 ``u_10`` 会被二次替换成 ``u_100``。
    所以必须有"记号边界"的判断 —— 只有**整个记号**恰好等于 old 才换。

    ★ 也不能"拆成记号再拼回去"：那样会**把用户自己写的空格全抹掉**。
    实测 ``3*u_2 + 5`` 改个名就变成 ``3*uA+5`` —— 语义没变，但用户看到
    自己写的式子被悄悄重排了一遍，会开始怀疑"是不是还改了别的"。
    表达式的正确性靠解析，可读性靠原文；这里只动命中的那一段字面量。
    """
    if not (expr or "").strip():
        return expr
    try:
        tokenize(expr)                # 语法关卡：本来就解析不了的原文一律不动
    except CircuitError:
        return expr
    return _IDENT_TOKEN_RE.sub(
        lambda m: new if m.group(1) == old else m.group(1), expr)


# ---------------------------------------------------------------- 参数表

@dataclass
class ParamTable:
    """一张电路的参数表：一批 ``Param`` + 它们的绑定。

    生命周期由 ``sync()`` 驱动：每次电路结构变化后调一次，
    它负责"保留旧名字、补新绑定、给失效绑定标孤儿"，并返回一份诊断。
    """

    items: list[Param] = field(default_factory=list)

    # ------------------------------------------------ 查询

    def symbols(self) -> list[str]:
        return [p.symbol for p in self.items]

    def index(self) -> dict[Binder, Param]:
        return {p.binder: p for p in self.items}

    def by_symbol(self, sym: str) -> Param | None:
        for p in self.items:
            if p.symbol == sym:
                return p
        return None

    def by_binder(self, binder: Binder) -> Param | None:
        return self.index().get((str(binder[0]), str(binder[1])))

    def live(self) -> list[Param]:
        """当前电路里还成立的参数。"""
        return [p for p in self.items if not p.orphan]

    def orphans(self) -> list[Param]:
        return [p for p in self.items if p.orphan]

    def by_quantity(self, quantity: str) -> list[Param]:
        return [p for p in self.live() if p.quantity == quantity]

    # ------------------------------------------------ 单位

    def unit_of(self, param: Param, circuit: "Circuit | None" = None) -> str:
        """这个参数的 SI 单位。

        ``value`` 类要看元件类型：R 是 Ω、C 是 F、受控源是 V/V / Ω / S / A/A。
        """
        q = param.quantity
        if q != "value":
            return QUANTITY_UNIT[q]
        kind = self._kind_of(param, circuit)
        return VALUE_UNIT.get(kind, "")

    def _kind_of(self, param: Param, circuit: "Circuit | None") -> str:
        if param.binder[0] != "value" or circuit is None:
            return ""
        try:
            return circuit.by_ref(param.binder[1]).kind
        except CircuitError:
            return ""

    # ------------------------------------------------ 同步

    def sync(self, circuit: "Circuit") -> dict[str, Any]:
        """把参数表对齐到当前电路。

        顺序即 UI 展示顺序：**节点电压 → 元件值 → 支路电流 → 支路电压**。
        按量分组而不是按元件行，是因为解题时人是按"先把所有电压求出来、
        再求电流"读的，跟教材的顺序一致。

        返回诊断字典：新增了哪些自动名、哪些绑定失效成了孤儿。
        诊断要往上抛，让报告能说"你这张表里有 2 个参数已经指向不存在的元件"。
        """
        wanted: list[Binder] = []
        for node in circuit.nodes:
            wanted.append(("node_u", node))
        for c in circuit.components:
            wanted.append(("value", c.ref))
        for c in circuit.components:
            wanted.append(("branch_i", c.ref))
        for c in circuit.components:
            wanted.append(("branch_u", c.ref))

        existing = {p.binder: p for p in self.items}
        used: set[str] = set()
        fresh: list[Param] = []

        for binder in wanted:
            p = existing.get(binder)
            if p is not None:
                # ★ 同一个名字不可能被两个绑定共用；真出现了说明表被外部改坏过，
                #   此时**保第一个、给后来的改名**，而不是让两个参数重名 ——
                #   重名会让表达式静默指向错的那个。
                if p.symbol in used:
                    p = Param(symbol=auto_symbol(binder, used), binder=binder,
                              auto=True, note=p.note)
                used.add(p.symbol)
                fresh.append(p)
            else:
                sym = auto_symbol(binder, used)
                used.add(sym)
                fresh.append(Param(symbol=sym, binder=binder, auto=True))

        # 表里有、电路里没有 → 孤儿（留着，并标出来）
        wanted_set = set(wanted)
        kept_orphans: list[Param] = []
        for p in self.items:
            if p.binder in wanted_set:
                continue
            if p.symbol in used:                 # 名字被新绑定占走了 → 也不能再重名
                p = Param(symbol=auto_symbol(p.binder, used), binder=p.binder,
                          auto=p.auto, orphan=True, note=p.note)
            used.add(p.symbol)
            p.orphan = True
            kept_orphans.append(p)

        self.items = fresh + kept_orphans

        return {
            "added": [p.symbol for p in fresh if p.auto],
            "orphans": [{"symbol": p.symbol, "binder": list(p.binder),
                         "binder_text": p.binder_text()} for p in kept_orphans],
            "count": len(fresh),
            "custom": sum(1 for p in fresh if not p.auto),
        }

    # ------------------------------------------------ 改名

    def rename(self, old: str, new: str,
               circuit: "Circuit | None" = None) -> dict[str, Any]:
        """改名。**并把所有引用旧名的表达式一起改写。**

        ★ 这一步不能省。表达式里存的是名字文本，名字改了而表达式没改，
        就得到"指向不存在的符号"的表达式：报错会推到很久以后的求解阶段，
        而且三条路径会一致地把它当 0 用（都不认识那个符号）——
        属于三法互校**拦不住**的那类错。

        返回 ``{"symbol": 新名, "rewritten": [被改写的受控源位号]}``。
        """
        new = str(new).strip()
        if not is_valid_symbol(new):
            raise CircuitError(
                f"参数名 {new!r} 不是合法标识符（字母/数字/下划线，不以数字开头）")
        p = self.by_symbol(old)
        if p is None:
            raise CircuitError(f"参数表里没有 {old!r}")
        clash = self.by_symbol(new)
        if clash is not None and clash is not p:
            raise CircuitError(
                f"参数名 {new!r} 已被 {clash.binder_text()} 占用，换一个")
        if new == old:
            return {"symbol": new, "rewritten": []}

        p.symbol = new
        p.auto = False                           # 用户动过了，不再算自动命名

        rewritten: list[str] = []
        if circuit is not None:
            for c in circuit.components:
                ctrl = getattr(c, "ctrl", None)
                if ctrl is None or not getattr(ctrl, "expr", ""):
                    continue
                new_expr = rewrite_symbol(ctrl.expr, old, new)
                if new_expr != ctrl.expr:
                    ctrl.expr = new_expr
                    rewritten.append(c.ref)
        return {"symbol": new, "rewritten": rewritten}

    def resolve_expression(self, expr: str) -> LinearExpr:
        """按本表校验并解析一个表达式（不认识的符号会被拦住）。"""
        return parse_linear(expr, self)

    # ------------------------------------------------ 序列化

    def to_dict(self) -> dict[str, Any]:
        return {"items": [p.to_dict() for p in self.items]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | Sequence[Any] | None) -> "ParamTable":
        if d is None:
            return cls()
        raw = d.get("items") if isinstance(d, Mapping) else d
        out: list[Param] = []
        for item in raw or []:
            try:
                out.append(Param.from_dict(item))
            except CircuitError:
                # 单条坏记录不该把整张表连坐 —— 如实丢掉它，
                # 剩下的照样能用（sync 会把缺的绑定重新补齐）
                continue
        return cls(items=out)

    def copy(self) -> "ParamTable":
        return ParamTable.from_dict(self.to_dict())


# ---------------------------------------------------------------- 界面视图

def _ensure_synced(circuit: "Circuit") -> ParamTable:
    table = circuit.params
    if not table.items:
        table.sync(circuit)
    return table


def params_view(circuit: "Circuit") -> dict[str, Any]:
    """参数表的**界面视图**：分组、标签、SI 单位、按绑定的索引，一次给全。

    为什么要放到 IR 层而不是在 API 里现拼：报告文本、界面表格、受控源表达式
    校验三处都要"参数名 ↔ 绑定"这一步翻译。各拼一份的结果是**同一张电路，
    报告里叫 ``u_3``、界面上叫别的名字** —— 正是"信息错位"的典型形态。
    翻译只此一处，三处都读它。
    """
    table = _ensure_synced(circuit)
    items: list[dict[str, Any]] = []
    for p in table.items:
        kind, key = p.binder
        unit = table.unit_of(p, circuit)
        entry: dict[str, Any] = {
            "symbol": p.symbol,
            "binder": [kind, key],
            "binder_kind": kind,
            "binder_text": p.binder_text(),
            "binder_label": BINDER_LABEL.get(kind, kind),
            "quantity": p.quantity,
            "quantity_label": QUANTITY_LABEL.get(p.quantity, p.quantity),
            "unit": unit,                     # ★ SI 单位，界面上每行都要标
            "auto": bool(p.auto),             # True = 系统自动命名，用户没动过
            "orphan": bool(p.orphan),
            "note": p.note,
            "value": None,
        }
        if kind == "value":
            try:
                entry["value"] = circuit.by_ref(key).value
            except CircuitError:
                entry["value"] = None
        elif kind == "node_u" and key == circuit.ref_node:
            entry["value"] = 0
        items.append(entry)

    groups = []
    for q in ("u", "i", "value"):
        rows = [e for e in items if e["quantity"] == q]
        if not rows:
            continue
        groups.append({
            "quantity": q,
            "label": QUANTITY_LABEL.get(q, q),
            "unit": QUANTITY_UNIT.get(q, ""),      # 表头用的量纲级单位
            "symbols": [e["symbol"] for e in rows],
        })

    controlled = []
    for c in circuit.controlled:
        ctrl = c.ctrl
        if ctrl is None:
            continue
        gain_param = table.by_binder(("value", c.ref))
        controlled.append({
            "ref": c.ref,
            "kind": c.kind,
            "gain_symbol": GAIN_SYMBOL.get(c.kind, ""),
            "gain_note": GAIN_NOTE.get(c.kind, ""),
            "unit": VALUE_UNIT.get(c.kind, ""),
            "value": c.value,
            "value_param": gain_param.symbol if gain_param else "",
            "mode": ctrl.mode,
            "control": c.control_text(),
            # ★★ 三个名字必须分开给，**不许**合成一个 "sampling"：
            #   ``ref``（本行的键）是**受控源自己**的位号 —— 早先界面直接拿它
            #   当"被采样支路"的当前值去选中下拉框，而选项里又刻意排除了自己，
            #   于是永远选不上、界面上显示"（未定）"，看起来像用户没填过。
            #   被采样支路必须**单独一个字段**。
            "sampled_ref": ctrl.ref,          # 题目里说的那条被采样支路（R1）
            "probe_ref": ctrl.sense_ref,      # 系统插入的 0V 探针（没有则空）
            "sampling_ref": ctrl.sampling,    # 求解实际取电流的那条 = probe or sampled
            "sampling_is_probe": bool(ctrl.sense_ref and ctrl.sense_ref != ctrl.ref),
            "control_nodes": list(ctrl.nodes or ()),
            "expr": ctrl.expr,
            "net_name": ("B" + c.ref) if ctrl.expr else c.ref,
        })

    return {
        "items": items,
        "groups": groups,
        "controlled": controlled,
        "orphans": [e["symbol"] for e in items if e["orphan"]],
        "auto_count": sum(1 for e in items if e["auto"]),
        "custom_count": sum(1 for e in items if not e["auto"]),
        "count": len(items),
        # ★ 按绑定的索引：报告要按 (量, 键) 找回名字时读它，
        #   而**不要**自己拼名字字符串（拼法一旦和 auto_symbol 分家，
        #   报告里就会出现一个表里不存在的名字）。
        "by_binder": {f"{e['binder_kind']}:{e['binder'][1]}": e["symbol"]
                      for e in items},
        "units": {"header": "单位一律 SI（V / A / Ω / F / H / S）；"
                            "受控源增益按种类分别为 V/V、S、Ω、A/A",
                  "quantity": dict(QUANTITY_UNIT),
                  "value_by_kind": dict(VALUE_UNIT)},
    }


def apply_param_edits(circuit: "Circuit", *,
                      renames: Mapping[str, str] | None = None,
                      exprs: Mapping[str, str] | None = None,
                      ctrls: Mapping[str, Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """把界面上改的**参数名**、**受控源表达式**与**控制关系**写回电路。

    ``ctrls``：``{受控源位号: {"nodes": ["3","0"]}}``（电压控制）
    或 ``{受控源位号: {"ref": "R1"}}``（电流控制）。
    ★ 必须能改控制关系，否则界面上把某个元件改成受控源之后就**没法用了**：
    它的控制支路永远是空的，求解只会说"控制端没定"，而用户找不到地方去定它。

    ★ 先在副本上全套做完，成功才写回。改名会连带改写表达式、表达式又要靠
    参数表解析 —— 中途失败就地写回会留下"一半新一半旧"的参数表，
    而它长得跟正常的一模一样，这种半程状态是最难查的一类。
    这里的选择是：**要么全成，要么原样不动**，并把失败原因原样抛给用户。
    三样东西在同一个事务里，是因为它们互相引用（表达式引参数名、
    控制关系决定表达式里能不能用某个电流），分开提交必然出现中间态。

    ``exprs`` 里值为空串表示"取消自定义表达式，回到用增益的标准形"。
    """
    work = circuit.copy()
    if not work.params.items:
        work.sync_params()

    renamed: list[dict[str, Any]] = []
    for old, new in (renames or {}).items():
        old, new = str(old).strip(), str(new).strip()
        if not old or old == new:
            continue
        info = work.params.rename(old, new, work)
        info["from"] = old
        renamed.append(info)

    expr_done: list[dict[str, Any]] = []
    for ref, text in (exprs or {}).items():
        ref = str(ref).strip()
        comp = work.by_ref(ref)                      # 不存在就抛，位置精确
        if getattr(comp, "ctrl", None) is None:
            raise CircuitError(
                f"{ref} 不是受控源，不能给它填控制表达式"
                f"（它的类型是 {comp.kind}）")
        text = str(text or "").strip()
        if text:
            # 解析一次即校验：未知符号、非线性（u_1*u_2 / 1/u_1）、语法错
            # 都会在这里被精准拒绝，而不是拖到求解阶段变成一句"算不出来"。
            work.params.resolve_expression(text)
        elif comp.value is None:
            raise CircuitError(
                f"{ref}: 清掉自定义表达式之后它既没有增益也没有控制关系，"
                f"请先填上增益（{GAIN_NOTE.get(comp.kind, '增益')}）")
        comp.ctrl.expr = text
        expr_done.append({"ref": ref, "expr": text,
                          "mode": "自定义表达式" if text else "标准形（增益）"})

    # ---- 控制关系本身（控制端结点 / 被采样支路）
    # ★ 这一项必须能改，否则界面上把某个元件**改成受控源之后就没法用了**：
    #   它的控制支路永远是空的，求解时只会得到一句"控制端没定"，
    #   而用户找不到任何地方去定它。手绘画布同样需要它 ——
    #   在画布上画一个受控源，控制端是图上哪两个点，只有用户知道。
    ctrl_done: list[dict[str, Any]] = []
    for ref, spec in (ctrls or {}).items():
        ref = str(ref).strip()
        comp = work.by_ref(ref)
        if comp.ctrl is None:
            raise CircuitError(
                f"{ref} 不是受控源，不能给它指定控制支路（它的类型是 {comp.kind}）")
        spec = spec or {}
        want_mode = str(spec.get("mode") or comp.ctrl.mode).strip().upper()
        if want_mode != comp.ctrl.mode:
            raise CircuitError(
                f"{ref}: 控制方式由元件类型决定 —— {comp.kind}（{CONTROL_NOTE.get(comp.kind, '')}）"
                f"只能是{'电压' if comp.ctrl.mode == 'V' else '电流'}控制，"
                f"不能改成{'电压' if want_mode == 'V' else '电流'}控制。"
                f"要换控制方式，请先把元件类型改成 {'E/G' if want_mode == 'V' else 'H/F'}。")
        if want_mode == "V":
            nodes = spec.get("nodes")
            if nodes is not None:
                nodes = [str(n).strip() for n in nodes]
                if len(nodes) != 2:
                    raise CircuitError(f"{ref}: 电压控制要给出**两个**控制端结点，收到 {nodes}")
                if nodes[0] == nodes[1]:
                    raise CircuitError(
                        f"{ref}: 控制端的两点都是 {nodes[0]} —— 控制量恒为 0，"
                        "这等于一个值为 0 的独立源，多半是选错了。")
                unknown = [n for n in nodes if n not in set(work.nodes)]
                if unknown:
                    raise CircuitError(
                        f"{ref}: 控制端结点 {unknown} 不在电路里（现有结点：{sorted(set(work.nodes))}）")
                comp.ctrl.nodes = (nodes[0], nodes[1])
                # 换控制端不影响被采样支路，但为稳妥起见把探针记录清掉 ——
                # 电流控制的探针只跟被采样支路有关，这里不动它。
        else:
            tref = (spec.get("ref")
                    if spec.get("ref") is not None else comp.ctrl.ref)
            tref = str(tref or "").strip()
            if not tref:
                raise CircuitError(f"{ref}: 电流控制要给出被采样的支路位号")
            if tref == ref:
                raise CircuitError(f"{ref}: 不能采样自己的电流（自指）—— 那会给出自己等于自己的方程")
            target = work.by_ref(tref)               # 不存在就抛
            # ★ 换了被采样支路，旧探针源就没用了：留着它会在网表与参数表里
            #   留下一个"为 A 而插、现在采的是 B"的 0V 源 —— 那是个纯粹的多余元件，
            #   而它会让节点数/支路数都多一个。清掉 sense_ref 让求解时重新插。
            if comp.ctrl.ref != tref or comp.ctrl.sense_ref:
                comp.ctrl.sense_ref = ""
            comp.ctrl.ref = tref
            if target is comp:
                raise CircuitError(f"{ref}: 被采样支路不能是它自己")
        ctrl_done.append({"ref": ref, "mode": want_mode,
                          "nodes": list(comp.ctrl.nodes or ()),
                          "ref": comp.ctrl.ref})

    # 改完必须过一遍结构校验，否则会把一张"表里有名字、电路里没有"的参数表
    # 存进会话 —— 等到用户点求解才报错，那时已经分不清是谁改坏的。
    work.validate(allow_incomplete=True)

    circuit.params = work.params
    src = {c.ref: c for c in work.components}
    for c in circuit.components:
        other = src.get(c.ref)
        if other is None or c.ctrl is None or other.ctrl is None:
            continue
        # ★ 只把**该改的**搬回原件。整块 `c.ctrl = other.ctrl` 会把探针位号、
        #   控制表达式一起换掉，那看起来也没错 —— 但如果 `other.ctrl` 是
        #   在副本里被 `ensure_sense_sources` 之类改过的中间态，搬回来就等于
        #   把"求解时才该发生的事"提前写回了会话，界面上的接线会当场变得对不上。
        c.ctrl.expr = other.ctrl.expr
        c.ctrl.nodes = other.ctrl.nodes
        c.ctrl.ref = other.ctrl.ref
        c.ctrl.sense_ref = other.ctrl.sense_ref
    return {"renames": renamed, "exprs": expr_done, "ctrls": ctrl_done,
            "params": params_view(circuit)}
