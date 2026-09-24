"""本地 OCR 层：用 Windows 内置识别引擎读原理图上的位号与数值。

选 Windows 内置引擎而不是 Tesseract/EasyOCR，理由有三：不用装额外运行时、
不用管理员权限、完全离线（电路图可能涉及未公开的东西，不该默默上传）。

**这一层的能力边界是实测出来的，不是猜的。** 下面每条都是在一台真机上
跑出来的结果，写在这里是为了防止后来者（包括未来的我）重新踩一遍：

实测结论 A：**短文本行会被静默丢弃**
    单独渲染的 ``R1`` / ``C1`` / ``V1`` / ``I1`` / ``L1`` / ``L2`` /
    ``1k`` / ``1M``，在 24/32/40/48/60 五种字号下**全部返回空**。
    这不是"认错"，是版面分析阶段整条就没被当成文本。
    而且**救不回来**：加导线、加矩形框、加描边、放大到 100px、整图缩放
    0.5~2.0 倍、白底黑字换成黑底白字、纵向膨胀合并相邻行 —— 全试过，全无效。
    唯一有效的是"这一行本身有足够多的字"（``R1 10k`` 同行就能读出 ``RI 1 Ok``），
    但真实版式里位号在上、数值在下，本来就是两行，所以这条帮不上忙。
    影响：``R1``/``C1`` 这类短位号在真实原理图里**大概率读不到**。

实测结论 B：**能读出来的时候，混淆有很强的规律**
    - ``1`` → ``I``：``R11`` → ``RI 1``；``R1 10k`` → ``RI 1 Ok``
    - ``0`` → ``O``：``0.1uF`` → ``O.1uF``；``10k`` → ``Ok``
    - 小数点 → 各种别的东西：``．``(全角句点) ``·``(间隔号) ``，``(全角逗号)
      ``，`` 甚至**直接消失变成空格**：``4.7k`` → ``4 7k``、``3.3V`` → ``3 3V``
    - ``m`` 被拆成 ``rn``：``2.2mH`` → ``2 2rnH``
    - ``V`` 被当成汉字：``3.3V`` → ``3 ． 3 伊``
    数值里**纯数字加单位**的部分很可靠：``100uF``/``220``/``12V``/``2N3904``
    在所有字号下都对。

实测结论 C：**环境限制**
    - ``Language`` 类在 ``winrt.windows.globalization``，不在 ``media.ocr``。
    - ``OcrEngine`` **没有** ``max_image_dimension`` 这个属性（此版本没有）。
    - ``try_create_from_language`` 对未安装的语言返回 ``None`` 而不是抛异常 ——
      所以"引擎建不出来"和"语言没装"是同一件事，必须靠回退链处理。
    - 这台机器只装了 ``zh-Hans-CN``，``en-US``/``ja-JP``/``zh-Hant-TW`` 都没有。
    - ``SoftwareBitmap.create_copy_from_buffer`` 只接受 **4 个参数**
      （buffer, format, width, height），传 alpha mode 会 ``Invalid parameter count``。
    - ``bounding_rect`` 返回的 ``Rect`` **不可迭代**，必须取
      ``.x/.y/.width/.height``。
    - ``recognize_async`` 返回 ``IAsyncOperation``，用 ``.get()`` 同步取结果。
      ★ 它是**阻塞**的，在 FastAPI 的 async 端点里直接调用会卡住事件循环，
      必须放到线程池里跑。

基于以上，这一层的设计立场是：**能读多少读多少，读不到的要能说出"那里有字"
而不是当它不存在**。所以除了 OCR，还独立做一遍几何上的"文本块检测"，
用来发现"有字但没读出来"的位置 —— 这正是本项目"不静默降级"的落实。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- 常量

#: 语言回退链的兜底。放在最后，因为它是"用户配置里指定的都没装"时的
#: 最后一条路：用系统当前语言配置建引擎（实测可用）。
FALLBACK_LANGS = ("zh-Hans-CN", "zh-Hant-TW", "en-US", "ja-JP")

#: 文本块检测时，认为"这是一个字"的高度范围（像素）。
#: 太小的是噪点或线头，太大的是符号本体或标题。
GLYPH_MIN_H = 8
GLYPH_MAX_H = 90

#: 长宽比超过这个值就认为是导线而不是文字（导线又细又长）。
WIRE_ASPECT = 6.0
#: 连成一片的墨迹面积占比超过这个值，认为是符号本体/填充块，不是文字。
BLOB_MAX_FILL = 0.62

#: 细轮廓围出的空洞占外接矩形的比例超过它 → 是符号本体（或闭合导线回路），不是字。
#: 实测字符最高 0.398、符号轮廓最低 0.686，取中间 0.5。
GLYPH_RING_HOLE_RATIO = 0.5
#: 字符的填充率下限。实测字符 0.43~0.49、符号轮廓 0.10~0.17。
GLYPH_MIN_FILL = 0.30

#: 抹掉文字时，在字符外接矩形外再扩这么多像素。
#: 只扩一点点是为了吃掉二值化阈值差异留下的抗锯齿灰边；
#: **不能扩多** —— 扩多了就会擦到紧邻的元件或导线。
TEXT_MASK_DILATE = 2

# ★ 上面这三个比值判据是为"**抹掉文字墨迹**"这个用途收紧的
# （见 ``mask_text_regions``）。原来它们只用来报告"有字没读出来"，
# 收得松一点无所谓；现在要拿去擦图，一旦把元件轮廓当成字擦掉，元件就凭空消失了。
# 本机实测（msyh 字体 + 合成符号），三列都是尺度无关的比值：
#
#   图形                 aspect   fill    洞/bbox
#   字符 R  (12x21)       1.75    0.437   0.000
#   字符 1  ( 7x20)       2.86    0.493   0.000
#   字符 0  (13x20)       1.54    0.473   0.377
#   字符 0  (45x70)       1.56    0.434   0.398   ← 字号放大 3.5 倍，比值几乎不变
#   电阻框  (161x61)      2.64    0.174   0.826   ← 要排除
#   空圆    (121x121)     1.00    0.099   0.686   ← 要排除
#   粗导线  (281x12)     23.42    1.000   0.000   ← 要排除
#
# 于是三条判据：``aspect > WIRE_ASPECT`` → 导线；
# ``洞/bbox > GLYPH_RING_HOLE_RATIO`` → 细轮廓围出大片空白 = 符号本体；
# ``fill < GLYPH_MIN_FILL`` → 笔画太稀疏的轮廓。

#: 词与词之间水平间距超过 高度×此值 就算换了一个文本块。
WORD_GAP_RATIO = 1.6


# ---------------------------------------------------------------- 结果类型


@dataclass
class OcrWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    #: OCR 原始文本（未经混淆修正）。保留下来是为了可追溯：
    #: 用户看到 ``R1`` 时，要能查出它是从 ``RI`` 修正来的。
    raw: str = ""
    #: 应用了哪些修正，逐条留痕
    fixes: list[str] = field(default_factory=list)
    #: 所属的 OCR 行序号
    line: int = 0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text, "x": self.x, "y": self.y,
                             "w": self.w, "h": self.h, "line": self.line}
        if self.raw != self.text:
            d["raw"] = self.raw
        if self.fixes:
            d["fixes"] = list(self.fixes)
        return d


@dataclass
class TextBlob:
    """几何检测出来的"这里有一块文字"（不依赖 OCR 是否读出来）。"""

    x: int
    y: int
    w: int
    h: int
    #: 被哪个 OCR 词覆盖了（None = 有字但没读出来）
    matched_word: str | None = None
    #: 里面有几个字（按连通块数估）
    glyph_count: int = 0
    #: ★ 组成这个文本块的**每个字符各自的外接矩形**。
    #: 抹文字时只擦这些矩形（外扩 ``TEXT_MASK_DILATE``），
    #: **不擦整个文本块的外接矩形** —— 后者会把紧邻的元件或导线一起擦掉。
    #: 实测代价：一个字的标签贴在电压源圆边上时，擦整块的外接矩形
    #: 会把圆环切断，圆就没有洞了，电压源**凭空消失**且不报错。
    glyph_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "glyph_count": self.glyph_count, "matched": self.matched_word,
                "glyph_boxes": [list(b) for b in self.glyph_boxes]}


@dataclass
class OcrResult:
    available: bool
    #: 实际用上的引擎语言（回退链走到哪一步）
    lang: str | None = None
    #: 请求过但没装的语言 —— 必须报出来，否则用户会以为是自己配错了
    missing_langs: list[str] = field(default_factory=list)
    words: list[OcrWord] = field(default_factory=list)
    #: 几何检测到的所有文本块
    blobs: list[TextBlob] = field(default_factory=list)
    #: ★ 有字但没读出来的文本块 —— 这是这一层最重要的输出之一
    dropped: list[TextBlob] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)
    #: 引擎不可用的原因（available=False 时）
    reason: str = ""
    hint: str = ""
    #: ★ 不可用的**种类**，决定上层拿它当软问题还是结构缺失：
    #:
    #: - ``"engine_missing"`` / ``"lang_missing"`` —— 环境缺件。**预期内的正常分支**：
    #:   用户没装语言包就会这样，连接照样由几何层定，缺的只是位号与数值，
    #:   人工填一下即可。上层记 soft，**不花一次模型调用**。
    #: - ``"image_unreadable"`` —— 传进来的图片没被接住（类型不认识 / 文件读不开）。
    #:   这是本层的 bug 或调用方误用，**不能**混进上面那一类：
    #:   实测踩过 —— 管线传了个 ``pathlib.Path`` 进来，本层只认 ``str``，
    #:   于是全部位号数值读不出来，而症状只是一条 soft 警告，
    #:   网表照样出、还不升级，等于"静默给出缺值的答案"。
    #: - ``"engine_error"`` —— 引擎在跑的过程中出错（图太大/格式特殊）。同样是异常。
    reason_kind: str = ""

    @property
    def ok(self) -> bool:
        return self.available

    def text_of_blob(self, blob: TextBlob) -> str:
        return blob.matched_word or ""

    def nearest_word(self, x: float, y: float, *, max_dist: float | None = None) -> OcrWord | None:
        """找离 (x, y) 最近的词。定位元件的位号/数值时用。"""
        best: OcrWord | None = None
        best_d = float("inf")
        for w in self.words:
            d = ((w.cx - x) ** 2 + (w.cy - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = w, d
        if best is None:
            return None
        if max_dist is not None and best_d > max_dist:
            return None
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "lang": self.lang,
            "missing_langs": list(self.missing_langs),
            "reason": self.reason,
            "reason_kind": self.reason_kind,
            "hint": self.hint,
            "image_size": {"w": self.image_size[0], "h": self.image_size[1]},
            "word_count": len(self.words),
            "words": [w.to_dict() for w in self.words],
            "blobs": [b.to_dict() for b in self.blobs],
            "dropped_count": len(self.dropped),
            "dropped": [b.to_dict() for b in self.dropped],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------- 混淆修正
#
# ★ 铁律：修正**只作用于位号与数值这两个语法位置**，绝不作用于自由文本。
#   理由是 "RI" 在图纸说明文字里可能真的就是 "RI"（比如作者署名、型号前缀），
#   盲改会把用户写对的东西改错。而在位号 ``R?`` 的数字位上，
#   字母本来就不可能出现，所以那里把 ``I`` 改成 ``1`` 是安全的。

#: 位号：1~3 个字母 + 可选的数字，例如 R1 / R10 / C2 / V1 / L1 / I1 / Q3 / U1A
_REFDES_RE = re.compile(r"^([A-Za-z]{1,3})(\d{0,4})([A-Za-z])?$")

#: 数字位上可能出现的字母 → 正确数字。
_DIGIT_SLOT_FIX = {
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "T": "7",
}

#: ★ 只在"整个位号只有两个字母、第二个本该是数字"时才敢动的替换子集。
#:
#: 为什么必须比 _DIGIT_SLOT_FIX 窄得多：``RI`` 要变成 ``R1`` 靠的是
#: "位号数字位上不可能有字母"。但当整个 token 没有数字位时（``RI`` 被正则
#: 整个吃进字母前缀），"第二个字母其实是数字"只是一个**推测**。
#: 所以这里只留形状几乎与数字重合的几个：``I/l/|/!`` 之于 ``1``、
#: ``O/o`` 之于 ``0``。
#:
#: 反例（说明为什么不能放宽）：``LED`` 的末位 ``D`` 在 _DIGIT_SLOT_FIX 里
#: 对应 ``0``，一旦放宽就会把 ``LED`` 改成 ``LE0``；``RT``（热敏电阻）
#: 会被改成 ``R7``。这两个都是会把对的改错，比不改更糟。
_TRAILING_LETTER_AS_DIGIT = {
    "I": "1", "l": "1", "|": "1", "!": "1",
    "O": "0", "o": "0",
}

#: ★ 只允许这几种**开头字母**被当成数字。
#:
#: 依据是位号命名惯例：为避免与 0/1 混淆，标准的元件类别字母里**不含 O**。
#: 所以开头的 ``O`` 只可能是 ``0`` 被认错（实测：``0k`` → ``Ok``、
#: ``0.1uF`` → ``O.1uF``）。
#:
#: 而开头的 ``I`` **绝不能动** —— 本项目用 ``I`` 表示理想电流源，
#: ``I1`` 是合法位号。如果把它改成 ``11``，电流源就变成了一个不存在的元件，
#: 而且这个错误会一路传进求解器，三法还会一致地给出同一个错答案。
_LEADING_LETTER_AS_DIGIT = {"O": "0", "o": "0"}

#: 数值里可能出现的字母 → 正确数字（只在**数字语境**里改）
_VALUE_DIGIT_FIX = dict(_DIGIT_SLOT_FIX)

#: 小数点的各种错误写法（实测出现的全在这里）。
#: 注意 ``，``(U+FF0C 全角逗号) 和 ``,``(半角逗号) 都在，因为 OCR 真会这么给。
_DECIMAL_CHARS = {
    "．": ".",   # U+FF0E 全角句点
    "。": ".",   # U+3002 中文句号
    "｡": ".",
    "·": ".",   # U+00B7 间隔号
    "•": ".",
    "，": ".",   # U+FF0C 全角逗号
    ",": ".",
    "`": ".",
    "'": ".",
}

#: ``m`` 被拆成 ``rn``（实测：2.2mH → 2 2rnH）。只在数值里替换。
_LIGATURE_FIX = (("rn", "m"), ("rri", "m"), ("iii", "m"), ("Ill", "m"))

#: 汉字顶替拉丁字母（实测：3.3V → 3 ． 3 伊）
_CJK_TO_LATIN = {
    "伊": "V", "∨": "V", "ν": "v",
    "扒": "R", "尺": "R",
    "亡": "L", "乙": "L",
    "工": "I", "王": "I",
    "口": "O", "回": "O",
    "四": "0", "○": "0", "〇": "0",
    "一": "1", "丨": "1",
}

#: 单位与 SI 前缀里允许出现的字母（用于判断"这串更像位号还是更像数值"）
_UNIT_LETTERS = set("RVACLHFWSmkKMunpµμΩGTP")


def _normalize_decimal_sep(s: str) -> tuple[str, list[str]]:
    """把各种假的小数点归一成 ``.``，并处理"小数点消失成空格"的情况。"""
    fixes: list[str] = []
    out = []
    for ch in s:
        if ch in _DECIMAL_CHARS:
            out.append(".")
        else:
            out.append(ch)
    if "".join(out) != s:
        fixes.append(f"小数点写法归一：{s!r} → {''.join(out)!r}")
    t = "".join(out)

    # 小数点两侧的空格：OCR 常把 ``4.7k`` 切成 ``4 ． 7k`` 这种带空格的形状，
    # 上一步把 ``．`` 换成 ``.`` 之后还剩两个空格，这里一并清掉。
    t2 = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", t)
    if t2 != t:
        fixes.append(f"去掉小数点两侧的空格：{t!r} → {t2!r}")
        t = t2

    # 数字与单位之间多出的空格：实测 ``3.3V`` 会被读成 ``3 ． 3 伊``，
    # 汉字换回拉丁字母后就成了 ``3.3 V``。只有在**这一串已经含小数点**时才敢
    # 删这个空格 —— 有小数点就已经是数值无疑，字母必然是单位。
    # 不含小数点时不动，因为 ``R1 10k`` 这类多词行是靠空格分子串的。
    if "." in t:
        t2 = re.sub(r"(?<=\d)\s+(?=[A-Za-zµμΩ]+$)", "", t)
        if t2 != t:
            fixes.append(f"去掉数字与单位之间的空格：{t!r} → {t2!r}")
            t = t2

    # 数字-空格-数字：几乎必然是丢掉的小数点。
    # 实测：4.7k→'4 7k'、3.3V→'3 3V'、0.01uF→'0 01uF'、2.2mH→'2 2mH'
    # 只在这两侧都是数字且整串看起来像数值时才动手。
    m = re.search(r"(?<=\d) +(?=\d)", t)
    while m:
        t2 = t[:m.start()] + "." + t[m.end():]
        if t2.count(".") > 1:
            break
        fixes.append(f"两位数字之间的空格 → '.'（小数点被 OCR 丢掉）：{t!r} → {t2!r}")
        t = t2
        m = re.search(r"(?<=\d) +(?=\d)", t)
    return t, fixes


def fix_refdes(text: str) -> tuple[str, list[str]]:
    """修正位号。只改数字位上的字母，字母位一律不动。

    ``RI`` → ``R1``、``CI`` → ``C1``、``VI`` → ``V1``、``LI`` → ``L1``。
    为什么安全：位号的数字位上按定义不可能出现字母，
    所以那里出现的 ``I``/``O`` 只可能是 OCR 把 ``1``/``0`` 看错了。
    而**首字母**绝不动 —— ``I1`` 是合法的电流源位号，把首字母 ``I`` 改成 ``1``
    会把电流源变成不存在的元件。
    """
    fixes: list[str] = []
    s = text.strip()
    if not s:
        return s, fixes

    # 先统一大小写形式上的可疑字符
    for bad, good in _CJK_TO_LATIN.items():
        if bad in s:
            s2 = s.replace(bad, good)
            fixes.append(f"{bad!r} → {good!r}（汉字误替拉丁字母）：{s!r} → {s2!r}")
            s = s2

    m = _REFDES_RE.match(s)
    if not m:
        return s, fixes
    prefix, digits, suffix = m.group(1), m.group(2), m.group(3) or ""

    # ---- 情况一：整个 token 只有字母，没有数字位（``RI`` 被贪婪匹配全吃进前缀）
    # 此时若"第二个字母其实是数字"，把它挪到数字位去。
    # 只在**恰好两个字符**且末位属于那几张高置信表时才动手 —— 见常量的注释。
    if not digits and not suffix and len(prefix) == 2 and prefix[1] in _TRAILING_LETTER_AS_DIGIT:
        head, tail = prefix[0], prefix[1]
        if head.isalpha():
            new = head + _TRAILING_LETTER_AS_DIGIT[tail]
            fixes.append(
                f"位号 {s!r} 的第二个字符 {tail!r} 疑为数字 {_TRAILING_LETTER_AS_DIGIT[tail]!r} 被认错"
                f"（Windows OCR 会把 1 读成 I、0 读成 O）：{s!r} → {new!r}。"
                "请核对。")
            return new, fixes
        return s, fixes

    new_digits = []
    for ch in digits:
        if ch.isdigit():
            new_digits.append(ch)
        elif ch in _DIGIT_SLOT_FIX:
            new_digits.append(_DIGIT_SLOT_FIX[ch])
            fixes.append(f"位号数字位 {ch!r} → {_DIGIT_SLOT_FIX[ch]!r}")
        else:
            new_digits.append(ch)
    # 后缀字母：位号里偶见的 A/B（如 U1A），也可能是 1 被看成 l
    new_suffix = _DIGIT_SLOT_FIX.get(suffix, suffix) if suffix else ""
    if suffix and new_suffix != suffix:
        fixes.append(f"位号后缀 {suffix!r} → {new_suffix!r}")

    out = prefix + "".join(new_digits) + new_suffix
    if out != text.strip():
        fixes.append(f"位号修正：{text.strip()!r} → {out!r}")
    return out, fixes


def fix_value(text: str) -> tuple[str, list[str]]:
    """修正数值。只做"显然的转录错误"，不做任何单位换算或量级推断。

    ``Ok`` → ``0k``、``O.1uF`` → ``0.1uF``、``4 7k`` → ``4.7k``、
    ``2 2rnH`` → ``2.2mH``。

    ★ 刻意**不**去猜 ``1M`` 到底是兆欧还是毫欧 —— 那是量级问题，
    属于 ``app/ingest/values.py`` 的职责，那边有专门的人工核对警告。
    这里只管"把字母改回它本该是的数字"。
    """
    fixes: list[str] = []
    s = text.strip()
    if not s:
        return s, fixes

    for bad, good in _CJK_TO_LATIN.items():
        if bad in s:
            s2 = s.replace(bad, good)
            fixes.append(f"{bad!r} → {good!r}（汉字误替拉丁字母）：{s!r} → {s2!r}")
            s = s2

    for bad, good in _LIGATURE_FIX:
        if bad in s:
            s2 = s.replace(bad, good)
            fixes.append(f"连字 {bad!r} → {good!r}：{s!r} → {s2!r}")
            s = s2

    s, f2 = _normalize_decimal_sep(s)
    fixes.extend(f2)

    # 字母 → 数字。
    #
    # 中间位置：只在"该字母紧邻数字"时才动手。这样 "12V" 里的 V、
    # "100uF" 里的 u/F、"4k7" 里的 k 都不会被动。
    #
    # ★ 开头位置单独处理，且只认 O/o（见 _LEADING_LETTER_AS_DIGIT）。
    #   如果开头也允许"紧邻数字就改"，``I1``（理想电流源位号）会被改成 ``11`` ——
    #   一个会把元件类型改掉的错误。所以开头一律走最窄的那张表。
    chars = list(s)
    for i, ch in enumerate(chars):
        if ch not in _VALUE_DIGIT_FIX:
            continue
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if i == 0:
            if ch not in _LEADING_LETTER_AS_DIGIT:
                continue
            rest = "".join(chars[1:])
            unit_like = bool(rest) and all(c in _UNIT_LETTERS for c in rest)
            if nxt == "." or unit_like:
                chars[i] = _LEADING_LETTER_AS_DIGIT[ch]
                fixes.append(
                    f"数值开头的 {ch!r} → {chars[i]!r}（0 被读成 O）："
                    f"{s!r} → {''.join(chars)!r}")
            continue
        if chars[i - 1].isdigit() or nxt.isdigit():
            chars[i] = _VALUE_DIGIT_FIX[ch]
            fixes.append(f"数值里 {ch!r} → {chars[i]!r}（紧邻数字）：{s!r} → {''.join(chars)!r}")
    out = "".join(chars)

    if out != text.strip():
        fixes.append(f"数值修正：{text.strip()!r} → {out!r}")
    return out, fixes


def looks_like_refdes(text: str) -> bool:
    """判断一个词像不像位号。用于决定该用哪套修正规则。

    ★ **必须带数字**。位号按定义是"类别字母 + 序号"，``R`` / ``C`` / ``V``
    这种只有字母的短串**不是**位号，而是"一个被 OCR 拆散的位号的头一段"
    （实测：``R12`` 会被读成 ``['R','1','2']`` 三个词）。
    不把这一条写进判据的后果是：那个 ``R`` 会被当成一个**合法位号**认领下来，
    于是元件名变成 ``R`` —— 既不报错、也不提示，还挡住了真正的位号
    ``R12``（因为词已经被"认领"了）。
    """
    s = text.strip()
    if not s or len(s) > 6:
        return False
    if not any(c.isalpha() for c in s):
        return False
    if not any(c.isdigit() for c in s):
        return False
    return bool(_REFDES_RE.match(s))


def looks_like_value(text: str) -> bool:
    """判断一个词像不像数值（工程记法）。"""
    s = text.strip()
    if not s or len(s) > 12:
        return False
    if not any(c.isdigit() for c in s):
        return False
    # 必须有数字打头，或者修正小数点后能数字打头
    return bool(re.match(r"^[-+]?\d", s)) or bool(re.match(r"^[-+]?\.\d", s))


def correct_token(text: str, *, kind: str = "auto") -> tuple[str, list[str]]:
    """按语法位置修正一个 token。``kind`` 可为 ``refdes`` / ``value`` / ``auto``。

    ``auto`` 的做法是**两条路都走一遍，再看哪条的结果站得住**：

    - ``fix_value`` 的成果若以数字（或 ``.``）打头且含数字 → 认它是数值；
    - ``fix_refdes`` 的成果若符合位号形状 → 认它是位号；
    - 两边都成立时，看**原始 token 的第一个字符**：字母打头（且不是那张
      窄表里的 ``O``）就判位号，否则判数值。

    这个"两边都试"不是偷懒，是因为 OCR 的错误本身会让 token 的形状失去判别力：
    ``Ok`` 是数值 ``0k`` 被读坏的样子，而它看起来完全像一个字母打头的位号。
    单靠正则判不出来，只能靠"修完之后哪种解释能站住"来判。
    """
    s = text.strip()
    if not s:
        return s, []
    if kind == "refdes":
        return fix_refdes(s)
    if kind == "value":
        return fix_value(s)

    v, v_fixes = fix_value(s)
    r, r_fixes = fix_refdes(s)

    v_is_value = bool(re.match(r"^[-+]?[\d.]", v)) and any(c.isdigit() for c in v)
    r_is_refdes = looks_like_refdes(r)

    if v_is_value and not r_is_refdes:
        return v, v_fixes
    if r_is_refdes and not v_is_value:
        return r, r_fixes
    if v_is_value and r_is_refdes:
        # 原始就是"小写/数字打头"或开头是那张窄表里的 O → 更可能是数值
        if s[0].isalpha() and s[0] not in _LEADING_LETTER_AS_DIGIT:
            return r, r_fixes
        return v, v_fixes
    return s, []


# ---------------------------------------------------------------- 引擎


def available_languages() -> list[str]:
    """系统已安装的 OCR 识别语言。引擎不可用时返回空表（不抛）。"""
    try:
        from winrt.windows.media.ocr import OcrEngine
        return [L.language_tag for L in OcrEngine.available_recognizer_languages]
    except Exception:                          # noqa: BLE001
        return []


def create_engine(preferred: Iterable[str] | None = None
                  ) -> tuple[Any | None, str | None, list[str], str, str]:
    """按回退链建引擎。

    返回 ``(引擎或None, 用上的语言, 请求过但缺的语言, 失败原因, 建议)``。

    为什么要这么啰嗦：``try_create_from_language`` 对未安装的语言返回
    ``None``，于是"语言没装"和"引擎建不出来"表现为同一个结果。
    如果不把"我试过哪些语言、哪些没装"报出来，用户只会看到一句
    "OCR 不可用"，然后无从下手 —— 而实际上他只需要去
    设置 → 时间和语言 → 语言和区域 里加一个语言包。
    """
    wanted = [x for x in (preferred or FALLBACK_LANGS) if x]
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine
    except Exception as e:                     # noqa: BLE001
        return (None, None, [], f"Windows OCR 运行时不可用：{type(e).__name__}: {e}",
                "安装 winrt 组件：pip install winrt-runtime winrt-Windows.Media.Ocr "
                "winrt-Windows.Globalization winrt-Windows.Graphics.Imaging "
                "winrt-Windows.Storage.Streams winrt-Windows.Foundation")

    missing: list[str] = []
    installed = set(available_languages())

    # 用户配置的语言优先；一个都没成功再看系统里装了哪些，最后兜底
    order = list(wanted) + [t for t in sorted(installed) if t not in wanted]
    for tag in order:
        try:
            eng = OcrEngine.try_create_from_language(Language(tag))
        except Exception:                      # noqa: BLE001
            eng = None
        if eng is not None:
            for t in wanted:
                if t != tag and t not in installed:
                    missing.append(t)
            return eng, tag, sorted(set(missing)), "", ""

    # 指定语言全失败 → 用系统当前语言配置（这是实测里最稳的一条路）
    try:
        eng = OcrEngine.try_create_from_user_profile_languages()
    except Exception:                          # noqa: BLE001
        eng = None
    if eng is not None:
        for t in wanted:
            if t not in installed:
                missing.append(t)
        return (eng, getattr(eng.recognizer_language, "language_tag", None),
                sorted(set(missing)), "",
                "配置里指定的 OCR 语言都没安装，已改用系统当前语言配置。")

    return (None, None, sorted(set(wanted)),
            "没有任何可用的 OCR 语言包。",
            "打开 设置 → 时间和语言 → 语言和区域，为中文（或英文）添加"
            "「可选语言功能 → 光学字符识别」。装好后重启本服务。")


def _to_software_bitmap(im: Any) -> Any:
    """PIL Image → SoftwareBitmap。

    ★ ``create_copy_from_buffer`` **只接受 4 个参数**（buffer, format, w, h）。
    多传一个 alpha mode 会得到 ``TypeError: Invalid parameter count``，
    而那个错误信息完全不提参数个数，很容易往"格式不对"的方向查错。
    """
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.storage.streams import Buffer

    rgba = im if im.mode == "RGBA" else im.convert("RGBA")
    px = rgba.tobytes()
    buf = Buffer(len(px))
    buf.length = len(px)
    memoryview(buf)[:] = px
    return SoftwareBitmap.create_copy_from_buffer(
        buf, BitmapPixelFormat.RGBA8, rgba.size[0], rgba.size[1])


# ---------------------------------------------------------------- 识别


def recognize(image: Any, *, langs: Iterable[str] | None = None,
              correct: bool = True,
              detect_blobs: bool = True) -> OcrResult:
    """对一张图做 OCR。

    ``image`` 可以是文件路径、bytes 或 PIL Image。

    **永不抛异常**：引擎不可用、图读不开、语言没装，都返回
    ``available=False`` 的结构化结果 + 原因 + 建议。
    理由是这一层失败是**预期内**的正常分支（用户没装语言包就会这样），
    让它抛异常会逼着每个调用点写 try，最后总有人忘了写。

    ★ 在 FastAPI 的 async 端点里调用本函数必须放进线程池 ——
    ``recognize_async().get()`` 是阻塞的，会卡住事件循环。
    """
    from PIL import Image as PILImage

    warnings: list[str] = []
    engine, lang, missing, reason, hint = create_engine(langs)
    result = OcrResult(available=engine is not None, lang=lang,
                       missing_langs=missing, reason=reason, hint=hint)
    if not result.available:
        # 引擎建不出来 = 环境缺件（语言包 / winrt 运行时）。这是**预期内的分支**，
        # 与"图片没被接住"要分开记（见 OcrResult.reason_kind）。
        result.reason_kind = "lang_missing" if missing else "engine_missing"

    if missing:
        result.warnings.append(
            "以下 OCR 语言没有安装： " + "、".join(missing)
            + "。已用「" + str(lang) + "」顶替，识别结果可能偏弱。")

    # 图要先读进来 —— 即使引擎不可用也要读，因为文本块检测不需要引擎
    im = None
    try:
        if isinstance(image, (str, Path)):
            im = PILImage.open(image)
        elif isinstance(image, (bytes, bytearray, memoryview)):
            im = PILImage.open(io.BytesIO(bytes(image)))
        elif hasattr(image, "read"):            # 文件对象
            im = PILImage.open(image)
        else:
            im = image                          # 已经是 PIL.Image
        if hasattr(im, "load"):
            im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB") if "A" not in im.mode else im.convert("RGBA")
    except Exception as e:                     # noqa: BLE001
        # ★ 这里必须分清两件事，它们的处置**完全不同**：
        #   (a) 环境缺件（语言包没装、winrt 没装）—— 预期内的正常分支，
        #       OCR 没了不影响连接，值可以人工填，所以上层只记一条 soft；
        #   (b) **调用方给的东西我们没接住**（类型不认识、文件读不开）——
        #       这是本层的 bug 或误用。它绝不能伪装成"OCR 环境没装"：
        #       实测踩过 —— 管线传进来一个 ``pathlib.Path``，本函数只认 ``str``，
        #       于是整张图的位号与数值**全部为空**，而症状只是一条 soft 警告，
        #       网表照样出、还不升级给模型。那是"静默给出缺值的答案"。
        result.available = False
        result.reason = f"图片读取失败：{type(e).__name__}: {e}"
        result.reason_kind = "image_unreadable"
        result.hint = ("★ 这一条**不是**「OCR 环境没装」，而是这一层没接住传进来的图片。"
                       "本函数收「文件路径(str/Path) / bytes / PIL.Image / 文件对象」四种，"
                       "请核对调用方传的是什么。")
        return result

    result.image_size = (im.size[0], im.size[1])

    # ---- 几何文本块检测（与 OCR 无关，所以在引擎不可用时照样能做）
    blobs: list[TextBlob] = []
    if detect_blobs:
        try:
            blobs = detect_text_blobs(im)
            result.blobs = blobs
        except Exception as e:                 # noqa: BLE001
            warnings.append(f"文本块检测失败（不影响 OCR）：{type(e).__name__}: {e}")

    if engine is None:
        result.dropped = blobs
        if blobs:
            result.warnings.append(
                f"OCR 引擎不可用，但几何上检出 {len(blobs)} 个文本块 —— "
                "这些位置的位号/数值需要人工填写或改用视觉模型。")
        result.warnings.extend(warnings)
        return result

    # ---- 真正的 OCR
    try:
        from winrt.windows.media.ocr import OcrEngine  # noqa: F401
        sb = _to_software_bitmap(im)
        raw = engine.recognize_async(sb).get()
    except Exception as e:                     # noqa: BLE001
        result.available = False
        result.reason = f"OCR 识别过程出错：{type(e).__name__}: {e}"
        result.reason_kind = "engine_error"
        result.hint = "这通常是图片尺寸过大或格式特殊导致的，可先缩放到 4000px 以内。"
        result.dropped = blobs
        result.warnings.extend(warnings)
        return result

    words: list[OcrWord] = []
    for li, line in enumerate(raw.lines):
        for w in line.words:
            r = w.bounding_rect            # ★ Rect 不可迭代，取属性
            txt = w.text or ""
            ow = OcrWord(text=txt, x=int(r.x), y=int(r.y),
                         w=max(1, int(r.width)), h=max(1, int(r.height)),
                         raw=txt, line=li)
            if correct:
                # 先按数值判，再按位号判 —— 顺序见 correct_token 的注释
                new, fx = correct_token(txt, kind="auto")
                if new != txt:
                    ow.text = new
                    ow.fixes = fx
            words.append(ow)
    result.words = words
    result.warnings.extend(warnings)

    # ---- 把文本块和词对上，找出"有字但没读出来"的
    if blobs:
        unmatched = []
        for b in blobs:
            hit = _covering_word(b, words)
            if hit is None:
                unmatched.append(b)
            else:
                b.matched_word = hit.text
        result.blobs = blobs
        result.dropped = unmatched
        if unmatched:
            result.warnings.append(
                f"几何上检出 {len(blobs)} 个文本块，OCR 只读出 "
                f"{len(blobs) - len(unmatched)} 个，有 {len(unmatched)} 处"
                "「有字但没读出来」（Windows OCR 对只有一两个字符的短文本"
                "会整条跳过，实测无解）。这些位置需要人工填写或改用视觉模型。")
        if correct:
            n_fixed = sum(1 for w in words if w.fixes)
            if n_fixed:
                result.warnings.append(
                    f"对 {n_fixed} 个词做了混淆修正（例如 1↔I、0↔O、"
                    "小数点被读成 ·／．／逗号或丢成空格）。每个词的原始文本"
                    "与修正明细都记在 words 里，可逐条核对。")

    return result


def _covering_word(blob: TextBlob, words: list[OcrWord]) -> OcrWord | None:
    """判断某个文本块是否被某个 OCR 词覆盖。

    用"交集面积占块面积的比例"而不是"中心点在不在块内"：
    因为 OCR 给的 bbox 常常比文本块略小或略偏移，
    用中心点判定会把"其实读出来了"误判成"没读出来"。
    """
    if not words:
        return None
    bx2, by2 = blob.x + blob.w, blob.y + blob.h
    best, best_ov = None, 0.0
    for w in words:
        wx2, wy2 = w.x + w.w, w.y + w.h
        ox = max(0, min(bx2, wx2) - max(blob.x, w.x))
        oy = max(0, min(by2, wy2) - max(blob.y, w.y))
        ov = ox * oy
        if ov <= 0:
            continue
        ratio = ov / max(1, min(blob.w * blob.h, w.w * w.h))
        if ratio > best_ov:
            best, best_ov = w, ratio
    return best if best_ov >= 0.35 else None


# ---------------------------------------------------------------- 文本块检测
#
# 为什么需要它：Windows OCR 会把 ``R1`` 这种短 token 整条丢掉（实测结论 A），
# 如果只信 OCR 的输出，代码里就看不到"这里本该有个位号"。
# 于是那个元件会被当成"没有位号"处理，用户也收不到任何提示 —— 静默降级。
#
# 这一段只回答一个问题："图里哪儿有文字？" 它不需要认识那些字，
# 所以用的是纯粹的几何判据（连通块尺寸 / 长宽比 / 填充率）。
# 它与 symbols.py 的闭合空洞法互补：那个找符号，这个找文字。


def _hole_areas(sub) -> list[int]:
    """一个墨迹块内部所有闭合空洞的面积。

    这里是 ``symbols.find_holes`` 的极简版，**刻意不 import symbols** ——
    两边职责不同（那边要找元件、这里只判"这片墨迹是不是细轮廓"），
    共用一个函数会把两个模块的判据绑死，改一边动另一边。
    """
    import numpy as np
    from scipy import ndimage

    filled = ndimage.binary_fill_holes(sub)
    holes = filled & ~sub
    if not holes.any():
        return []
    lbl, n = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    return [int((lbl == i).sum()) for i in range(1, n + 1)]


def detect_text_blobs(image: Any, *, merge_gap_ratio: float = WORD_GAP_RATIO
                      ) -> list[TextBlob]:
    """几何检测文本块。返回按阅读顺序（先上后下、先左后右）排好的列表。"""
    import numpy as np
    from scipy import ndimage

    im = image
    if not hasattr(im, "convert"):
        from PIL import Image as PILImage
        im = PILImage.open(im)
    g = np.array(im.convert("L"))
    ink = g < 160                              # 墨迹

    lbl, n = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []

    objs = ndimage.find_objects(lbl)
    glyphs: list[tuple[int, int, int, int]] = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if h <= 0 or w <= 0:
            continue
        area = int((lbl[sl] == i).sum())
        # 太细太长 → 导线。
        # ★ 这里**不能**再加"长度超过 4 倍字高才算导线"那种附加条件：
        # 一条 12px 宽、281px 长的粗导线长宽比 23、长度却不到 360，
        # 加上那条条件它就会掉进"字"里，抹文字时整根线被擦掉。
        # 单字符的长宽比上限实测只有 2.86，6.0 这条线很安全。
        if max(w, h) / max(1, min(w, h)) > WIRE_ASPECT:
            continue
        # 高度不在文字范围 → 不是字
        if not (GLYPH_MIN_H <= h <= GLYPH_MAX_H):
            continue
        # 实心大块 → 符号本体 / 填充
        if area / (w * h) > BLOB_MAX_FILL and min(w, h) > GLYPH_MIN_H * 2:
            continue
        # 细轮廓围出大片空白 → 符号本体（电阻框、电压源圆、闭合导线回路）。
        # ★ 这一条是"抹文字"用途下必须的：电阻框的填充率 0.17、长宽比 2.6、
        # 高度 60，前三条判据全都拦不住它，于是它会被当成一个"字"、
        # 进而在抹文字时把整个电阻擦掉 —— 元件凭空消失，且不报错。
        sub = lbl[sl] == i
        holes = _hole_areas(sub)
        if holes and max(holes) / float(w * h) > GLYPH_RING_HOLE_RATIO:
            continue
        if area / (w * h) < GLYPH_MIN_FILL:
            continue
        glyphs.append((xs.start, ys.start, w, h))

    if not glyphs:
        return []

    # 按行聚类：竖直方向有重叠就归同一行
    glyphs.sort(key=lambda t: (t[1], t[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    for gbox in glyphs:
        gx, gy, gw, gh = gbox
        placed = False
        for row in rows:
            rx, ry, rw, rh = _union(row)
            # 竖直重叠超过较矮者的 50%
            oy = min(ry + rh, gy + gh) - max(ry, gy)
            if oy > 0.5 * min(rh, gh):
                row.append(gbox)
                placed = True
                break
        if not placed:
            rows.append([gbox])

    blobs: list[TextBlob] = []
    for row in rows:
        row.sort(key=lambda t: t[0])
        # 行内按水平间距切成若干块
        cur: list[tuple[int, int, int, int]] = [row[0]]
        for gbox in row[1:]:
            px, py, pw, ph = _union(cur)
            _, gy, _, gh = gbox
            gap = gbox[0] - (px + pw)
            if gap > WORD_GAP_RATIO * max(gh, 1):
                blobs.append(_mk_blob(cur))
                cur = [gbox]
            else:
                cur.append(gbox)
        blobs.append(_mk_blob(cur))

    blobs = [b for b in blobs if b.glyph_count >= 1]
    blobs.sort(key=lambda b: (b.y, b.x))
    return blobs


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _mk_blob(boxes: list[tuple[int, int, int, int]]) -> TextBlob:
    x, y, w, h = _union(boxes)
    return TextBlob(x=x, y=y, w=w, h=h, glyph_count=len(boxes),
                    glyph_boxes=list(boxes))


def mask_text_regions(image: Any, blobs: list[TextBlob],
                      *, dilate: int = TEXT_MASK_DILATE
                      ) -> tuple[Any, dict[str, Any]]:
    """把文字墨迹涂白，返回 ``(新图, 报告)``。**不改任何几何。**

    ★ 为什么必须做这一步（它是本地层能不能用的分水岭）：

    文字墨迹与元件墨迹在二值图上没有任何区别。于是
      - 一个文字的「0」有自己的闭合空洞 → 被判成一个电容/电压源符号；
      - 「R1 10k」几块字的墨迹被当一个"元件本体"，它的 slot_region
        会盖住下面的导线，导线被切碎成好几段；
      - 结果：**凭空多出元件、结点数变多、IR 不连通**。
    而真实电路图**几乎都是有标注的**，所以这不是边角情况，
    是"本地层在真图上直接不可用"。

    ★ 只擦**逐个字符的**外接矩形（``TextBlob.glyph_boxes``），
    **不擦整个文本块的外接矩形**。这一点是踩出来的：
    一个字的标签若是贴在电压源的圆边上，擦整个词的外接矩形会把圆环切断，
    圆就没有闭合空洞了，于是**电压源凭空消失、且不报任何错**。
    逐字符擦的代价是可能留下个别未成组的笔画残墨（残墨顶多多出一个假元件，
    会被"未解释墨迹"报出来），远比切断元件安全。

    做法上故意选了"涂白"而不是"从 mask 里减掉"：涂白得到一张普通图片，
    ``detect_symbols`` / ``build_wire_graph`` 不用改签名就能共用；
    而且是抹背景色、不动任何坐标，回绘叠图照样对得上。

    ★ 报告必须交出去（不静默降级）：擦了哪些区域、擦掉多少墨迹，
    都要能让用户核对 —— 万一擦掉的是元件，只能靠人看出来。
    """
    import numpy as np
    from PIL import Image as PILImage

    im = image
    if not hasattr(im, "convert"):
        im = PILImage.open(im)
    if hasattr(im, "load"):
        im.load()
    im = im.convert("RGB")
    if not blobs:
        return im, {"masked_boxes": [], "masked_pixels": 0, "blob_count": 0,
                    "glyph_count": 0, "note": "没有检出文本块，未做任何涂白。"}

    arr = np.array(im)
    W, H = im.size
    boxes: list[tuple[int, int, int, int]] = []
    n_glyph = 0
    before = int((arr.sum(axis=2) < 3 * 160).sum())
    for b in blobs:
        # 兜底：万一某个文本块没记下逐字符矩形（旧对象/外部构造），
        # 退回用它自己的外接矩形，但**不外扩**，尽量少擦。
        src = b.glyph_boxes or [(b.x, b.y, b.w, b.h)]
        for (gx, gy, gw, gh) in src:
            n_glyph += 1
            x0, y0 = max(0, gx - dilate), max(0, gy - dilate)
            x1, y1 = min(W, gx + gw + dilate), min(H, gy + gh + dilate)
            if x1 <= x0 or y1 <= y0:
                continue
            arr[y0:y1, x0:x1] = 255
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    out = PILImage.fromarray(arr)
    after = int((arr.sum(axis=2) < 3 * 160).sum())
    report = {
        "masked_boxes": boxes,
        "blob_count": len(blobs),
        "glyph_count": n_glyph,
        "masked_pixels": before - after,
        "note": (f"把 {len(blobs)} 块文字（共 {n_glyph} 个字符）的外接矩形涂白了，"
                 f"抹掉 {before - after} 个墨迹像素。"
                 "只擦逐个字符的矩形、不擦整词矩形，是为了不切断紧邻的元件或导线；"
                 "文字墨迹与元件墨迹在二值图上无法区分，不涂白会凭空多出元件、"
                 "并把导线切成碎段。请核对涂白区域有没有盖住元件。"),
    }
    return out, report


def probe() -> dict[str, Any]:
    """给 /api/health 用的体检报告。不启动识别，只看引擎能不能建起来。"""
    langs = available_languages()
    engine, used, missing, reason, hint = create_engine()
    return {
        "available": engine is not None,
        "installed_languages": langs,
        "engine_language": used,
        "missing_languages": missing,
        "reason": reason,
        "hint": hint,
    }


# ---------------------------------------------------------------- 人工修正
#
# 为什么要有这一层：OCR 的错分两类，而**只有人能判**。
#
# 1. **读错**：``RI`` 其实想写 ``R1``，``Ok`` 其实想写 ``0k``。模板修正能救一部分，
#    但救不完 —— 模板是猜的，猜错时没有任何机制能发现。
# 2. **没读出来**：Windows OCR 把 ``R1`` 这种短 token **整条丢掉**（实测无解，
#    见 tests/_probe_ocr_lanelen.py）。这一处连"想读什么"都不知道，
#    只能靠几何层告诉我们"这里本来有字"，再由人填。
#
# ★ 所以这一层的设计原则是：**人不改的，原样保留；人改的，逐条留痕。**
#   绝不"顺手也修一下" —— 用户改完的字如果被本层再猜一次，
#   他就永远不知道自己看到的是自己的输入还是机器的猜测。


def _blob_copy(b: TextBlob) -> TextBlob:
    """复制一个文本块。**必须复制，不能共享** —— 见 apply_ocr_edits 里的注释。"""
    return TextBlob(x=b.x, y=b.y, w=b.w, h=b.h, matched_word=b.matched_word,
                    glyph_count=b.glyph_count, glyph_boxes=list(b.glyph_boxes))


def apply_ocr_edits(res: OcrResult | None,
                    edits: list[dict[str, Any]] | None) -> tuple[OcrResult | None, list[str]]:
    """按界面上的修改产出一份**新的** ``OcrResult``（原对象不动）。

    支持的编辑（``i`` 是**原始** ``words`` 里的下标，从 0 开始）：

    ============  ==========================================================
    ``set``       改第 i 个词的文本：``{"op":"set","i":3,"text":"R1"}``
    ``drop``      丢掉第 i 个词（读出来的是噪声/水印）：``{"op":"drop","i":3}``
    ``add``       补一个词（填"有字没读出来"的块）：``{"op":"add","text":"R1",
                  "x":..,"y":..,"w":..,"h":..}``
    ============  ==========================================================

    返回 ``(新结果, 说明清单)``。说明清单是**给人看的**，逐条讲清
    "原来是什么 → 现在是什么"，直接摆到界面上，不静默。

    ★ 调用方应当**永远从"原样"重放全部编辑**，而不是在上一次的结果上继续改：
    这样"改错了再改回来"是真的回得去，也不需要任何撤销栈。
    """
    notes: list[str] = []
    if res is None:
        return None, notes

    edits = [e for e in (edits or []) if isinstance(e, dict)]
    if not edits:
        return res, notes

    # 深拷贝：本函数不许碰到调用方那份（它可能还要被重放/对照）
    words: list[OcrWord] = [
        OcrWord(text=w.text, x=w.x, y=w.y, w=w.w, h=w.h,
                raw=(w.raw or w.text), fixes=list(w.fixes), line=w.line)
        for w in res.words
    ]
    n_orig = len(words)
    dropped_idx: list[int] = []

    for e in edits:
        op = str(e.get("op") or "").strip()
        if op in ("set", "drop"):
            try:
                i = int(e.get("i"))
            except (TypeError, ValueError):
                notes.append(f"忽略一条编辑：{op} 缺少合法的下标 i。")
                continue
            if not (0 <= i < n_orig):
                notes.append(f"忽略一条编辑：下标 {i} 超出范围（原始只有 {n_orig} 个词）。")
                continue
            if op == "set":
                new = str(e.get("text") if e.get("text") is not None else "").strip()
                old = words[i].text
                if new == old:
                    continue
                words[i].text = new
                words[i].fixes = list(words[i].fixes) + [f"人工改为 {new!r}"]
                notes.append(f"第 {i} 个词：{old!r} → {new!r}（人工）")
            else:
                if i in dropped_idx:
                    continue
                dropped_idx.append(i)
                notes.append(f"第 {i} 个词：{words[i].text!r} 被标为噪声并丢弃（人工）")
        elif op == "add":
            txt = str(e.get("text") or "").strip()
            if not txt:
                continue
            try:
                x, y = float(e.get("x")), float(e.get("y"))
                w = float(e.get("w") or 0)
                h = float(e.get("h") or 0)
            except (TypeError, ValueError):
                notes.append("忽略一条新增：坐标不是数字。")
                continue
            if w <= 0 or h <= 0:
                # 没给尺寸就按字数估一个：位号/数值一般是等宽的几个字符。
                # 这只是"给个能用的 bbox"，不是还原真实字形 —— 说明里写清楚。
                h = h if h > 0 else 20.0
                w = w if w > 0 else max(14.0, 0.62 * h * len(txt))
                notes.append(f"新增词 {txt!r} 没给尺寸，按 {int(w)}×{int(h)} 估算"
                             "（位置准、框大小是估的，只影响显示不影响判定）。")
            ow = OcrWord(text=txt, x=int(x), y=int(y), w=int(w), h=int(h),
                         raw="", fixes=["人工补填（原始 OCR 没读出来）"],
                         line=-1)
            words.append(ow)
            notes.append(f"新增词：{txt!r} @({int(x)},{int(y)})（人工补填）")
        else:
            notes.append(f"忽略一条不认识的编辑：{op!r}。"
                         "目前只支持 set / drop / add 三种。")

    # 丢掉被标为噪声的词（按下标，倒着删）
    for i in sorted(dropped_idx, reverse=True):
        if 0 <= i < n_orig:
            words.pop(i)
    # 补填的词接在后面，按阅读顺序重排一次，让界面与"最近的词"查找都自然
    words.sort(key=lambda w: (w.line if w.line >= 0 else 10 ** 6, w.y, w.x))

    out = OcrResult(
        available=res.available, lang=res.lang,
        missing_langs=list(res.missing_langs),
        words=words,
        # ★ 块也必须逐个复制。`list(res.blobs)` 只复制了列表，元素还是同一批对象 ——
        #   下面要改 `b.matched_word`，那就会**改到调用方那一份**。
        #   后果很隐蔽：调用方拿"原样"重放编辑时，看到的是上一次改过的 matched_word，
        #   于是"改错了再改回去"回不到原样，而界面上看不出为什么。
        blobs=[_blob_copy(b) for b in res.blobs],
        dropped=[_blob_copy(b) for b in res.dropped],
        image_size=res.image_size,
        warnings=list(res.warnings),
        reason=res.reason, hint=res.hint, reason_kind=res.reason_kind,
    )

    # ---- 重算"哪个块读出来了、哪个块没读出来"
    # 人工补填的词必须能让对应的块从 dropped 里出来，否则界面上会出现
    # "明明填了字、却还报有字没读出来"这种自相矛盾。
    if out.blobs:
        unmatched: list[TextBlob] = []
        for b in out.blobs:
            b.matched_word = None
            hit = _covering_word(b, words)
            if hit is None:
                unmatched.append(b)
            else:
                b.matched_word = hit.text
        out.dropped = unmatched

    if notes:
        out.warnings.append(f"人工修正了 {len(notes)} 处文字：" + "；".join(notes))
    return out, notes
