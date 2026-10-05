# -*- coding: utf-8 -*-
"""两级视觉通道编排：**本地优先，结构缺失才升级给视觉大模型**。

这个模块是"补充视觉逻辑"这条需求的落地处。用户已确认的四条口径，逐条写死在这里：

1. **本地优先**：本地这一层（几何 + Win 内置 OCR + 模板匹配）能出网表就不调模型。
   省一次调用、也省一次把整张图送出去。
2. **分级升级：只有"结构缺失"才升级**。
   - 结构性缺失（符号定位不到、导线追踪失败、拓扑有歧义、类型认不出）→ 升级给模型；
   - 非结构性（某个电阻的数值看不清、位号被读成 ``RI``）→ **本地照出网表**，
     把那一位标 ``needs_human`` 交人工填，不花这笔钱。
   理由：「不许靠肉眼定连接」这条禁令针对的是**连接**；数值读错是另一类问题 ——
   而且它在三法互校里**拦不住**（三条路径共用同一份 IR），本来就只能靠人。
3. **升级时整张图交给模型**，并对它的结论**给最大信任**：
   直接采用模型给出的网表结构，不是只让它补几段文字。
4. **最大信任 ≠ 隐藏不确定**：``source="vlm"`` 必须如实记录、原始返回必须留档、
   叠图核对必须跑、模型自报的低置信度必须照实压下来。

调用方式（第二点是硬要求，写在这里免得忘）：

- ``run()`` 是**同步阻塞**的（内含阻塞式 OCR 与阻塞式 HTTP）。
  在 FastAPI 的 async 端点里必须丢进线程池（``run_in_threadpool``），
  否则会卡住事件循环 —— 这个坑在 ``ocr``/``vlm`` 两个模块里都踩过一次记录。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..ingest.values import parse_engineering
from ..ir.model import (
    ALLOWED_KINDS,
    CONFIDENCE_GATE,
    CONTROLLED_KINDS,
    CONTROL_NOTE,
    REF_NODE,
    Circuit,
    CircuitError,
    Component,
    Evidence,
)
from . import ocr as OCR
from . import symbols as SY
from . import wires as WR
from .config import load_config

# ---------------------------------------------------------------- 参数

#: 位号的搜索半径 = 元件本体尺寸乘这个系数。
#:
#: ★ 它与 ``VALUE_SEARCH_FACTOR`` **必须同档**，这不是"顺手调大"，是修一个实测缺陷：
#: 位号与数值是**同一个元件并排写的两个标签**（教科书一律 ``R1`` 在上、``10k`` 在下，
#: 或左右并排），离本体的距离根本没有理由差一档。实测（合成图，元件框 57px 宽）：
#:
#: ===========  ==============  ==============
#: 文字离框    位号(1.1)        数值(1.6)
#: ===========  ==============  ==============
#: 70px         认不到          认到了
#: 30px         认不到          认到了
#: 20px         认不到          认到了
#: 14px         认到了          认到了
#: ===========  ==============  ==============
#:
#: 后果不是"少读一个名字"，而是**界面自相矛盾**：用户在文字校对表里明明
#: 看到并把 ``RI`` 改成了 ``R1``，网表却仍写"位号没读出来，已按类型自动编号"
#: （还可能碰巧撞成同一个名字，于是看起来"改了没反应"）。
#: 也就是说，位号本来是用户最想手工纠正的一格，却恰好是够不着的那一格。
#:
#: 半径放大带来的新风险（元件自己没标位号时去抢旁边那个的）由 ``_owns_refdes``
#: 的**归属判据**兜住：一个 ``R…`` 的词只属于离它最近的那个 R 槽位。
REFDES_SEARCH_FACTOR = 1.6
#: 数值的搜索半径系数。与位号同档，理由见上。
VALUE_SEARCH_FACTOR = 1.6

#: 模型报出的置信度上限（对齐 PROJECT 的"最大信任"口径）：
#: 模型没自报置信度时用这个值。它**高于** ``CONFIDENCE_GATE``，
#: 即"最大信任"的落地 —— 不因为它不是 exact 就强制人工确认。
#: 但这不等于免检：``source="vlm"`` 会如实写进 Evidence、原始返回会留档。
VLM_DEFAULT_CONFIDENCE = 0.9

#: 从图上读不出极性时，电压源/电流源给的置信度。低于闸门 → 强制人工确认极性。
POLARITY_UNREAD_CONFIDENCE = 0.6

#: 数字读不清时给的置信度。
VALUE_UNREAD_CONFIDENCE = 0.6

#: 档位
VALID_TIERS = ("local", "vlm", "none")

# ---------------------------------------------------------------- 拆词拼合
#
# 实测（Windows 内置 OCR，逐条见 tests/test_vision.py 的 C 组）：**字符一拉开间距，
# 一个位号就会被读成好几个词** —— ``R12`` → ``['R','1','2']``、``10k`` → ``['1','0','k']``。
# 这不是边角情况：字号偏小、教科书排版字距偏大时都会出现。
#
# ★ 缝宽**不能**当判据。实测把两类缝按"字高的倍数"量出来是重叠的
#   （见 tests/_probe_ocr_lanelen.py 第③组，可复跑）：
#
#     token 内被拆开的字符缝：0.24 ~ 1.44 字高
#     正常词间空格：          0.39 ~ 0.80 字高
#
#   所以只能靠**形状**判：位号的头是"只有字母的短段"，其余各段是**单个数字字符**。

#: 位号的"头"：只有字母的短段（``R`` / ``C`` / ``V``…）。它**本身不是位号**。
_REFDES_HEAD_RE = re.compile(r"^[A-Za-z]{1,3}$")
#: 位号被拆开后的数字段：**单个**数字字符。
_REFDES_DIGIT_FRAG_RE = re.compile(r"^\d$")
#: 纯单位词（``k`` / ``uF`` / ``V``）—— 数值被按字符切开时，单位会单独成一个词。
_BARE_UNIT_RE = re.compile(r"^[A-Za-zµμΩ]{1,3}$")
#: 至多拼几个数字段。教科书图里位号极少超过两位数，多于此就不敢认了。
MAX_REFDES_DIGIT_FRAGS = 3
#: 拼接时允许的水平缝上限（字高的倍数）。只当**上限**用，不当判据 ——
#: 定 1.3 是保守取法：实测 token 内缝可到 1.44 字高，所以字距特别大的那几张图
#: 会有拼不上的情况。拼不上就退回"自动编号 + 提示人工核对"，是**安全的那一侧**
#: （真正决定成败的是形状判据，缝宽只是防止把远处的东西吃进来）。
SPLIT_GAP_MAX_FACTOR = 1.3


# ---------------------------------------------------------------- 结构


@dataclass
class Issue:
    """一条"本地层没把握"的记录。``severity`` 决定要不要升级给模型。"""

    severity: str          # "structural" | "soft"
    code: str
    text: str

    @property
    def structural(self) -> bool:
        return self.severity == "structural"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "text": self.text}


@dataclass
class LocalOutcome:
    """本地一遍的完整痕迹。**即使最终采用模型结论，这一遍也必须留档。**"""

    ran: bool = False
    circuit: Circuit | None = None
    symbols: SY.SymbolReport | None = None
    graph: WR.WireGraph | None = None
    ocr: OCR.OcrResult | None = None
    #: ★ **人工校对之后**真正生效的那一份。``ocr`` 永远保持"机器原始读数"不动。
    #:
    #: 为什么要分成两份，而不是校完直接覆盖 ``ocr``：
    #: 界面上的下标 ``i``（"第 3 个词"）必须**永远指向原始读数**，
    #: 否则用户改一次、词表顺次挪位，第二次编辑就会改到另一个词上。
    #: 而且"从原始重放全部编辑"要求那份原始读数始终在 —— 一旦被覆盖，
    #: 把改过的字**改回原样就回不去了**（实测踩过：清空编辑清单后数值没还原，
    #: 因为清单是空的了、但基线已经不是原样）。
    ocr_effective: OCR.OcrResult | None = None
    issues: list[Issue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    #: 涂白文字那一趟的回执（擦了哪些区域）。留着是为了**用改过的文字重算网表时**
    #: 能把它原样贴回新电路的 diagnostics —— 否则用户改一次文字，
    #: 「这个结论是在涂掉了哪些区域之后得到的」这条痕迹就悄悄没了。
    text_mask: dict[str, Any] | None = None

    # ------------------------------------------------ 判定

    @property
    def structural_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.structural]

    @property
    def soft_issues(self) -> list[Issue]:
        return [i for i in self.issues if not i.structural]

    @property
    def complete(self) -> bool:
        """本地是否"结构完整"—— 这才是"本地成功"的定义。

        注意不是"能跑出解"：缺一个电阻的数值也照样能建出网表结构，
        那是软问题。这里问的是**拓扑与类型有没有缺口**。
        """
        return bool(self.ran and self.circuit is not None
                    and not self.structural_issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "complete": self.complete,
            "error": self.error,
            "issues": [i.to_dict() for i in self.issues],
            "structural_count": len(self.structural_issues),
            "soft_count": len(self.soft_issues),
            "warnings": list(self.warnings),
            "circuit": self.circuit.to_dict() if self.circuit else None,
            "symbols": self.symbols.to_dict() if self.symbols else None,
            "graph": self.graph.to_dict() if self.graph else None,
            #: 机器**原始**读数 —— 界面校对表的行号以下标这份为准，永远不挪位
            "ocr": self.ocr.to_dict() if self.ocr else None,
            #: **人工校对之后**真正生效的那份。没有人工改动时为 None
            #: （界面上"这一格你是不是改过"就靠它判断，不必自己重算一遍）。
            "ocr_effective": self.ocr_effective.to_dict() if self.ocr_effective else None,
        }

    @property
    def ocr_in_use(self) -> OCR.OcrResult | None:
        """当前**真正生效**的 OCR 结果：有人工修正就用修正后的。"""
        return self.ocr_effective or self.ocr


@dataclass
class VisionOutcome:
    """整条视觉通道的结果。``tier`` 说明**最终采信的是哪一层**。"""

    tier: str = "none"                 # local | vlm | none
    circuit: Circuit | None = None
    issues: list[Issue] = field(default_factory=list)
    local: LocalOutcome = field(default_factory=LocalOutcome)
    #: 调模型的痕迹（模型环境、耗时、原始返回、升级原因）
    vlm: dict[str, Any] | None = None
    #: 为什么升级/为什么不升级 —— 必须是一句人话
    escalation: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.circuit is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "ok": self.ok,
            "escalation": self.escalation,
            "issues": [i.to_dict() for i in self.issues],
            "structural_count": sum(1 for i in self.issues if i.structural),
            "soft_count": sum(1 for i in self.issues if not i.structural),
            "circuit": self.circuit.to_dict() if self.circuit else None,
            "vlm": self.vlm,
            "warnings": list(self.warnings),
            "local": self.local.to_dict(),
        }


# ---------------------------------------------------------------- 本地：图 -> IR


def _node_names_of(graph: WR.WireGraph) -> list[str]:
    return [j.name for j in graph.junctions]


def _pick_ref_node(graph: WR.WireGraph) -> tuple[str, str]:
    """挑一个结点当参考节点（地）。返回 ``(图里的结点名, 为什么是它)``。

    ★ 为什么这一步可以自动做、不需要人工确认：**参考节点的选择不改变任何物理答案**。
    它不是电路的一部分，只是电压的零点；换一个参考点，所有节点电压整体平移一个常数，
    所有支路电流、所有功率一模一样。所以这里不值得花用户一次确认。

    规则：接得最多的那个结点；并列时取最靠下的（教科书画地常在下方），
    再并列取最靠左的。规则本身写进依据，可复核。
    """
    if not graph.junctions:
        return "", "图里一个结点都没有"
    js = list(graph.junctions)
    best = max(js, key=lambda j: (j.degree, j.y, -j.x))
    ties = [j for j in js if (j.degree, j.y, -j.x) == (best.degree, best.y, -best.x)]
    why = (f"按「连接数最多 → 最靠下 → 最靠左」挑出结点 {best.name}"
           f"（连接 {best.degree} 条线，位于 ({best.x:.0f},{best.y:.0f})）"
           f"当地。{'并列时取了最靠下的那个。' if len(ties) > 1 else ''}"
           "参考节点只是电压零点，换它不改变任何电流与功率，所以这一步自动做。")
    return best.name, why


def _map_node_names(graph: WR.WireGraph, ref_name: str) -> dict[str, str]:
    """把图里的结点名映射成 IR 节点名：参考节点恒为 ``"0"``，其余按阅读顺序 1..n。"""
    out: dict[str, str] = {}
    k = 1
    for j in graph.junctions:                      # junctions 已按阅读顺序排好
        if j.name == ref_name:
            out[j.name] = REF_NODE
        else:
            out[j.name] = str(k)
            k += 1
    return out


def _search_radius(slot: SY.SymbolSlot, factor: float) -> float:
    return max(20.0, max(slot.w, slot.h) * factor)


def _owns_refdes(w: OCR.OcrWord, slot: SY.SymbolSlot,
                 slots: "list[SY.SymbolSlot]", kind: str) -> bool:
    """这个位号词**应该归我吗**：在类型匹配的槽位里，我是不是离它最近的。

    ★ 为什么半径一大就非有这条不可：位号搜索半径调到与数值同档（1.6）之后，
    两个相邻元件的搜索圈会重叠。此时如果某个元件**自己没标位号**，
    它就会把旁边元件的位号抢过来 —— 而抢过来之后：
      - 它自己看起来"读到了位号"，不留任何警告；
      - 真正的主人只好退回自动编号；
      - 于是两个元件**静默串号**：报告里 ``R1`` 指的是另一个电阻。
    这比"没读出来"坏得多：没读出来至少会报一条待人工确认。

    判据很简单也很硬：一个 ``R…`` 的词，只属于**离它最近的那个 R 槽位**。
    一个词不可能同时更靠近另一个同类元件 —— 那它就不是我的。
    真实图里位号都贴着自己的本体，所以这条几乎不会误伤；
    而它挡掉的正是"抢邻居"这一类。
    """
    L = kind.upper()
    best_d = float("inf")
    for s in slots:
        if s.kind.upper() != L:
            continue
        d = ((w.cx - s.cx) ** 2 + (w.cy - s.cy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
    my_d = ((w.cx - slot.cx) ** 2 + (w.cy - slot.cy) ** 2) ** 0.5
    # 浮点相等要留容差：同一个位置上不可能有两个槽位，但边界情况别抖
    return my_d <= best_d + 1e-9


def _claim_word(ocr_res: OCR.OcrResult | None, slot: SY.SymbolSlot,
                claimed: set[int], *, factor: float, want: str,
                kind: str, owner_ok: Any = None) -> OCR.OcrWord | None:
    """在槽位附近找一个还没被别人认领的词。

    ``want`` 是 ``"refdes"`` 或 ``"value"``。
    ★ 位号额外要求**首字母与元件类型一致**（R 开头的才认给电阻）——
    光靠距离会把两个相邻元件的位号互相抢走，而那**不会报错**。

    ``owner_ok`` 是额外的把门函数（可选）：返回 False 的候选直接出局。
    位号那一档传 ``_owns_refdes`` 进来，用来挡"抢邻居的位号"。
    """
    if ocr_res is None or not ocr_res.words:
        return None
    r = _search_radius(slot, factor)
    best: OCR.OcrWord | None = None
    best_i = -1
    best_d = float("inf")
    for idx, w in enumerate(ocr_res.words):
        if idx in claimed:
            continue
        d = ((w.cx - slot.cx) ** 2 + (w.cy - slot.cy) ** 2) ** 0.5
        if d > r or d >= best_d:
            continue
        if want == "refdes":
            if not OCR.looks_like_refdes(w.text):
                continue
            if not w.text.strip().upper().startswith(kind.upper()):
                continue
            if owner_ok is not None and not owner_ok(w):
                continue
        else:
            if not OCR.looks_like_value(w.text):
                continue
        best, best_i, best_d = w, idx, d
    if best is not None:
        # ★ 用**下标**认领，不用 ``list.index()`` —— OcrWord 是 dataclass，
        # 它的 ``==`` 按字段比，两个读出来一样的词会撞在一起，
        # 于是会把另一个元件的词误标成"已被认领"。
        claimed.add(best_i)
    return best


def _h_gap(a: OCR.OcrWord, b: OCR.OcrWord) -> float:
    """两个词之间的**水平**缝（左词右边到右词左边）。重叠时为 0。"""
    if b.x >= a.x:
        return max(0.0, b.x - (a.x + a.w))
    return max(0.0, a.x - (b.x + b.w))


def is_refdes_fragment(text: str) -> bool:
    """这个 OCR 词是"被拆散的位号"的一段吗？

    只有两种形状算：**纯字母短段**（位号的头）与**单个数字字符**（数字段）。
    ``R1`` / ``10k`` / ``12V`` 这种自成一体的词都不是 —— 它们不需要拼。
    """
    s = text.strip()
    return bool(_REFDES_HEAD_RE.match(s) or _REFDES_DIGIT_FRAG_RE.match(s))


def _claim_refdes(ocr_res: OCR.OcrResult | None, slot: SY.SymbolSlot,
                  claimed: set[int], *, factor: float, kind: str,
                  slots: "list[SY.SymbolSlot] | None" = None
                  ) -> tuple[str, str]:
    """认领位号，返回 ``(位号或空串, 说明)``。

    两步：

    1. **单个词就是完整位号**（``R1``）—— 位号必须带数字，
       所以"只有字母"的词在这一步一律不算。
    2. 认不到就找"被拆开的位号"：以**字母段**为头，
       向右吃掉同一基线上的**单个数字字符**段（实测 ``R12`` → ``['R','1','2']``）。
       拼出来的东西必须真的长成位号、且首字母与元件类型一致才算数。

    第 2 步是必要的：不做的话那个 ``R`` 会被当成一个合法位号认领下来，
    元件名就变成 ``R`` —— 既不报错也不提示，而人对着教科书上的 ``R12``
    会以为是自己看错了。★ 拼接依据是**形状**而不是缝宽：实测 token 内
    字符缝 0.24~1.44 字高、正常词间空格 0.39~0.80 字高，两者重叠，
    缝宽单独作判据必然误伤（见常量区的注释）。
    """
    if ocr_res is None or not ocr_res.words:
        return "", ""
    r = _search_radius(slot, factor)
    words = ocr_res.words
    # 归属把门：没给 slots 就不启用（单测可以直接只传一个槽位）
    owner = ((lambda w: _owns_refdes(w, slot, slots, kind))
             if slots is not None else None)

    # ---- 第一步：完整位号，就近取
    w = _claim_word(ocr_res, slot, claimed, factor=factor,
                    want="refdes", kind=kind, owner_ok=owner)
    if w is not None:
        return w.text.strip().upper(), f"位号由本地 OCR 读出：{w.raw or w.text}"

    # ---- 第二步：把被拆开的拼回来
    heads: list[tuple[float, str, list[int]]] = []
    for i, h in enumerate(words):
        if i in claimed:
            continue
        if not _REFDES_HEAD_RE.match(h.text.strip()):
            continue
        if not h.text.strip().upper().startswith(kind.upper()):
            continue
        if owner is not None and not owner(h):
            continue                        # 这个字母段不是我的头，别拿它拼
        d = ((h.cx - slot.cx) ** 2 + (h.cy - slot.cy) ** 2) ** 0.5
        if d > r:
            continue

        tol = max(6.0, SPLIT_GAP_MAX_FACTOR * max(1, h.h))
        run = [i]
        cur = i
        for j in range(i + 1, len(words)):
            cand = words[j]
            if j in claimed:
                break
            # 字母段 = 另一个位号的头，到此为止（实测 'R','1','R','2' 是两个位号）
            if _REFDES_HEAD_RE.match(cand.text.strip()):
                break
            if not _REFDES_DIGIT_FRAG_RE.match(cand.text.strip()):
                break
            if abs(cand.cy - h.cy) > max(3.0, 0.6 * h.h):
                break                       # 不在同一基线上
            if _h_gap(words[cur], cand) > tol:
                break
            run.append(j)
            cur = j
            if len(run) - 1 >= MAX_REFDES_DIGIT_FRAGS:
                break
        if len(run) < 2:
            continue
        joined = "".join(words[k].text.strip() for k in run).upper()
        if not OCR.looks_like_refdes(joined):
            continue
        if not joined.startswith(kind.upper()):
            continue
        heads.append((d, joined, run))

    if not heads:
        return "", ""
    heads.sort(key=lambda t: t[0])
    _d, joined, run = heads[0]
    for k in run:
        claimed.add(k)
    parts = "、".join(repr(words[k].text.strip()) for k in run)
    return joined, (f"★ 位号是从图上 {len(run)} 段拼回来的"
                    f"（{parts} → {joined}）—— 实测 Windows OCR 会因字距把一个位号"
                    "读成好几段，拼合依据是形状（字母段 + 单字符数字段）而不是缝宽。"
                    "请核对位号。")


def _find_plus_sign(ocr_res: OCR.OcrResult | None, slot: SY.SymbolSlot
                    ) -> OCR.OcrWord | None:
    """在本体附近找 ``+`` 标记。找不到就返回 None（**绝不猜极性**）。"""
    if ocr_res is None or not ocr_res.words:
        return None
    r = _search_radius(slot, 0.9)
    for w in ocr_res.words:
        if w.text.strip() not in ("+", "＋", "十"):
            continue
        if ((w.cx - slot.cx) ** 2 + (w.cy - slot.cy) ** 2) ** 0.5 <= r:
            return w
    return None


def _value_words_in_same_block(ocr_res: OCR.OcrResult | None,
                               word: OCR.OcrWord) -> list[OCR.OcrWord]:
    """找与 ``word`` 落在**同一个文本块**里的所有"像数值"的词，按 x 排好。

    ★ 为什么需要它：Windows OCR 会把一个数值切成好几段。
    实测 ``12V`` → ``"1"``、``"2"``（那个 V 直接丢了），
    ``10k`` → ``"1"``、``"0"``、``"k"``（字距一大就按字符切）。
    这时"取离元件最近的、像数值的那个词"会拿到 ``"2"`` ——
    于是一个 12V 的电压源被静默地记成 2V。
    这种错误**在三法互校里拦不住**（三条路径共用同一份 IR，会一致地算出错答案），
    所以必须在这一层认出来。

    ★ 为什么**只报不拼**（哪怕拼起来看着挺对）：``['1','0','k']`` 拼出来是
    ``10k``，但真值也可能是 ``1.0k`` —— 小数点被 OCR 吞掉之后，
    这两种情况的词形状**一模一样**。而 ``10k`` 与 ``1.0k`` 差 10 倍，
    这个错和"12V 记成 2V"是同一类：三法互校一致地给错答案。
    所以这里把拼出来的候选**告诉人**，但不替人做决定。
    位号那边可以拼（``R12`` → ``['R','1','2']``），是因为位号的数字位
    按定义不含小数点，拼接不存在这种二义性。
    """
    if ocr_res is None:
        return [word]
    # 先定位这个词属于哪个文本块（取包含它中心点的那个）
    block = None
    for b in ocr_res.blobs:
        if (b.x - 2 <= word.cx <= b.x + b.w + 2
                and b.y - 2 <= word.cy <= b.y + b.h + 2):
            block = b
            break
    if block is None:
        return [word]
    inside = [w for w in ocr_res.words
              if OCR.looks_like_value(w.text)
              and block.x - 2 <= w.cx <= block.x + block.w + 2
              and block.y - 2 <= w.cy <= block.y + block.h + 2]
    inside.sort(key=lambda w: w.x)
    if not inside:
        return [word]

    # ★ 把尾随的**纯单位词**也收进来。实测 ``10k`` 会被切成 ``['1','0','k']`` ——
    #   那个 ``k`` 单独看"不像数值"（没有数字），于是只收 ``['1','0']``，
    #   报出来就变成"拼起来是 '10'（解析为 10）"。这比不报更糟：
    #   人会以为图上真的写了 10。收上 ``k`` 之后是"拼起来是 '10k'（解析为 10000）"，
    #   才是图上真正印着的东西。
    tail = [w for w in ocr_res.words
            if _BARE_UNIT_RE.match(w.text.strip())
            and block.x - 2 <= w.cx <= block.x + block.w + 2
            and block.y - 2 <= w.cy <= block.y + block.h + 2
            and w.x >= inside[-1].x]
    tail.sort(key=lambda w: w.x)
    if tail:
        last = inside[-1]
        nxt = tail[0]
        tol = max(6.0, SPLIT_GAP_MAX_FACTOR * max(1, last.h))
        if (abs(nxt.cy - last.cy) <= max(3.0, 0.6 * max(1, last.h))
                and _h_gap(last, nxt) <= tol):
            inside.append(nxt)
    return inside


def build_local_ir(symbols: SY.SymbolReport, graph: WR.WireGraph,
                   ocr_res: OCR.OcrResult | None, *,
                   name: str = "photo") -> tuple[Circuit | None, list[Issue]]:
    """本地几何 + OCR 的结果 -> IR，并把"哪里没把握"逐条列出来。

    这个函数**不抛异常**：任何一步不够格，就记一条 Issue 并把那个元件跳过，
    让上层按"结构性缺失 → 升级给模型"处理。宁可报缺，不许凑数。
    """
    issues: list[Issue] = []

    if not symbols.slots:
        issues.append(Issue("structural", "no_symbol",
                            "一个元件符号都没定位到。可能图里没有闭合符号图形、"
                            "线宽过细导致形态学运算断线、或对比度太低。"))
        return None, issues
    if not graph.segments:
        issues.append(Issue("structural", "no_wire",
                            "一条导线线段都没抽出来，无法判定连接。"))
        return None, issues

    ref_name, ref_why = _pick_ref_node(graph)
    naming = _map_node_names(graph, ref_name)

    # ---- 结构性缺口：这些一出现就必须升级给模型
    for w in symbols.wire_holes:
        # ★ 这条记 **soft**：正常电路图**全都是闭合回路**，把它记成结构缺失
        #   等于每张图都升级给模型，"本地优先"就架空了。
        #   判据本身有实测余量（回路 0.820/0.881 对图内真符号 0.017/0.015），
        #   而且万一真吞掉了一个大元件，IR 会因不连通而报出来（那条是 structural），
        #   所以这里保持"报出来但不升级"是合理的。
        issues.append(Issue(
            "soft", "wire_loop",
            f"有一处 {w.w}×{w.h} 的空白被一圈导线围住，已按「闭合导线回路」处理、"
            "没当元件。正常电路图都有闭合回路，所以这通常没问题；"
            "但若原图那里其实画了一个很大的元件（本体边长接近 200px 量级），"
            "就会漏 —— 请在叠图上核一眼。"))
    for d in graph.suspect_dots:
        issues.append(Issue(
            "structural", "suspect_dot",
            f"({d['x']:.0f},{d['y']:.0f}) 处比导线粗、但没粗到能算连接圆点。"
            "若原图那里真有圆点，这个交叉其实是相连的。"))
    for u in symbols.unexplained:
        issues.append(Issue(
            "structural", "unexplained_ink",
            f"({u['x']},{u['y']}) 处有 {u['w']}×{u['h']} 的墨迹"
            "既不像导线、也没被任何符号解释掉 —— 可能就是一个没认出来的元件。"))
    n_unassigned = sum(1 for ts in graph.terminals.values()
                       for t in ts if t.node is None)
    if n_unassigned:
        issues.append(Issue(
            "structural", "dangling_terminal",
            f"有 {n_unassigned} 个元件端子附近找不到导线 —— "
            "元件可能悬空，也可能那段导线被本体区域误剃掉了。"))
    if ocr_res is not None and not ocr_res.available:
        # ★ "OCR 不可用"必须再分一层，不能一律当软问题：
        #   环境缺件（语言包 / 运行时没装）是**预期内的正常分支** —— 连接照旧由
        #   几何层定，缺的只是位号数值，人工填一下就行，不值得花一次模型调用。
        #   但"传进来的图片没被接住"是我们的 bug / 调用方误用，它表现为
        #   **整张图的位号数值全空**，如果只记 soft，就会一路静默地给出一个
        #   缺值的网表（实测踩过：管线传 Path，ocr 只认 str）。
        #   所以这一类记 structural，逼着它升级 / 报出来。
        kind = getattr(ocr_res, "reason_kind", "")
        ours = kind in ("image_unreadable", "engine_error")
        issues.append(Issue(
            "structural" if ours else "soft", "ocr_unavailable",
            f"本地 OCR 不可用（{ocr_res.reason}）{ocr_res.hint}"
            + ("　这一类**不是环境问题**，位号与数值会全空，所以按结构缺失处理。"
               if ours else
               "　这一类不影响连接，只缺位号数值，按软问题处理、不升级给模型。")))

    # ---- 逐槽位建元件
    comps: list[Component] = []
    claimed: set[int] = set()
    used_refs: list[str] = []
    n_lowconf = 0
    lowconf_list: list[str] = []

    for si, s in enumerate(symbols.slots):
        kind = s.kind
        if kind not in ALLOWED_KINDS:
            issues.append(Issue(
                "structural", "kind_unknown",
                f"({s.x},{s.y}) 处的符号类型判不出来"
                f"（最高分 {s.template_kind or '?'}，置信度 {s.confidence:.2f}）。"
                "类型不同意味着直流下的行为完全不同（电容开路、电感短路），"
                "不能猜。"))
            continue

        terms = graph.terminals.get(si) or []
        if len(terms) < 2 or any(t.node is None for t in terms):
            issues.append(Issue(
                "structural", "terminal_unbound",
                f"{kind}@({s.cx:.0f},{s.cy:.0f}) 的端子没能全部接到导线结点上"
                f"（{[(t.side, t.node) for t in terms]}）。"))
            continue

        na, nb = naming.get(terms[0].node, ""), naming.get(terms[1].node, "")
        if not na or not nb:
            issues.append(Issue("structural", "node_unnamed",
                                f"{kind}@({s.cx:.0f},{s.cy:.0f}) 的端子结点没有名字。"))
            continue
        if na == nb:
            issues.append(Issue(
                "structural", "self_short",
                f"{kind}@({s.cx:.0f},{s.cy:.0f}) 的两个端子落到了同一个结点 {na} —— "
                "要么本体把自己短接了，要么漏了一处该断开的交叉。"))
            continue

        # 位号
        ref, ref_note = _claim_refdes(ocr_res, s, claimed,
                                      factor=REFDES_SEARCH_FACTOR, kind=kind,
                                      slots=symbols.slots)
        if not ref:
            ref = Circuit.auto_ref(kind, used_refs)
            ref_note = "位号没读出来，已按类型自动编号，请核对"
        if ref in used_refs:
            ref = Circuit.auto_ref(kind, used_refs)
            ref_note = "读出的位号与前面重复，已改自动编号，请核对"
        used_refs.append(ref)

        # 数值
        value: float | None = None
        value_note = ""
        if kind in ("R", "V", "I"):
            wval = _claim_word(ocr_res, s, claimed, factor=VALUE_SEARCH_FACTOR,
                               want="value", kind=kind)
            if wval is not None:
                group = _value_words_in_same_block(ocr_res, wval)
                if len(group) > 1:
                    # ★ 数值被切成了几段 → **不采用**，留空请人工填，并把候选告诉他。
                    # 静默取其中最近的一段是危险的：实测 "12V" 会被切成 "1"、"2"，
                    # 于是 12V 变成 2V，而三法互校根本拦不住这种错。
                    # 拼起来也不采用：['1','0','k'] 拼成 10k，但真值也可能是 1.0k
                    # （小数点被吞掉后形状完全一样），差 10 倍。
                    joined = "".join(w.text.strip() for w in group)
                    cand, _cw = parse_engineering(joined, kind=kind)
                    issues.append(Issue(
                        "soft", "value_split",
                        f"{ref}（{kind}）的数值被本地 OCR 切成了 {len(group)} 段"
                        f"（{'、'.join(repr(w.text) for w in group)}）。"
                        f"这类断词不可靠 —— 实测 `12V` 会被切成 `1` 与 `2`、"
                        f"`10k` 会被切成 `1`、`0`、`k`；按最近距离取就会错成 2V，"
                        f"把几段拼起来又会把 10k 与 1.0k 混掉（小数点被吞了，"
                        f"形状一模一样，差 10 倍）。已留空，请人工确认后填写。"
                        + (f"拼起来的候选是 {joined!r}（解析为 {cand:g}）。"
                           if cand is not None else
                           f"拼起来是 {joined!r}，解析不了。")))
                    value = None
                else:
                    value, vw = parse_engineering(wval.text, kind=kind)
                    value_note = f"数值由本地 OCR 读出：{wval.raw or wval.text}"
                    if vw:
                        value_note += "；" + "；".join(vw)
            if value is None:
                issues.append(Issue(
                    "soft", "value_missing",
                    f"{ref}（{kind}）的数值没能从图上读出来。"
                    "这一类问题**不升级给模型** —— 连接已经确定，"
                    "只缺一个数字，请直接在确认面板里填。"))

        # 极性（只对电压源/电流源）
        conf = s.confidence
        nodes = (na, nb)
        polarity_note = ""
        if kind in ("V", "I"):
            plus = _find_plus_sign(ocr_res, s)
            if plus is not None:
                # 把带 + 的那一侧放到 nodes[0]
                side_plus = "a" if abs(plus.cx - terms[0].x) <= abs(plus.cx - terms[1].x) \
                    else "b"
                if side_plus == "b":
                    nodes = (nb, na)
                polarity_note = (f"在 ({plus.cx:.0f},{plus.cy:.0f}) 读到「+」标记，"
                                 f"已把那一侧作为 nodes[0]")
                conf = max(conf, POLARITY_UNREAD_CONFIDENCE)
            else:
                polarity_note = (
                    "★ 图上没读到「+」标记，极性未能确定。已按几何左右/上下顺序"
                    "排 nodes[0]、nodes[1]，**这个次序就是参考方向**，"
                    "请核对 + 端在哪一侧（电压源）／箭头指向哪边（电流源）。")
                conf = min(conf, POLARITY_UNREAD_CONFIDENCE)

        ev_detail = (f"本地几何层判定。{'；'.join(s.evidence[:2])}。"
                     f"{ref_note}。{value_note}。{polarity_note}")
        comps.append(Component(
            ref=ref, kind=kind, nodes=nodes, value=value,
            evidence=Evidence(source="cv", confidence=min(0.99, conf),
                              detail=ev_detail.strip("；")),
            geom={"box": [s.x, s.y, s.w, s.h], "cx": s.cx, "cy": s.cy,
                  "orientation": s.orientation,
                  "template": s.template_kind,
                  "template_scores": s.template_scores},
            note=polarity_note if "极性未能确定" in polarity_note else "",
        ))
        if s.needs_human and kind in ("R", "C", "L"):
            # ★ 这里记 **soft**，不是 structural —— 这条分类很关键，想清楚再改：
            #   本地这一层**已经把类型做出来了**（kind 不是 "?"），只是要人点一下确认。
            #   它与"缺一个数字"是同一类问题，落在确认面板里，不需要花一次模型调用。
            #   真正属于结构缺失的是下面那条 `kind_unknown`（判不出类型）——
            #   类型决定直流下的行为（电容开路、电感短路），判不出就没法建模。
            #   为什么电阻会经常落在这里：模板判据**刻意不拉伸**（拉了就抹掉长宽比），
            #   而项目自己的 R 模板长宽比是 1.62（渲染器的 30×20 本体），
            #   教科书上的电阻常画成 3:1 —— 比例不同的画法本来就匹配不上。
            n_lowconf += 1
            lowconf_list.append(f"{ref}({kind}, {s.confidence:.2f})")
            issues.append(Issue(
                "soft", "kind_low_confidence",
                f"{ref} 的类型判定置信度只有 {s.confidence:.2f}（低于闸门 "
                f"{CONFIDENCE_GATE}），但类型已经给出（{kind}），不是判不出来。"
                "已按这个类型建网表，请在确认面板里核一眼。"
                + ("（模板判据没投票：它的得分对不上 —— 典型原因是原图的"
                   "符号长宽比与项目模板不同，例如电阻常画成 3:1 而模板是 1.6:1。"
                   "模板判据刻意不拉伸，所以这是预期内的，不是故障。）"
                   if not s.agreed else "")))

    if not comps:
        issues.append(Issue("structural", "no_component",
                            "没能建出任何一个元件（每个候选位都缺类型或缺连接）。"))
        return None, issues

    # ★ 一条闸门：个别元件置信度不足 → 人工点一下就行（soft）；
    # 但**过半**元件都不足 → 这一层的判据整体不适用于这张图（手绘风格、
    # 长宽比与模板差异大、线宽异常……），这时候升级给模型比让人挨个核对划算。
    # 阈值 2 是下限：只有 1 个元件时"过半"没有意义，不该触发整体升级。
    if n_lowconf >= 2 and n_lowconf * 2 >= len(comps):
        issues.append(Issue(
            "structural", "overall_low_confidence",
            f"{n_lowconf}/{len(comps)} 个元件的类型置信度都低于闸门"
            f"（{'、'.join(lowconf_list[:6])}）。个别元件不足只须人工确认，"
            "但过半都不足说明本地这一层的判据整体不适用于这张图的画法，"
            "升级给视觉模型更划算。"))

    circuit = Circuit(name=name, components=comps)
    circuit.diagnostics = [
        {"kind": "ref_node_choice", "text": ref_why},
        {"kind": "node_naming",
         "text": "图里的几何结点 → IR 节点名：" + "、".join(
             f"{k}→{v}" for k, v in naming.items())},
        {"kind": "local_layer",
         "text": f"本地层：定位到 {len(symbols.slots)} 个候选符号位、"
                 f"{len(graph.segments)} 条导线线段、{len(graph.junctions)} 个结点、"
                 f"{len(graph.dots)} 个连接圆点"
                 + (f"；OCR 读出 {len(ocr_res.words)} 个词"
                    if ocr_res else "；OCR 未运行")},
    ]
    if ocr_res is not None and ocr_res.dropped:
        circuit.diagnostics.append({
            "kind": "ocr_dropped",
            "text": f"★ 有 {len(ocr_res.dropped)} 块文字本地 OCR 完全没读出来"
                    "（Windows OCR 对「R1」这种两三个字符的短行会整行丢弃，"
                    "这是实测结论、不是参数没调好）。位置：" +
                    "、".join(f"({b.x},{b.y},{b.w}×{b.h})" for b in ocr_res.dropped[:8]),
        })

    # ---- 拓扑自检：IR 侧必须自己过一遍，别把问题留给求解器
    try:
        for w in circuit.validate(allow_incomplete=True):
            issues.append(Issue("soft", "ir_warning", w))
    except CircuitError as e:
        issues.append(Issue(
            "structural", "ir_invalid",
            f"本地建出的 IR 没通过自检：{e}"))
        return circuit, issues
    return circuit, issues


# ---------------------------------------------------------------- 升级判定


def should_escalate(local: LocalOutcome, cfg: Any) -> tuple[bool, str]:
    """要不要调视觉大模型？返回 ``(要不要, 一句人话的原因)``。

    ★ 分级的落地处：``escalate_on="structural"``（默认）时，
    **只有结构性缺失才升级**。数值读不清、位号读成 RI 这些软问题留在本地处理 ——
    连接已经确定，缺的只是一个数字，花一次模型调用不值得，
    而且数值本来就必须人工核对（三法互校拦不住数值错）。
    """
    mode = getattr(cfg, "mode", "local_first")
    on = getattr(cfg, "escalate_on", "manual")

    if not getattr(cfg, "enabled", True):
        return False, "视觉通道在配置里被关掉了，只用本地结果。"
    if mode == "local_only":
        return False, "配置为 local_only：按要求只跑本地，不调用模型。"
    if mode == "vlm_only":
        return True, "配置为 vlm_only：跳过本地判定，直接整张图交给模型。"
    if on == "manual":
        # ★ 默认档。这里返回 False 不是"放弃升级"，而是"把决定权交出去" ——
        #   上层会把 reason 摆到界面上，由用户点按钮触发。
        #   所以这句话必须**说清本地这一遍到底缺什么**，用户才知道该不该按。
        if not local.ran:
            return False, (f"本地这一遍没跑起来（{local.error or '原因未记录'}）。"
                           "建议交给视觉模型；也可以先按上面的问题改图重试。")
        if local.circuit is None:
            return False, "本地没能建出网表。建议交给视觉模型。"
        ns, nf = len(local.structural_issues), len(local.soft_issues)
        if ns:
            head = "；".join(i.text for i in local.structural_issues[:3])
            return False, (f"本地有 {ns} 处**结构性**缺失（{head}）。"
                           "建议交给视觉模型 —— 这类问题本地做不出来。")
        if nf:
            return False, (f"本地**结构完整**（拓扑与类型都没缺口），"
                           f"只有 {nf} 处软问题（数值/位号读不清）。"
                           "这类问题不用花模型调用：直接在下表把文字改对即可。")
        return False, ("本地这一遍结构完整、文字也都读出来了。"
                       "除非你发现哪里读错了，否则不需要交给模型。")
    if on == "never":
        return False, "配置为 escalate_on=never：本地出什么就用什么，不升级。"

    if not local.ran:
        return True, f"本地这一遍根本没跑起来（{local.error or '原因未记录'}），只能交给模型。"
    if local.circuit is None:
        return True, "本地没能建出网表，属于结构缺失，交给模型。"

    ns = len(local.structural_issues)
    if ns:
        head = "；".join(i.text for i in local.structural_issues[:3])
        return True, (f"本地存在 {ns} 处**结构性**缺失（{head}），"
                      "按「结构缺失才调模型」升级给模型。")
    if on == "any":
        nf = len(local.soft_issues)
        if nf:
            return True, (f"配置为 escalate_on=any：本地还有 {nf} 处软问题"
                          "（数值/文字读不清），一并交给模型复核。")
    return False, ("本地这一遍**结构完整**（拓扑与类型都没有缺口），"
                   f"只有 {len(local.soft_issues)} 处软问题（数值/位号读不清），"
                   "按「结构缺失才调模型」不升级 —— 请人工补齐那几位数值。")


# ---------------------------------------------------------------- 模型：返回 -> IR


def vlm_payload_to_ir(payload: dict[str, Any], *, name: str = "photo",
                      meta: dict[str, Any] | None = None
                      ) -> tuple[Circuit | None, list[Issue]]:
    """把模型返回的 JSON 网表转成 IR。**逐字段校验，绝不整块吞下。**

    ★ 为什么要逐字段校验而不是"信任就全信"：
    "最大信任"指的是**不替它改主意**，不是"不检查格式"。
    模型完全可能返回 ``kind="BAT"``、``nodes`` 有 3 个、或者参考节点写着 ``"GND"`` ——
    这些进不了 IR，硬吞下去的解出来的答案没有任何意义，
    而且报告会显示"算出来了"。所以：能修的格式问题修掉并留痕，
    修不了的整条元件标 ``needs_human``，一条都不静默丢弃。
    """
    issues: list[Issue] = []
    meta = meta or {}

    if not isinstance(payload, dict):
        issues.append(Issue("structural", "vlm_not_dict",
                            "模型返回的不是一个 JSON 对象。"))
        return None, issues
    if payload.get("failed"):
        issues.append(Issue(
            "structural", "vlm_failed",
            f"模型自己报告这张图它读不了：{payload.get('reason') or '未给原因'}"))
        return None, issues

    raw_comps = payload.get("components")
    if not isinstance(raw_comps, list) or not raw_comps:
        issues.append(Issue("structural", "vlm_no_components",
                            "模型返回里没有 components 数组，或者它是空的。"))
        return None, issues

    # 参考节点：契约要求必须是 "0"，但模型未必听话
    ref_node = str(payload.get("ref_node", REF_NODE)).strip() or REF_NODE
    if ref_node != REF_NODE:
        issues.append(Issue(
            "soft", "vlm_ref_node",
            f"模型把参考节点写成 {ref_node!r}，契约要求 {REF_NODE!r}。"
            f"已按契约改名 —— 这件事只影响节点编号，不影响拓扑。"))

    comps: list[Component] = []
    seen_refs: list[str] = []
    for i, raw in enumerate(raw_comps):
        if not isinstance(raw, dict):
            issues.append(Issue("structural", "vlm_bad_item",
                                f"components[{i}] 不是一个对象，已跳过。"))
            continue
        kind = str(raw.get("kind", "")).strip().upper()

        # ---- ★ 受控源：位图通道**认得出来，但不会替它定型控制关系**。
        #   为什么不能顺手建出来：``E``/``G``/``H``/``F`` 是"输出量由另一个
        #   支路的电压/电流决定"的元件，而"控制端接在哪两条节点上""由哪条
        #   支路的电流控制"这两件事，靠图上读出来是**猜**。猜错的后果不是
        #   差一点，而是整题答案换个样 —— 而且三法互校与功率守恒全拦不住：
        #   三条路径共用同一份 IR，会一致地给出同一个错答案。
        #   所以这里按已定的策略处理：**只标记、交人工定型**，
        #   并把"怎么补"写清楚，绝不静默丢弃。
        if kind in CONTROLLED_KINDS:
            issues.append(Issue(
                "structural", "vlm_controlled_source",
                f"components[{i}]（{raw.get('ref') or '未命名'}）是**受控源**："
                f"{CONTROL_NOTE.get(kind, kind)}。"
                "位图通道不替它定型控制关系 —— 控制端接在哪、由哪条支路的电流控制，"
                "从图上读出来只能是猜；猜错会让整题答案换个样，"
                "而三法互校与功率守恒都发现不了（三条路径共用同一份 IR）。"
                "已**保留为一处待人工确认**，未纳入本次建模："
                "请在「手绘」面板里把它画出来并指明控制端（电流控制型还要指出采样支路），"
                "或在「参数」页里补上它的控制关系。"))
            continue

        if kind not in ALLOWED_KINDS:
            issues.append(Issue(
                "structural", "vlm_bad_kind",
                f"components[{i}] 的类型 {raw.get('kind')!r} 不在支持范围 "
                f"{sorted(ALLOWED_KINDS)}（变压器、器件模型等暂不支持），已跳过。"))
            continue
        nodes = raw.get("nodes")
        if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
            issues.append(Issue(
                "structural", "vlm_bad_nodes",
                f"components[{i}] 的 nodes 不是两个端点：{nodes!r}，已跳过。"))
            continue
        n1, n2 = str(nodes[0]).strip(), str(nodes[1]).strip()
        if n1 != REF_NODE and n1 == ref_node:
            n1 = REF_NODE
        if n2 != REF_NODE and n2 == ref_node:
            n2 = REF_NODE
        if not n1 or not n2:
            issues.append(Issue("structural", "vlm_empty_node",
                                f"components[{i}] 的节点名是空的：{nodes!r}，已跳过。"))
            continue
        if n1 == n2:
            issues.append(Issue(
                "structural", "vlm_self_short",
                f"components[{i}] 两端接到同一个节点 {n1}，"
                "模型很可能看错了这一处的连线，已跳过并留痕。"))
            continue

        ref = str(raw.get("ref", "")).strip()
        if not ref:
            ref = Circuit.auto_ref(kind, seen_refs)
            issues.append(Issue("soft", "vlm_no_ref",
                                f"components[{i}] 没给位号，已自动编为 {ref}，请核对。"))
        if ref in seen_refs:
            new = Circuit.auto_ref(kind, seen_refs)
            issues.append(Issue("soft", "vlm_dup_ref",
                                f"位号 {ref} 重复，已把后一个改为 {new}，请核对。"))
            ref = new
        seen_refs.append(ref)

        # 数值：模型可能给 "1k"、"4.7uF"、也可能给 "?" 或不给
        value: float | None = None
        raw_value = raw.get("value")
        if raw_value is not None and str(raw_value).strip() not in ("", "?", "？", "null"):
            value, vw = parse_engineering(raw_value, kind=kind)
            if vw:
                issues.append(Issue("soft", "vlm_value_warn",
                                    f"{ref} 的数值 {raw_value!r}：" + "；".join(vw)))
        if kind in ("R", "V", "I") and value is None:
            issues.append(Issue(
                "soft", "vlm_value_missing",
                f"{ref}（{kind}）：模型也没能给出可用数值，请人工填。"))

        try:
            conf = float(raw.get("confidence", VLM_DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            conf = VLM_DEFAULT_CONFIDENCE
        conf = min(1.0, max(0.0, conf))
        if conf < CONFIDENCE_GATE:
            issues.append(Issue(
                "structural", "vlm_low_confidence",
                f"{ref}：模型自报置信度只有 {conf:.2f}，低于闸门 {CONFIDENCE_GATE}，"
                "这条必须人工核对。"))

        box = raw.get("box")
        geom: dict[str, Any] = {"source": "vlm"}
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            geom["box"] = list(box)[:4]
        comps.append(Component(
            ref=ref, kind=kind, nodes=(n1, n2), value=value,
            evidence=Evidence(
                source="vlm", confidence=conf,
                detail=(f"整张图交给视觉大模型识别，采信其结论。"
                        f"模型自报置信度 {conf:.2f}。"
                        f"{'模型附注：' + str(raw.get('note')) if raw.get('note') else ''}"
                        ).strip()),
            geom=geom,
            note=str(raw.get("note", "") or ""),
        ))

    if not comps:
        issues.append(Issue("structural", "vlm_no_usable",
                            "模型返回的元件没有一条能通过格式校验。"))
        return None, issues

    circuit = Circuit(name=name, components=comps, ref_node=REF_NODE)
    circuit.origin = {
        "channel": "bitmap-vlm",
        "vision_tier": "vlm",
        "vlm": dict(meta),
        "note": ("整张图交给视觉大模型识别，其结论被直接采用（用户口径："
                 "「优先对视觉大模型最大信任」）。source=vlm 已如实标注，"
                 "原始返回已留档，请用叠图核对。"),
    }
    circuit.diagnostics = [
        {"kind": "vlm_payload",
         "text": f"模型返回 {len(raw_comps)} 个元件，其中 {len(comps)} 个通过校验。"},
    ]
    for w in payload.get("warnings") or []:
        circuit.diagnostics.append({"kind": "vlm_warning", "text": str(w)})
    try:
        for w in circuit.validate(allow_incomplete=True):
            issues.append(Issue("soft", "ir_warning", w))
    except CircuitError as e:
        issues.append(Issue(
            "structural", "vlm_ir_invalid",
            f"模型给的网表没通过 IR 自检：{e}"))
    return circuit, issues


# ---------------------------------------------------------------- 总入口


def run_local(image: Any, *, cfg: Any = None,
              name: str = "photo") -> LocalOutcome:
    """本地这一遍：OCR → 涂白文字 → 符号定位 → 导线追踪 → 建 IR。**永不抛异常。**

    ★ 这四步的**顺序不能换**，每一步都有它非在那个位置的道理：

    1. **OCR 必须在涂白之前**：涂白把文字擦掉了，擦完再 OCR 就一个字也读不到。
    2. **涂白必须在符号/导线之前**：文字墨迹与元件墨迹在二值图上没有区别 ——
       一个文字的「0」有自己的空洞，会被判成一个电容；几块字的墨迹会当成
       "元件本体"，它的 slot_region 盖住导线、把导线切碎。
       而真实电路图几乎都有标注，所以这一步不是边角情况。
    3. **导线追踪必须在符号定位之后**：它要用符号的本体区域剔除本体墨迹，
       否则电阻框会把自己的两个端子短接（闭环导体），两个结点被误并。
    """
    cfg = cfg or load_config().config
    out = LocalOutcome()

    # ---- 1. OCR（不依赖符号/导线，先跑）
    # 确认过口径：本地成功门槛是"结构缺失才调模型"，
    # 而 OCR 只影响位号/数值（软问题），所以它不可用**不该**阻止本地这一遍。
    try:
        ocr_res = OCR.recognize(image, langs=list(cfg.ocr.langs)
                                if getattr(cfg, "ocr", None) else None,
                                correct=True, detect_blobs=True)
    except Exception as e:                        # noqa: BLE001
        ocr_res = None
        out.warnings.append(f"OCR 抛异常（按「没有 OCR」继续）：{type(e).__name__}: {e}")
    out.ocr = ocr_res
    # ocr_effective 刻意留在 None：它表示"有人工改动"，机器这一遍没有人改。
    # 要用"当前生效的那份"请走 out.ocr_in_use。
    if ocr_res is not None:
        out.warnings.extend(ocr_res.warnings)

    # ---- 2. 涂白文字区域（几何不变，只为把文字墨迹从图上拿掉）
    work = image
    text_mask_report: dict[str, Any] | None = None
    blobs = list(ocr_res.blobs) if ocr_res is not None else []
    if getattr(cfg, "blank_text", True) and blobs:
        try:
            work, text_mask_report = OCR.mask_text_regions(image, blobs)
            out.warnings.append(
                "已把文字区域涂白再交给几何层：" + text_mask_report["note"])
        except Exception as e:                    # noqa: BLE001
            work = image
            out.warnings.append(
                f"涂白文字时出错，已退回用原图（文字可能被当成元件）："
                f"{type(e).__name__}: {e}")

    # ---- 3. 符号定位
    try:
        rep = SY.detect_symbols(work)
    except Exception as e:                        # noqa: BLE001
        out.error = f"符号定位抛异常：{type(e).__name__}: {e}"
        out.warnings.append(out.error)
        return out
    out.symbols = rep
    out.warnings.extend(rep.warnings)

    # ---- 4. 导线追踪
    try:
        graph = WR.build_wire_graph(work, rep)
    except Exception as e:                        # noqa: BLE001
        out.error = f"导线追踪抛异常：{type(e).__name__}: {e}"
        out.warnings.append(out.error)
        return out
    out.graph = graph
    out.warnings.extend(graph.warnings)

    circuit, issues = build_local_ir(rep, graph, ocr_res, name=name)
    out.text_mask = text_mask_report
    if circuit is not None and text_mask_report is not None:
        circuit.diagnostics.append({
            "kind": "text_blanked", **text_mask_report})
    out.circuit = circuit
    out.issues = issues
    out.ran = True
    return out


def rebuild_local_ir(local: "LocalOutcome", ocr_res: OCR.OcrResult | None, *,
                     name: str = "photo") -> tuple[Circuit | None, list[Issue]]:
    """用**改过文字**的 OCR 结果重算本地网表。几何（符号、导线）完全不动。

    ★ 这是"先审后算"的关键：用户在校对表里把 ``RI`` 改成 ``R1``、把没读出来的块补上，
    改的只是**文字**，而拓扑是几何层算出来的、与文字无关。
    所以这一步是纯重放：拿同一份 ``symbols`` 与 ``graph``，换一份 ``ocr_res``，
    重新跑一遍位号/数值的认领 —— 毫秒级，可以随便点。

    重算的好处是**没有任何增量状态**：改错了再改回去，结果一定与最初一致，
    不需要撤销栈，也不会出现"改了三次之后和改一次不一样"这种鬼故事。
    """
    if not local.ran:
        return None, list(local.issues)
    if local.symbols is None or local.graph is None:
        return None, list(local.issues)
    circuit, issues = build_local_ir(local.symbols, local.graph, ocr_res, name=name)
    if circuit is not None and local.text_mask is not None:
        circuit.diagnostics.append({"kind": "text_blanked", **local.text_mask})
    return circuit, issues


def _svg_esc(s: Any) -> str:
    """XML 转义。只这几个字符需要 —— 但一个都不能漏，否则整张叠图变空白。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


#: 叠图配色。叠图是画在**原图（多半是白底照片）**之上的，
#: 所以这里取的是深色高饱和、在白底上都能看清的一组；
#: 不跟随 IDE 主题 —— 它盖住的是用户的照片，不是界面。
_OV_COLORS = {
    "segment": "#e8552d",     # 抽出来的导线线段
    "junction": "#15803d",    # 结点（端点/T接/圆点合并）
    "dot": "#1d4ed8",         # 检出的连接圆点
    "suspect": "#b45309",     # 可疑圆点：比线粗但不够圆点，**必须人看一眼**
    "symbol": "#7c3aed",      # 符号本体框
    "terminal": "#0891b2",    # 元件端子
    "word": "#0f766e",        # OCR 读出来的词
    "dropped": "#dc2626",     # 有字但没读出来
    "ink": "#dc2626",         # 没被解释掉的墨迹
    "loop": "#6b7280",        # 被判为"闭合导线回路"的空白
}


def overlay_payload(outcome: Any, image_size: tuple[int, int] | None = None,
                    *, source_url: str | None = None) -> dict[str, Any]:
    """把视觉层的**每一个判断**画回原图坐标上，供人逐项核对。

    ★ 为什么必须单独写一份（不能复用 ``ir.render.render_overlay_payload``）：
    那个是照 **IR 的节点坐标** 画的，而位图通道的 IR 里**没有节点坐标** ——
    连接是几何层从像素里算出来的，节点名只是给结点编的号。
    所以它对着位图 IR 只会回一句"没有元件带视觉坐标，无法按原图坐标回绘"。
    位图这一路的核对对象本来就是**几何量**：抽出来的线段、判定的结点、
    检出的圆点、符号本体框、OCR 读到的词、以及**没读出来的文字块**
    （``dropped``）和**没被解释掉的墨迹**（``unexplained``）。
    这几样画出来，人才能一眼看出"它是不是把某个交叉当成不相连了"。

    载荷形状与 ``render_overlay_payload`` 对齐（``svg`` / ``image_size`` /
    ``viewbox`` / ``source_url``），前端同一段叠图代码就能吃两种通道的结果。

    ``outcome`` 可以是 ``VisionOutcome``（取它的本地那一遍）或 ``LocalOutcome``。
    """
    loc = getattr(outcome, "local", outcome) or None
    if loc is None:
        return {"error": "没有本地这一遍的结论可画。"}
    sym, graph = loc.symbols, loc.graph
    # ★ 画**当前生效**的那份文字（人工校对过的），不是机器原始读数：
    #   叠图的意义是"核对最终真正被用上的是什么"。
    #   人改过之后还画旧的那份，等于让人对着一张自己已经修过的图反复怀疑。
    ocr_res = loc.ocr_in_use if hasattr(loc, "ocr_in_use") else loc.ocr

    if image_size is None and ocr_res is not None:
        image_size = tuple(ocr_res.image_size) or None       # type: ignore[assignment]
    if (not image_size or image_size == (0, 0)):
        # 兜底：用所有几何量的外接框推一个画布（不该走到这里，但宁可画歪也别不画）
        xs: list[float] = []
        ys: list[float] = []
        if sym:
            xs += [v for s in sym.slots for v in (s.x, s.x + s.w)]
            ys += [v for s in sym.slots for v in (s.y, s.y + s.h)]
        if graph:
            xs += [v for s in graph.segments for v in (s.x1, s.x2)]
            ys += [v for s in graph.segments for v in (s.y1, s.y2)]
        if not xs:
            return {"error": "既没有图片尺寸、也没有任何几何量，画不出叠图。"}
        image_size = (int(max(xs) + 10), int(max(ys) + 10))
    W, H = int(image_size[0]), int(image_size[1])

    # 字号随图缩放：600px 的图用 12px，2000px 的图用 40px
    fs = max(11, int(round(min(W, H) / 50)))
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" class="ca-overlay-vision">')
    parts.append(
        '<g fill="none" stroke-linecap="round">')
    parts.append(
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="none" '
        f'stroke="{_OV_COLORS["loop"]}" stroke-dasharray="4 4" opacity="0.35"/>')
    parts.append('</g>')

    # ---- 符号本体框 + 类型/置信度/位号
    ref_of: dict[tuple, str] = {}
    if loc.circuit is not None:
        for c in loc.circuit.components:
            b = (c.geom or {}).get("box")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                ref_of[tuple(int(round(v)) for v in b)] = f"{c.ref}({c.kind})"
    parts.append('<g class="ca-ov-symbols">')
    if sym:
        for s in sym.slots:
            col = _OV_COLORS["symbol"]
            parts.append(
                f'<rect x="{s.x}" y="{s.y}" width="{s.w}" height="{s.h}" '
                f'fill="none" stroke="{col}" stroke-width="2"/>')
            key = (s.x, s.y, s.w, s.h)
            tag = ref_of.get(key) or f'{s.kind}?'
            txt = f"{tag} {s.confidence:.2f}"
            if s.template_kind:
                txt += f"[{s.template_kind}]"
            ty = s.y - 3 if s.y > fs + 4 else s.y + s.h + fs
            parts.append(
                f'<text x="{s.x}" y="{ty}" font-size="{fs}" '
                f'font-family="monospace" fill="{col}" stroke="none" '
                f'font-weight="bold">{_svg_esc(txt)}</text>')
    parts.append('</g>')

    # ---- 线段 + 结点 + 圆点
    parts.append('<g class="ca-ov-wires">')
    if graph:
        for s in graph.segments:
            parts.append(
                f'<line x1="{s.x1:.1f}" y1="{s.y1:.1f}" x2="{s.x2:.1f}" '
                f'y2="{s.y2:.1f}" stroke="{_OV_COLORS["segment"]}" '
                f'stroke-width="2" opacity="0.85"/>')
        for d in graph.dots:
            parts.append(
                f'<circle cx="{d["x"]:.1f}" cy="{d["y"]:.1f}" '
                f'r="{max(3.0, float(d.get("radius", 4))):.1f}" '
                f'fill="{_OV_COLORS["dot"]}" opacity="0.55"/>')
        for d in graph.suspect_dots:
            parts.append(
                f'<circle cx="{d["x"]:.1f}" cy="{d["y"]:.1f}" r="7" '
                f'fill="none" stroke="{_OV_COLORS["suspect"]}" '
                f'stroke-width="2" stroke-dasharray="3 3"/>')
        for j in graph.junctions:
            parts.append(
                f'<circle cx="{j.x:.1f}" cy="{j.y:.1f}" r="4.5" '
                f'fill="#ffffff" stroke="{_OV_COLORS["junction"]}" stroke-width="2.5"/>')
            parts.append(
                f'<text x="{j.x + 7:.1f}" y="{j.y - 6:.1f}" font-size="{fs}" '
                f'font-family="monospace" fill="{_OV_COLORS["junction"]}" '
                f'stroke="none" font-weight="bold">{_svg_esc(j.name)}</text>')
        for ts in graph.terminals.values():
            for t in ts:
                parts.append(
                    f'<circle cx="{t.x:.1f}" cy="{t.y:.1f}" r="2.5" '
                    f'fill="{_OV_COLORS["terminal"]}"/>')
    parts.append('</g>')

    # ---- 被判为"闭合导线回路"的空白（软问题，但要看一眼）
    parts.append('<g class="ca-ov-loops">')
    if sym:
        for w in sym.wire_holes:
            parts.append(
                f'<rect x="{w.x}" y="{w.y}" width="{w.w}" height="{w.h}" '
                f'fill="none" stroke="{_OV_COLORS["loop"]}" stroke-width="1.5" '
                f'stroke-dasharray="6 4" opacity="0.8"/>')
        for u in sym.unexplained:
            parts.append(
                f'<rect x="{u["x"]}" y="{u["y"]}" width="{u["w"]}" '
                f'height="{u["h"]}" fill="none" stroke="{_OV_COLORS["ink"]}" '
                f'stroke-width="2"/>')
            parts.append(
                f'<text x="{u["x"]}" y="{u["y"] - 3}" font-size="{fs}" '
                f'font-family="monospace" fill="{_OV_COLORS["ink"]}" '
                f'stroke="none">未解释的墨迹</text>')
    parts.append('</g>')

    # ---- OCR：读出来的词 / ★ 有字却没读出来的块
    parts.append('<g class="ca-ov-text">')
    if ocr_res is not None:
        for w in ocr_res.words:
            # 修正过的词把原文一并标出（R1 ← RI），否则"它读对了"这件事没法核
            lab = w.text if w.raw in ("", w.text) else f"{w.text}←{w.raw}"
            parts.append(
                f'<rect x="{w.x}" y="{w.y}" width="{w.w}" height="{w.h}" '
                f'fill="none" stroke="{_OV_COLORS["word"]}" '
                f'stroke-width="1.5" opacity="0.9"/>')
            parts.append(
                f'<text x="{w.x}" y="{w.y - 3}" font-size="{fs}" '
                f'font-family="monospace" fill="{_OV_COLORS["word"]}" '
                f'stroke="none">{_svg_esc(lab)}</text>')
        for b in ocr_res.dropped:
            parts.append(
                f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" '
                f'fill="none" stroke="{_OV_COLORS["dropped"]}" '
                f'stroke-width="2" stroke-dasharray="4 3"/>')
            parts.append(
                f'<text x="{b.x}" y="{b.y - 3}" font-size="{fs}" '
                f'font-family="monospace" fill="{_OV_COLORS["dropped"]}" '
                f'stroke="none" font-weight="bold">有字没读出来</text>')
    parts.append('</g>')
    parts.append('</svg>')

    return {
        "svg": "\n".join(parts),
        "image_size": {"w": W, "h": H},
        "viewbox": [0, 0, W, H],
        "mode": "vision",
        "kind": "raster",
        "source_url": source_url,
        "legend": {
            "segment": "抽出来的导线线段",
            "junction": "结点（端点/T接/圆点合并），旁边是 IR 节点名",
            "dot": "检出的连接圆点（交叉处有它就是相连）",
            "suspect": "可疑圆点 —— 比线粗但不够圆点，按不相连处理了，请看一眼",
            "symbol": "符号本体框（标注：位号(类型) 置信度[模板判据]）",
            "terminal": "元件端子",
            "word": "OCR 读到的词（修正过的写 '改后←原文'）",
            "dropped": "★ 几何上检出文字、OCR 却没读出来",
            "ink": "没被任何符号解释掉的墨迹（可能是个没认出来的元件）",
            "loop": "被判为「闭合导线回路」的空白",
        },
    }


def run(image: Any, *, cfg: Any = None, name: str = "photo",
        client: Any = None, escalate: bool | None = None) -> VisionOutcome:
    """整条两级通道。**永不抛异常**，失败也返回带原因的结构化结果。

    ``client`` 只为测试注入（默认用 ``vlm.call_vision``）。
    它是同步阻塞的：在 async 端点里请用线程池调用本函数。

    ``escalate`` 是**显式覆盖**升级闸门：``True`` 表示"不管配置怎么说，
    这次就是要交给模型"（界面上的「交给视觉模型识别」按钮走这条），
    ``False`` 表示"这次坚决不叫模型"。默认 ``None`` = 按配置的 ``escalate_on`` 判定。
    """
    from . import vlm as VLM

    cfg = cfg or load_config().config
    outcome = VisionOutcome()

    # ---- 第一级：本地
    if getattr(cfg, "mode", "local_first") != "vlm_only":
        outcome.local = run_local(image, cfg=cfg, name=name)
        outcome.warnings.extend(outcome.local.warnings)
    else:
        outcome.local = LocalOutcome(ran=False, error="配置为 vlm_only，跳过本地")
        # vlm_only 也要留一份本地的符号/导线结果：报告里要能对照
        try:
            outcome.local.symbols = SY.detect_symbols(image)
            outcome.local.graph = WR.build_wire_graph(image, outcome.local.symbols)
        except Exception as e:                    # noqa: BLE001
            outcome.local.warnings.append(
                f"vlm_only 模式下仍尝试跑本地定位以供对照，但它失败了："
                f"{type(e).__name__}: {e}")

    do_vlm, why = should_escalate(outcome.local, cfg)
    # 显式覆盖：界面上的按钮说了算，配置只当默认值。
    # 覆盖时要把理由改掉 —— 否则报告里会写"按配置不升级"却升级了，自相矛盾。
    if escalate is True and not do_vlm:
        do_vlm, why = True, "你手动要求把这张图交给视觉模型（覆盖了配置里的升级条件）。"
    elif escalate is False:
        do_vlm, why = False, "你手动选择了只用本地结果，这次不叫模型。"
    outcome.escalation = why

    # ---- 本地就能给结论
    if not do_vlm:
        outcome.tier = "local" if outcome.local.circuit is not None else "none"
        outcome.circuit = outcome.local.circuit
        outcome.issues = list(outcome.local.issues)
        if outcome.circuit is not None:
            outcome.circuit.origin = {
                "channel": "bitmap-local",
                "vision_tier": "local",
                "escalation": why,
                "note": ("本地几何层给的结论：连接由「线段端点 + 圆点圆心」算出，"
                         "类型由「空洞 + 模板」双判据给出。"
                         "所有 source=cv 的判断都可以在叠图上核对并改写。"),
            }
        return outcome

    # ---- 第二级：视觉大模型
    _vlm_into(outcome, image, cfg=cfg, name=name, why=why, client=client)
    return outcome


def _vlm_into(outcome: "VisionOutcome", image: Any, *, cfg: Any, name: str,
              why: str, client: Any = None) -> None:
    """第二级：把整张图交给视觉模型，结果**就地**写进 ``outcome``。

    抽成独立函数是为了让**手动升级**走同一条代码路径：用户点按钮时本地那一遍
    早就跑完了，既不该重跑本地、更不该把模型调用再实现一份（两份实现早晚会分叉，
    而分叉的那一天表现是"按钮给的结论和自动给的结论不一样"）。

    **永不抛异常**：模型调不通也把失败写进 ``outcome``，并退回本地结论。
    """
    from . import vlm as VLM

    call = client or VLM.call_vision

    def _fall_back(code: str, text: str, vlm_info: dict[str, Any]) -> None:
        """模型这条路走不通：退回本地结论 + 记一条结构性问题。"""
        outcome.tier = "local" if outcome.local.circuit is not None else "none"
        outcome.circuit = outcome.local.circuit
        outcome.issues = list(outcome.local.issues)
        outcome.issues.append(Issue("structural", code, text))
        outcome.vlm = vlm_info
        outcome.warnings.append(text)
        # ★ escalation 必须跟着改口。这句是"为什么升级"的解释，会被写进
        #   origin 与报告里；调用明明失败了却还写着"已把整张图交给视觉模型、
        #   采信其结论"，那就是**报告在说谎** —— 用户会以为模型看过这张图。
        outcome.escalation = (outcome.escalation or "").rstrip() + \
            f"（★ 但这次调用没成功：{text}）"

    try:
        res = call(image, cfg=cfg)
    except VLM.VlmNotConfigured as e:
        _fall_back("vlm_not_configured",
                   f"需要调模型但没配好：{e}。{e.hint}",
                   {"ok": False, "kind": e.kind, "message": str(e),
                    "hint": e.hint, "raw": e.raw})
        return
    except VLM.VlmError as e:
        _fall_back("vlm_failed",
                   f"模型调用失败（{e.kind}）：{e}。{e.hint}",
                   {"ok": False, "kind": e.kind, "message": str(e),
                    "hint": e.hint, "raw": e.raw})
        return
    except Exception as e:                        # noqa: BLE001
        _fall_back("vlm_crashed",
                   f"调用模型时出了没预料到的错：{type(e).__name__}: {e}",
                   {"ok": False, "kind": "crash",
                    "message": f"{type(e).__name__}: {e}", "hint": "", "raw": ""})
        return

    meta = {
        "model": res.model, "url": res.url,
        "elapsed_s": round(res.elapsed, 2),
        "finish_reason": res.finish_reason,
        "usage": res.usage,
        "image": res.image.summary() if res.image else None,
        "escalated_because": why,
        "raw_text": res.text,
    }
    outcome.vlm = {"ok": True, **meta, "payload": res.payload}
    circuit, issues = vlm_payload_to_ir(res.payload or {}, name=name, meta=meta)
    outcome.circuit = circuit
    outcome.issues = list(issues)
    outcome.warnings.extend(res.warnings)
    outcome.tier = "vlm" if circuit is not None else "none"

    # ★ 最大信任 ≠ 免检：本地那一遍的结果始终留在 outcome.local 里，
    #   报告与 WebUI 要把两层并排呈现，让用户能一眼看出模型改了什么。
    if circuit is not None and outcome.local.circuit is not None:
        a = {c.ref for c in circuit.components}
        b = {c.ref for c in outcome.local.circuit.components}
        if a != b:
            outcome.warnings.append(
                "★ 模型结论与本地结论不一致，已**按口径采信模型**："
                f"模型给出 {sorted(a)}，本地给出 {sorted(b)}。"
                "两层的结果都留在报告里，请用叠图核对这一处差异。")


def escalate_now(outcome: "VisionOutcome", image: Any, *, cfg: Any = None,
                 name: str = "photo", client: Any = None) -> VisionOutcome:
    """界面上的「交给视觉模型识别」按钮走这条。

    与 ``run(..., escalate=True)`` 的区别：本地那一遍**不重跑** ——
    它已经跑完了，而且用户很可能刚在界面上手工改过文字，
    重跑会把他改的东西冲掉。这里就地补上第二级。
    """
    cfg = cfg or load_config().config
    why = "你手动要求把这张图交给视觉模型（覆盖了配置里的升级条件）。"
    outcome.escalation = why
    _vlm_into(outcome, image, cfg=cfg, name=name, why=why, client=client)
    return outcome


def probe(cfg: Any = None) -> dict[str, Any]:
    """给 ``/api/health`` 用：报告这条通道**当前**能不能用、走哪一路。

    **不发起任何网络请求，也不返回密钥明文。**
    """
    from . import vlm as VLM

    loaded = load_config() if cfg is None else None
    cfg = cfg or loaded.config
    problems = cfg.validate()
    return {
        "enabled": cfg.enabled,
        "mode": cfg.mode,
        "escalate_on": cfg.escalate_on,
        "trust_vlm": cfg.trust_vlm,
        "local": {
            "symbols": SY.probe(),
            "wires": WR.probe(),
            "ocr": OCR.probe(),
        },
        "vlm": VLM.probe(cfg),
        "will_call_vlm": bool(cfg.enabled and cfg.mode in ("local_first", "vlm_only")
                              and cfg.escalate_on != "never"),
        "problems": problems,
        "config_sources": loaded.sources if loaded else {},
        "hint": ("启用模型需要在 config/secrets.local.json 里填 vision.api_key，"
                 "或设环境变量 CIRCUIT_AGENT_VLM_API_KEY。"
                 "没配也能跑 —— 本地那一层会照常工作，只是缺结构时无法升级。"),
    }
