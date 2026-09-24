# -*- coding: utf-8 -*-
"""视觉通道自检：本地两级 + 升级闸门 + 模型返回转 IR。

分七组：

  A. **本地通道端到端** —— 合成一张有标注的串联回路，要求本地出网表，
     且拓扑正确（不是"能跑"而是"接法对"）。
  B. **两条铁律** —— 十字交叉无圆点不相连 / 有圆点就相连 / T 接相连。
  C. **OCR 修正表** —— 实测出来的那几类混淆必须被修回来。
  D. **洞指纹与朝向无关** —— 圆内竖直径线与横直径线都给"两个洞 → 电压源"。
  E. **分级升级闸门** —— 结构完整就不调模型；结构缺失才调；三种档位都要对。
  F. **模型返回转 IR** —— 假 HTTP 端点跑真请求，逐字段校验按契约拦住坏数据。
  G. **没配 key 时的行为** —— 不抛异常、给一句能照做的建议、且**绝不泄漏明文 key**。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_vision.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont                       # noqa: E402

from app.ir.model import CONFIDENCE_GATE                          # noqa: E402
from app.vision import ocr as OCR                                 # noqa: E402
from app.vision import pipeline as PL                             # noqa: E402
from app.vision import symbols as SY                              # noqa: E402
from app.vision import wires as WR                                # noqa: E402
from app.vision.config import VisionConfig                        # noqa: E402
from app.vision.vlm import (VlmAuthError, VlmBadResponse,         # noqa: E402
                            VlmNotConfigured, VlmModelRejectsImage,
                            call_vision)

FAILS: list[str] = []
SKIPS: list[str] = []


def ck(name: str, cond: bool, extra: str = "") -> bool:
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (("   -> " + str(extra)) if extra else ""))
    if not cond:
        FAILS.append(name)
    return bool(cond)


def head(t: str) -> None:
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def _font(size: int = 26):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:                          # noqa: BLE001
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------- 合成图

def sch_loop(*, r_body=(60, 40), labels=True, extra_solid=False,
             label="R1 10k", font_size=20):
    """一张闭合串联回路：上边 R、左边 V、其余是导线。

    正确的电气拓扑只有 **2 个结点**（回路里只有两个电位）：
    R1 与 V1 都接在同一对结点上（电路上叫并联，画成回路的样子）。
    ``r_body`` 是电阻**本体**尺寸 —— 项目自己的模板比例是 1.5:1。

    ★ ``label`` / ``font_size`` 这两个参数是**实测**定下来的，不是随手挑的。
    同一张图上，标注的读法随字号剧烈变化，而且**换一个版面就换一套结果**：

    ==============================  ========================================
    这张图（620×480，字号 20）          实测读法
    ==============================  ========================================
    20 px（本测试用的档）            ``['R1','10k']`` —— 两段都干净，数值 = 10000
    26 px                            **一个字都没读出来**（整块被丢掉）
    22 / 30 / 34 px                  ``['R1','1','0k']`` —— 数值被切成三段
    ==============================  ========================================

    换一张 900×160 的图再量，同一批字号又是另一个分布（20~34 全都干净、
    40 反而全丢）—— 见 ``tests/_probe_ocr_lanelen.py`` 第①组。
    所以**不要指望"某个字号一定稳"**。真正稳的只有两条：

    1. **行越长越稳**：``"R1 10k R2 20k"`` 在 18~56px 每一档都读得干干净净，
       而单独一个 ``"R1"`` 在每一档都被整条丢掉 —— 位号必须和数值写在**同一行**。
    2. 短文本行被整条跳过是**无解**的已知行为（见 ``ocr.recognize`` 里的
       ``dropped``），所以本组的数值断言在那种情况下退化成"必须**报出来**"，
       而不是硬判失败 —— 见 ``sec_a``。
    """
    im = Image.new("RGB", (620, 480), "white")
    d = ImageDraw.Draw(im)
    rw, rh = r_body
    cx, cy = 300, 120
    d.rectangle([cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2],
                outline="black", width=4)
    d.line([120, 120, cx - rw // 2, 120], fill="black", width=4)
    d.line([cx + rw // 2, 120, 480, 120], fill="black", width=4)
    d.line([480, 120, 480, 360], fill="black", width=4)
    d.line([480, 360, 120, 360], fill="black", width=4)
    d.ellipse([95, 215, 145, 265], outline="black", width=4)
    d.line([120, 120, 120, 215], fill="black", width=4)
    d.line([120, 265, 120, 360], fill="black", width=4)
    if labels:
        f = _font(font_size)
        d.text((200, 145), label, fill="black", font=f)
    if extra_solid:
        # 一块涂黑的墨迹（水印/阴影）—— 本地层必须把它报成"未解释"，这就是一处结构缺失
        d.ellipse([520, 400, 600, 470], fill="black")
    return im


def _cfg(**kw) -> VisionConfig:
    c = VisionConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --------------------------------------------------------------- A 本地端到端

def sec_a() -> None:
    head("A. 本地通道端到端：合成回路图 -> 网表，要求拓扑正确")
    im = sch_loop()
    out = PL.run(im, cfg=_cfg(mode="local_only"), name="A")
    ck("本地能建出网表", out.ok, out.local.error)
    if not out.circuit:
        return
    c = out.circuit
    print("      " + "  ".join(f"{x.ref}{x.kind}{x.nodes}"
                              f"(v={x.value})" for x in c.components))
    print(f"      节点 {c.nodes}")
    ck("结点数正好 2（回路里只有两个电位，碎段没被当成结点）",
       len(c.nodes) == 2, c.nodes)
    refs = {x.ref: x for x in c.components}
    ck("定位到 R1 与 V1", {"R1", "V1"} <= set(refs), sorted(refs))
    if {"R1", "V1"} <= set(refs):
        r, v = refs["R1"], refs["V1"]
        ck("★ R1 两端不是同一个结点（否则等于把电阻短路）",
           r.nodes[0] != r.nodes[1], r.nodes)
        ck("★ V1 两端不是同一个结点", v.nodes[0] != v.nodes[1], v.nodes)
        ck("★ R1 与 V1 接在**同一对**结点上（回路的正确接法）",
           set(r.nodes) == set(v.nodes), (r.nodes, v.nodes))
    ck("类型置信度达到闸门（R 的洞法 + 模板法互相印证）",
       refs["R1"].evidence.confidence >= CONFIDENCE_GATE,
       refs["R1"].evidence.confidence)
    ck("Evidence 如实写明来源是几何层", refs["R1"].evidence.source == "cv",
       refs["R1"].evidence.source)

    # ---- 位号与数值：一位一位地判，别把"OCR 自己丢了"记成"我们读错了"
    ocr_res = out.local.ocr
    ck("★ 位号是完整位号，不是被拆散后剩下的字母段（如 'R'）",
       refs["R1"].ref == "R1", refs["R1"].ref)
    got_words = [w.text for w in (ocr_res.words if ocr_res else [])]
    if ocr_res is not None and ocr_res.dropped:
        # 实测：Windows OCR 对短文本行会**整条跳过**（同一张图换字号就从
        # "读得出来"变成"一个字都没有"）。这不是本项目的 bug，
        # 而是必须被**报出来**的已知行为 —— 所以这里查"有没有报出来"。
        SKIPS.append("A 组数值：这一档字号上 Windows OCR 把整块标注丢了（实测行为）")
        ck("★ 标注被 OCR 整条丢掉时，必须有一条可见的警告（不许静默）",
           any("有字但没读出来" in w for w in out.local.warnings),
           [w[:36] for w in out.local.warnings])
        ck("丢掉的那一处必须作为 Issue 报出来，供人工填",
           any(i.code == "value_missing" for i in out.local.issues),
           [i.code for i in out.local.issues])
    else:
        ck("数值从图上读出来了（10k -> 10000）", refs["R1"].value == 10000.0,
           (refs["R1"].value, got_words))
    diag = " ".join(d.get("text", "") for d in c.diagnostics)
    ck("诊断里记下了参考节点的挑选理由", "参考节点" in diag or "当地" in diag)
    ck("诊断里记下了文字涂白（不静默改动）", "text_blanked" in
       " ".join(d.get("kind", "") for d in c.diagnostics))


# --------------------------------------------------------------- B 铁律

def sec_b() -> None:
    head("B. 两条铁律：交叉的连通性由圆点决定")
    base = Image.new("RGB", (500, 500), "white")
    d = ImageDraw.Draw(base)
    d.line([60, 250, 440, 250], fill="black", width=4)
    d.line([250, 60, 250, 440], fill="black", width=4)

    g1 = WR.build_wire_graph(base, SY.detect_symbols(base, correct=False))
    ck("十字交叉且**无**圆点 → 两个独立结点（不相连）",
       len(g1.junctions) == 2, len(g1.junctions))
    ck("没有把交叉处的方块误当圆点", len(g1.dots) == 0, g1.dots)

    d.ellipse([244, 244, 256, 256], fill="black")
    g2 = WR.build_wire_graph(base, SY.detect_symbols(base, correct=False))
    ck("同一交叉**加上**圆点 → 并成 1 个结点（相连）",
       len(g2.junctions) == 1, len(g2.junctions))
    ck("检出圆点", len(g2.dots) >= 1, g2.dots)

    tee = Image.new("RGB", (500, 400), "white")
    d3 = ImageDraw.Draw(tee)
    d3.line([60, 150, 440, 150], fill="black", width=4)
    d3.line([250, 150, 250, 330], fill="black", width=4)
    g3 = WR.build_wire_graph(tee, SY.detect_symbols(tee, correct=False))
    ck("T 形接入且无圆点 → 相连（教科书惯例）",
       len(g3.junctions) == 1, len(g3.junctions))

    # 可疑圆点只在**内部交叉**处才值得上报
    ck("拐角处的粗块没有被报成可疑圆点",
       not any(abs(x["x"] - 60) < 20 or abs(x["x"] - 440) < 20
               for x in g1.suspect_dots), g1.suspect_dots)


# --------------------------------------------------------------- C OCR 表

def sec_c() -> None:
    head("C. OCR 修正表与拆词拼合：实测出来的混淆必须修回来")
    # 这一张表里的每一种形状都是**量出来的**，不是编的：
    #   1↔I、0↔O 来自对合成标注的实测；小数点丢失、rn↔m 来自对真实教材图的实测。
    # ★ 注意不要往里塞"一个词里带空格"的用例 —— Windows OCR 的词是**按空格切的**，
    #   一个词里不会带空格。实测它把位号拆开的形态是**拆成好几个词**：
    #   'R12' → ['R','1','2']、'10k' → ['1','0','k']（见本组下半段）。
    cases = [
        ("RI", "refdes", "R1"),        # 1 被读成 I
        ("Rl", "refdes", "R1"),        # 1 被读成 l
        ("CI", "refdes", "C1"),
        ("Ok", "value", "0k"),         # 0 被读成 O
        ("O.1uF", "value", "0.1uF"),
        ("4 7k", "value", "4.7k"),     # 小数点消失成空格
        ("2 2rnH", "value", "2.2mH"),  # m 被切成 rn
        ("3 3V", "value", "3.3V"),
    ]
    for raw, kind, want in cases:
        got, fixes = OCR.correct_token(raw, kind=kind)
        ck(f"修正 {raw!r} → {want!r}", got == want, f"{got!r} {fixes}")
    t, _ = OCR.correct_token("4 7k", kind="value")
    v, _ = __import__("app.ingest.values", fromlist=["x"]).parse_engineering(t)
    ck("修完能解析成 4700", v == 4700.0, (t, v))
    ck("像位号 / 像数值两个判据各就各位",
       OCR.looks_like_refdes("R12") and OCR.looks_like_value("4.7k")
       and not OCR.looks_like_refdes("4.7k"))
    ck("★ 只有字母的短串**不算**位号（它是被拆散的位号的头，"
       "认下来会把元件名变成 R）",
       not OCR.looks_like_refdes("R") and not OCR.looks_like_refdes("RI"))


def _words(items):
    """按 ``[(文本, x, y)]`` 造一批 OCR 词（宽高按字符数估，够用就行）。"""
    out = []
    for k, (txt, x, y) in enumerate(items):
        out.append(OCR.OcrWord(text=txt, x=x, y=y, w=14 * max(1, len(txt)), h=20,
                               raw=txt, line=0))
    return out


def _blob_over(words, *, pad=6):
    """造一个盖住这批词的中心点的文本块（找同块词的逻辑要用它）。"""
    x0 = min(w.x for w in words) - pad
    y0 = min(w.y for w in words) - pad
    x1 = max(w.x + w.w for w in words) + pad
    y1 = max(w.y + w.h for w in words) + pad
    return OCR.TextBlob(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _slot(cx, cy, *, kind="R", w=60, h=40):
    """造一个够用的候选元件位（只有搜索半径用得到它的中心与尺寸）。"""
    return SY.SymbolSlot(x=cx - w // 2, y=cy - h // 2, w=w, h=h, kind=kind,
                         orientation="h", confidence=0.95, holes=[],
                         template_kind=kind, template_scores={}, agreed=True)


def sec_c2() -> None:
    head("C2. 位号被拆成好几段 → 必须拼回来；数值被拆 → 必须**不**拼")
    # ---- 位号：实测 R12 → ['R','1','2']
    ws = _words([("R", 330, 100), ("1", 348, 100), ("2", 364, 100)])
    res = OCR.OcrResult(available=True, words=ws, blobs=[_blob_over(ws)])
    ref, note = PL._claim_refdes(res, _slot(337, 110), set(), factor=1.1,
                                 kind="R")
    ck("★ 'R','1','2' 三段拼回 'R12'（不拼就只剩一个字母 R，且不报错）",
       ref == "R12", (ref, note))
    ck("拼回来的动作有留痕（不静默重建）", "拼回来" in note, note[:44])

    # ---- 位号：正常标注 'R1' + '10k' 两个词，绝不许拼成 'R110k'
    ws2 = _words([("R1", 330, 100), ("10k", 372, 100)])
    res2 = OCR.OcrResult(available=True, words=ws2, blobs=[_blob_over(ws2)])
    ref2, _n2 = PL._claim_refdes(res2, _slot(340, 110), set(), factor=1.1,
                                 kind="R")
    ck("★ 正常标注 'R1 10k' 不许被拼成 'R110k'（那会把位号弄错）",
       ref2 == "R1", ref2)

    # ---- 位号：两个被拆开的位号相邻 → 各吃各的，不许跨过字母段
    ws3 = _words([("R", 300, 100), ("1", 316, 100), ("R", 356, 100),
                  ("2", 372, 100)])
    res3 = OCR.OcrResult(available=True, words=ws3, blobs=[_blob_over(ws3)])
    ref3, _n3 = PL._claim_refdes(res3, _slot(307, 110), set(), factor=1.1,
                                kind="R")
    ck("★ 'R','1','R','2' 里的第一个只吃自己那一截（字母段 = 另一个位号的头）",
       ref3 == "R1", ref3)

    # ---- 位号：首字母与类型不符的字母段不许被吃（电感旁边那个 k 之类）
    ws4 = _words([("k", 330, 100), ("1", 348, 100)])
    res4 = OCR.OcrResult(available=True, words=ws4, blobs=[_blob_over(ws4)])
    ref4, _n4 = PL._claim_refdes(res4, _slot(337, 110), set(), factor=1.1,
                                 kind="R")
    ck("首字母与元件类型不符 → 不认（k 不能当电阻的位号头）", ref4 == "", ref4)

    # ---- 数值：同样的形状**不许**拼
    ws5 = _words([("1", 330, 100), ("0", 346, 100), ("k", 362, 100)])
    res5 = OCR.OcrResult(available=True, words=ws5, blobs=[_blob_over(ws5)])
    grp = PL._value_words_in_same_block(res5, res5.words[0])
    ck("被切开的数值整组报出来交给人工（'1','0','k' 三段都在）",
       [w.text for w in grp] == ["1", "0", "k"], [w.text for w in grp])
    ck("★ 但**不**替人拼成 10k —— 真值可能是 1.0k，小数点被吞后形状一样，差 10 倍",
       PL.is_refdes_fragment("1") and PL.is_refdes_fragment("k")
       and not PL.is_refdes_fragment("10k")
       and not PL.is_refdes_fragment("R1"), "位号可拼、数值只报不拼")


# ------------------------------------------------- C3 位号归属（不许抢邻居）

def sec_c3() -> None:
    head("C3. 位号归属：搜索半径放大后，不许去抢旁边元件的位号")
    # ★ 这一节的由来（实测缺陷）：位号的搜索半径曾经是 1.1、数值是 1.6，
    #   而两者在图上是**并排写的同一组标注**。实测（元件框 57px 宽）：
    #   文字离框 20~70px 时，数值认得到、**位号一个都认不到**。
    #   后果是界面自相矛盾 —— 用户在文字校对表里把 RI 改成 R1，
    #   网表却仍写"位号没读出来，已按类型自动编号"（还可能碰巧同名，
    #   看起来就是"改了没反应"）。位号本来是用户最想手工纠正的一格。
    #   修法：两者同档；半径一大就必须补上"归属"判据挡住抢邻居。
    ck("位号的搜索半径与数值同档（并排写的两个标签没有理由差一档）",
       PL.REFDES_SEARCH_FACTOR == PL.VALUE_SEARCH_FACTOR,
       (PL.REFDES_SEARCH_FACTOR, PL.VALUE_SEARCH_FACTOR))

    # ---- 场景：两个并排电阻（槽位中心相距 240），只有**左边**标了 R1。
    #     右边那个自己的位号没读出来，绝不许把左边的 R1 抢过来 ——
    #     抢走之后它看起来"读到了"，不留任何警告，而两个元件就静默串号了。
    left = _slot(300, 120, kind="R", w=60, h=40)
    right = _slot(540, 120, kind="R", w=60, h=40)
    slots = [left, right]
    ws = _words([("R1", 240, 100)])          # 只这一个位号词，离左边近
    res = OCR.OcrResult(available=True, words=ws, blobs=[_blob_over(ws)])

    ref_l, _nl = PL._claim_refdes(res, left, set(), factor=1.6, kind="R",
                                 slots=slots)
    ck("近的那个认到了自己的位号 R1", ref_l == "R1", ref_l)

    claimed = {0}
    ref_r, note_r = PL._claim_refdes(res, right, claimed, factor=1.6, kind="R",
                                     slots=slots)
    ck("★ 远的那个**抢不走**别人的位号（抢走就是静默串号，比没读出来坏得多）",
       ref_r == "", (ref_r, note_r))

    # ---- 反过来：两个都标了，各自就近认领，不许互相串
    ws2 = _words([("R1", 240, 100), ("R2", 480, 100)])
    res2 = OCR.OcrResult(available=True, words=ws2, blobs=[_blob_over(ws2)])
    a, _ = PL._claim_refdes(res2, left, set(), factor=1.6, kind="R", slots=slots)
    b, _ = PL._claim_refdes(res2, right, {0}, factor=1.6, kind="R", slots=slots)
    ck("两个都标了：左边拿 R1、右边拿 R2（各就各的近）",
       (a, b) == ("R1", "R2"), (a, b))

    # ---- 归属判据按**类型**分家：一个 R 的词不该被电压源据为己有
    v_slot = _slot(300, 120, kind="V", w=60, h=40)
    ws3 = _words([("R1", 240, 100)])
    res3 = OCR.OcrResult(available=True, words=ws3, blobs=[_blob_over(ws3)])
    gr, _ = PL._claim_refdes(res3, v_slot, set(), factor=1.6, kind="V",
                             slots=[v_slot])
    ck("电压源不许认领 R 开头的位号（首字母必须与类型一致）", gr == "", gr)


# --------------------------------------------------------------- D 洞指纹

def sec_d() -> None:
    head("D. 洞指纹与朝向无关：圆内直径线是**两个洞**")
    for horizontal in (False, True):
        im = Image.new("L", (160, 160), 255)
        d = ImageDraw.Draw(im)
        d.ellipse([20, 20, 140, 140], outline=0, width=4)
        if horizontal:
            d.line([24, 80, 136, 80], fill=0, width=3)
        else:
            d.line([80, 24, 80, 136], fill=0, width=3)
        holes = [h for h in SY.find_holes(SY.ink_mask(im)) if h.area >= 30]
        tag = "横" if horizontal else "竖"
        ck(f"圆 + {tag}直径线 → 2 个洞", len(holes) == 2, len(holes))
        pairs = SY.group_holes(holes)
        ck(f"圆 + {tag}直径线 → 归成 1 组（= 一个电压源）",
           len(pairs) == 1, len(pairs))


# --------------------------------------------------------------- E 升级闸门

def sec_e() -> None:
    head("E. 分级升级闸门：结构完整不调模型，结构缺失才调")

    local_clean = PL.run(sch_loop(), cfg=_cfg(mode="local_only"))
    ck("干净的图：本地结构完整", local_clean.local.complete,
       [i.code for i in local_clean.local.structural_issues])

    # 用假客户端记录"到底有没有被调用"
    called: list[int] = []

    def fake_client(image, *, cfg=None, **kw):
        called.append(1)
        raise AssertionError("这一档位不该调用模型")

    for mode, label, expect_call in [
        ("local_only", "local_only：一律不调", False),
        ("local_first", "local_first + 结构完整：不调", False),
    ]:
        called.clear()
        out = PL.run(sch_loop(), cfg=_cfg(mode=mode), client=fake_client)
        ck(label, (len(called) > 0) == expect_call,
           f"调用次数={len(called)} 判定={out.escalation[:40]}")

    # 结构缺失：图里多一块涂黑的墨迹 → 本地必须承认"没解释掉"，这时才升级
    dirty = sch_loop(extra_solid=True)
    d_local = PL.run(dirty, cfg=_cfg(mode="local_only"))
    ck("带异物墨迹的图：本地报出结构性缺失",
       bool(d_local.local.structural_issues),
       [i.code for i in d_local.local.structural_issues])

    called.clear()

    def fake_client2(image, *, cfg=None, **kw):
        called.append(1)
        raise VlmNotConfigured("测试用：故意不配")

    out = PL.run(dirty, cfg=_cfg(mode="local_first", escalate_on="structural"),
                 client=fake_client2)
    ck("local_first + 结构缺失：**确实调了**模型", len(called) == 1, len(called))
    ck("调不通时仍然返回本地那份网表（不把本地结果丢掉）",
       out.circuit is not None and out.tier == "local", out.tier)

    # ---- ★ 默认档位是 manual：结构缺失也**不自动**调模型
    #   用户拍板的口径是"由用户手动选择是否提交 LLM"。默认自动升级会在
    #   用户还没看到本地结论时就把钱花掉，而且他连"本地到底读成了什么"都不知道。
    ck("★ 默认 escalate_on 就是 manual（导入时不自动调模型）",
       VisionConfig().escalate_on == "manual", VisionConfig().escalate_on)
    called.clear()
    out_m = PL.run(dirty, cfg=_cfg(mode="local_first"), client=fake_client)
    ck("★ manual + 结构缺失：**不自动调**（等用户按按钮）", len(called) == 0,
       len(called))
    ck("★ 但要把『该不该交给模型』的理由说清楚给人看（不能只是一句『不升级』）",
       "建议交给视觉模型" in out_m.escalation, out_m.escalation[:70])
    ck("结构缺失时本地那份网表照样给出（人可以自己先改）",
       out_m.local.circuit is None or out_m.circuit is not None, out_m.tier)

    # ---- ★ 手动要求：显式覆盖配置，必须真的调
    called.clear()
    out_force = PL.run(dirty, cfg=_cfg(mode="local_first"), client=fake_client2,
                       escalate=True)
    ck("★ escalate=True 覆盖配置：手动要求就必须调", len(called) == 1,
       len(called))
    ck("覆盖时理由要改口（否则报告里会写『按配置不升级』却升级了，自相矛盾）",
       "手动" in out_force.escalation, out_force.escalation[:70])

    # ---- ★ 手动拒绝：escalate=False 时任何档位都不许调
    called.clear()
    PL.run(dirty, cfg=_cfg(mode="local_first", escalate_on="any"),
           client=fake_client, escalate=False)
    ck("★ escalate=False：用户明确说不用模型，那就一次都不调", len(called) == 0,
       len(called))

    # ---- ★ 手动升级不重跑本地：用户可能刚在校对表里改过文字，
    #      重跑 OCR+几何会把他的修改冲掉。
    called.clear()
    out_local = PL.run(sch_loop(), cfg=_cfg(mode="local_first"))
    n_before = len(out_local.local.ocr.words) if out_local.local.ocr else 0
    PL.escalate_now(out_local, sch_loop(), cfg=_cfg(mode="local_first"),
                    client=fake_client2)
    n_after = len(out_local.local.ocr.words) if out_local.local.ocr else 0
    ck("★ 手动升级复用本地那一遍（不重跑 OCR：词表对象没被换掉）",
       len(called) == 1 and n_after == n_before, (len(called), n_before, n_after))
    ck("手动升级后 escalation 说明了是人工触发的",
       "手动" in out_local.escalation, out_local.escalation[:60])

    # ---- ★ 调不通时，escalation 必须跟着改口（报告不许说"采信了模型结论"）
    ck("★ 模型没调通时 escalation 里点明了失败（否则报告在说谎）",
       "没成功" in out_local.escalation, out_local.escalation[-60:])

    # escalate_on=never
    called.clear()
    out = PL.run(dirty, cfg=_cfg(mode="local_first", escalate_on="never"),
                 client=fake_client)
    ck("escalate_on=never：即便结构缺失也不调", len(called) == 0,
       out.escalation[:50])

    # escalate_on=any：软问题也升级
    called.clear()
    soft = sch_loop()
    PL.run(soft, cfg=_cfg(mode="local_first", escalate_on="any"),
           client=fake_client2)
    ck("escalate_on=any：本地有软问题也调（哪怕结构完整）",
       len(called) == 1, len(called))

    # vlm_only：跳过本地判定，直接调
    called.clear()
    out = PL.run(soft, cfg=_cfg(mode="vlm_only"), client=fake_client2)
    ck("vlm_only：直接调模型", len(called) == 1, len(called))
    ck("vlm_only 下仍留一份本地定位结果供对照",
       out.local.symbols is not None)


# --------------------------------------------------------------- F 模型返回

class _FakeVLM(BaseHTTPRequestHandler):
    """一个说 OpenAI 协议的假端点。记录收到什么，回什么由 handler 决定。

    ``payload`` 是**模型内容**（会被包进合法的 chat/completions 外壳）；
    ``raw_body`` 是"整个响应体由我说了算"的口子 —— 要测"响应本身不合规"
    这一路错误分类，只能从这里下手（只改 payload 是测不出来的：
    外壳永远是合规的）。
    """

    payload: dict = {}
    seen: list = []
    status = 200
    raw_body: bytes | None = None

    def do_POST(self):                             # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).seen.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "body": body,
        })
        if type(self).raw_body is not None:
            raw = type(self).raw_body
        else:
            content = json.dumps(type(self).payload, ensure_ascii=False)
            raw = json.dumps({
                "model": "deepseek-v4-flash-vision-exp",
                "choices": [{"message": {"content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):                     # noqa: D102
        pass


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


GOOD_PAYLOAD = {
    "schema": "circuit_agent.netlist/1",
    "ref_node": "0",
    "failed": False,
    "components": [
        {"ref": "R1", "kind": "R", "nodes": ["0", "1"], "value": "1k",
         "box": [0.1, 0.2, 0.3, 0.1], "confidence": 0.95},
        {"ref": "V1", "kind": "V", "nodes": ["1", "0"], "value": "12V",
         "box": [0.5, 0.2, 0.1, 0.3], "confidence": 0.9, "note": "长线在左"},
    ],
    "warnings": ["图上有一处字迹不清"],
}


def sec_e2() -> None:
    head("E2. 送进模型前的那一步：三种图片来源都要收（PIL 对象是真踩过的坑）")
    from app.vision.vlm import VlmBadResponse, prepare_image

    im = sch_loop()
    p_img = prepare_image(im)
    ck("收 PIL.Image（管线里流下来的就是这个类型）",
       p_img.data[:4] in (b"\x89PNG", b"\xff\xd8\xff"), p_img.mime)

    tmp = Path(tempfile.mkdtemp()) / "_vision_prep_probe.png"
    im.save(tmp)
    p_path = prepare_image(tmp)
    ck("收路径", p_path.data[:4] in (b"\x89PNG", b"\xff\xd8\xff"), p_path.mime)
    p_bytes = prepare_image(tmp.read_bytes())
    ck("收 bytes", p_bytes.data[:4] in (b"\x89PNG", b"\xff\xd8\xff"), p_bytes.mime)
    ck("三种来源出的是同一张图（尺寸一致）",
       (p_img.width, p_img.height) == (p_path.width, p_path.height)
       == (p_bytes.width, p_bytes.height),
       (p_img.width, p_img.height))

    # ★ 类型不认识时，错误信息必须说"类型不对"，不许说成"这张图读不开"
    try:
        prepare_image(12345)
        ck("不认识的来源类型 → 报错", False, "静默通过了")
    except VlmBadResponse as e:
        ck("★ 不认识的来源类型要指名道姓，不能甩锅给「文件坏了」",
           "int" in str(e) and "读不开" not in str(e), str(e)[:44])
    try:
        prepare_image(ROOT / "runs" / "_definitely_not_here.png")
        ck("文件不存在 → 报错", False, "静默通过了")
    except VlmBadResponse as e:
        ck("文件不存在 → 报错并说清是哪个文件", "读不开" in str(e), str(e)[:44])


def sec_f() -> None:
    head("F. 模型返回 -> IR：请求契约 + 逐字段校验")
    srv, url = _serve()
    try:
        cfg = _cfg(mode="vlm_only", base_url=url, api_key="sk-test-abcdef123456",
                   model="deepseek-v4-flash-vision-exp")

        # --- 请求契约
        _FakeVLM.seen.clear()
        _FakeVLM.payload = GOOD_PAYLOAD
        _FakeVLM.status = 200
        res = call_vision(sch_loop(), cfg=cfg)
        sent = _FakeVLM.seen[-1]
        ck("请求打到 /chat/completions", sent["path"].endswith("/chat/completions"),
           sent["path"])
        ck("带上了 Bearer 鉴权头", sent["auth"].startswith("Bearer sk-test-"),
           sent["auth"][:14])
        ck("送的是配置里的模型名", sent["body"]["model"] == cfg.model)
        ck("温度压到 0（照抄任务不是创作任务）", sent["body"]["temperature"] == 0.0)
        ck("没开流式", sent["body"]["stream"] is False)
        ck("没发 response_format（并非所有模型都支持，不支持会报 400）",
           "response_format" not in sent["body"])
        roles = [m["role"] for m in sent["body"]["messages"]]
        ck("消息是 system + user", roles == ["system", "user"], roles)
        img_parts = [c for m in sent["body"]["messages"] for c in
                     (m["content"] if isinstance(m["content"], list) else [])]
        ck("图片放在 **user** 消息里（放 system/assistant 会被拒）",
           any(p.get("type") == "image_url" for p in img_parts))
        ck("图片用 base64 data URL",
           any(str(p.get("image_url", {}).get("url", "")).startswith(
               "data:image/") for p in img_parts))
        # ★ 提示词分两半：**行为约束**在 system，**字段契约**在 user。
        #   查契约要去 user 消息里查 —— 一开始写成查 system，结果是绿的假象。
        sysp = sent["body"]["messages"][0]["content"]
        usrp = [c.get("text", "") for m in sent["body"]["messages"]
                if m["role"] == "user" and isinstance(m["content"], list)
                for c in m["content"] if isinstance(c, dict) and c.get("type") == "text"]
        ck("system 提示词写明「只读不补全」（不许模型自作主张补元件）",
           "补全" in sysp and "看不清" in sysp, sysp[:24] + "…")
        ck("user 提示词里写明了 5 种元件（契约字段表在这里）",
           all(k in (usrp[0] if usrp else "") for k in ("R / V / I / C / L",)),
           (usrp[0] if usrp else "")[:40])
        ck("user 提示词写死了电压源 nodes[0] = 标 + 的那一端",
           "nodes[0]" in (usrp[0] if usrp else "")
           and "+" in (usrp[0] if usrp else ""))
        ck("user 提示词写死了参考地必须叫 0",
           'reference_node": "0"' in (usrp[0] if usrp else ""))
        ck("user 提示词要求受控源等写进 unsupported，不许塞进 components",
           "unsupported" in (usrp[0] if usrp else "")
           and "受控源" in (usrp[0] if usrp else ""))

        # --- 返回 -> IR
        out = PL.run(sch_loop(), cfg=cfg)
        ck("采信模型的结论（tier=vlm）", out.tier == "vlm", out.tier)
        ck("两个元件都进来了", len(out.circuit.components) == 2,
           [c.ref for c in out.circuit.components])
        r1 = out.circuit.components[0]
        ck("★ source 如实标为 vlm（最大信任 ≠ 隐去来源）",
           r1.evidence.source == "vlm", r1.evidence.source)
        ck("数值按工程记法解析（1k -> 1000）", r1.value == 1000.0, r1.value)
        ck("模型自报置信度被采用", abs(r1.evidence.confidence - 0.95) < 1e-9,
           r1.evidence.confidence)
        ck("原始返回被留档（可复核）",
           (out.vlm or {}).get("raw_text", "").find("circuit_agent.netlist") >= 0)
        ck("origin 里注明了这是整张图交给模型的结论",
           out.circuit.origin.get("vision_tier") == "vlm")

        # --- 逐字段校验：坏数据必须被拦住并留痕
        bad_cases = [
            ({"components": [{"ref": "X1", "kind": "BAT", "nodes": ["0", "1"]}]},
             "不支持的元件类型被拦住", "vlm_bad_kind"),
            ({"components": [{"ref": "X1", "kind": "R", "nodes": ["0", "1", "2"]}]},
             "三端元件被拦住（本项目只支持二端）", "vlm_bad_nodes"),
            ({"components": [{"ref": "X1", "kind": "R", "nodes": ["0", "0"]}]},
             "两端同结点被拦住（等于短路）", "vlm_self_short"),
            ({"components": [{"ref": "R1", "kind": "R", "nodes": ["0", "1"]},
                             {"ref": "R1", "kind": "R", "nodes": ["1", "2"]}]},
             "重复位号被改掉并留痕", "vlm_dup_ref"),
            ({"failed": True, "reason": "不像电路图"}, "模型自报读不了 -> 结构化失败",
             "vlm_failed"),
        ]
        for payload, label, code in bad_cases:
            _, issues = PL.vlm_payload_to_ir(payload)
            ck(label, any(i.code == code for i in issues),
               [i.code for i in issues])

        # --- 自报低置信度必须照实压下来
        low = {"components": [{"ref": "R1", "kind": "R", "nodes": ["0", "1"],
                               "value": "1k", "confidence": 0.4}]}
        _, issues = PL.vlm_payload_to_ir(low)
        ck("模型自报低置信度 -> 记为需人工",
           any(i.code == "vlm_low_confidence" for i in issues),
           [i.code for i in issues])

        # --- 与本地结论不一致时必须报出来（两层都留在报告里）
        # ★ 必须用一张**会升级**的图：干净图上本地结构完整、按口径根本不调模型，
        #   那样测的就不是"不一致告警"，而是"没调模型"。
        differ = {"components": [
            {"ref": "R9", "kind": "R", "nodes": ["0", "1"], "value": "2k",
             "confidence": 0.95}]}
        _FakeVLM.payload = differ
        out2 = PL.run(sch_loop(extra_solid=True),
                      cfg=_cfg(mode="local_first",
                               escalate_on="structural",
                               base_url=url,
                               api_key="sk-test-abcdef123456"))
        ck("模型与本地结论不同 -> 报出差异（已按口径采信模型）",
           any("不一致" in w for w in out2.warnings),
           [w[:40] for w in out2.warnings])

        # --- 错误分类
        _FakeVLM.status = 401
        _FakeVLM.payload = {"error": {"message": "bad key"}}
        try:
            call_vision(sch_loop(), cfg=cfg)
            ck("401 判为鉴权错误", False, "没抛异常")
        except VlmAuthError as e:
            ck("401 判为鉴权错误（并给出怎么改）", bool(e.hint), e.hint[:60])

        _FakeVLM.status = 400
        _FakeVLM.payload = {"error": {"message": "This model does not support image"}}
        try:
            call_vision(sch_loop(), cfg=cfg)
            ck("模型不收图 -> 专门的异常类型", False, "没抛异常")
        except VlmModelRejectsImage as e:
            ck("模型不收图 -> 专门的异常类型（不是笼统的 400）",
               bool(e.hint), e.hint[:60])
        except Exception as e:                     # noqa: BLE001
            ck("模型不收图 -> 专门的异常类型", False, type(e).__name__)

        _FakeVLM.status = 200
        _FakeVLM.payload = GOOD_PAYLOAD
        # ★ 要测"响应体本身就不对"，必须让假服务端能发一个**坏包** ——
        #   只改 payload 是没用的：外壳始终被包成合法的 chat/completions 响应，
        #   那样测的只是"模型内容里没有 components"，不是"响应不合规"。
        _FakeVLM.raw_body = b"<html><body>502 Bad Gateway</body></html>"
        try:
            call_vision(sch_loop(), cfg=cfg)
            ck("响应不是 JSON -> 报 bad_response", False, "没抛异常")
        except VlmBadResponse as e:
            ck("响应不是 JSON -> 报 bad_response（不静默当成空网表）",
               bool(e.hint), type(e).__name__)
        except Exception as e:                     # noqa: BLE001
            ck("响应不是 JSON -> 报 bad_response", False, type(e).__name__)

        # 合法 JSON、但没有 choices（有些网关故障时会这样）
        _FakeVLM.raw_body = json.dumps({"error": "boom"}).encode()
        try:
            call_vision(sch_loop(), cfg=cfg)
            ck("响应没有 choices -> 报 bad_response", False, "没抛异常")
        except VlmBadResponse as e:
            ck("响应没有 choices -> 报 bad_response（不静默当成空网表）",
               "choices" in str(e), str(e)[:40])
        except Exception as e:                     # noqa: BLE001
            ck("响应没有 choices -> 报 bad_response", False, type(e).__name__)
        _FakeVLM.raw_body = None
    finally:
        srv.shutdown()


# --------------------------------------------------------------- G 没配 key

def sec_g() -> None:
    head("G. 没配 key 时的行为：不抛异常、给可照做的建议、不泄漏明文")
    secret = "sk-SHOULD-NEVER-APPEAR-1234567890"
    # escalate_on 必须显式给 structural：默认是 manual（等用户按按钮），
    # 这一节要测的是"真的想调模型但没配 key"这条路。
    cfg = _cfg(mode="local_first", escalate_on="structural", api_key="")
    out = PL.run(sch_loop(extra_solid=True), cfg=cfg)   # 有结构缺失 -> 想调模型
    ck("没配 key 时不抛异常（返回结构化结果）", isinstance(out.to_dict(), dict))
    ck("本地那份网表照样返回", out.circuit is not None, out.tier)
    txt = json.dumps(out.to_dict(), ensure_ascii=False)
    ck("给出一句能照做的建议（提到 api_key 与存放位置）",
       "api_key" in txt and ("secrets.local.json" in txt
                             or "CIRCUIT_AGENT_VLM_API_KEY" in txt))
    ck("明确说了本地 OCR 层不需要 key", "不需要 key" in txt or "本地" in txt)

    cfg2 = _cfg(api_key=secret)
    ck("掩码不含明文", secret not in cfg2.mask() and secret not in
       json.dumps(cfg2.to_dict(mask=True), ensure_ascii=False))
    ck("短 key 直接全遮（掩码本身也不泄漏内容）",
       _cfg(api_key="sk-short").mask() == "*" * 8,
       _cfg(api_key="sk-short").mask())

    pl = PL.probe(cfg2)
    s = json.dumps(pl, ensure_ascii=False)
    ck("probe 不发起网络请求也不返回明文 key", secret not in s)
    ck("probe 报出这一层每一块的可用状态",
       all(k in pl for k in ("enabled", "mode", "escalate_on", "local", "vlm",
                             "will_call_vlm")))
    print("      " + json.dumps({k: pl[k] for k in ("mode", "escalate_on",
                                                    "will_call_vlm")},
                                 ensure_ascii=False))


def main() -> int:
    sec_a()
    sec_b()
    sec_c()
    sec_c2()
    sec_c3()
    sec_d()
    sec_e()
    sec_e2()
    sec_f()
    sec_g()
    print("\n" + "=" * 76)
    if SKIPS:
        print("跳过项：" + "、".join(SKIPS))
    print(f"总计失败项：{len(FAILS)}")
    for f in FAILS:
        print("  -", f)
    print("=" * 76)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
