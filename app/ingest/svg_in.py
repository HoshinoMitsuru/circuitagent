"""SVG 通道：直接解析 XML 拿**精确几何**，零猜测。

这是四条通道里精度最高的一条 —— SVG 是矢量格式，端点坐标是**确切的数字**，
不存在位图那种"这根线到哪儿为止"的模糊。

本通道做三件事：
1. 解析 ``line`` / ``polyline`` / ``polygon`` / ``path`` / ``rect`` / ``circle`` /
   ``ellipse`` / ``text``，并**正确应用 ``transform``**（matrix/translate/scale/rotate）。
   transform 不处理干净会得到一套整体错位但看起来"像那么回事"的几何，
   比完全解析失败更危险。
2. 读取**语义标记**（``data-ca-*``）：本工具回绘的 SVG 自带这些标记，
   于是"导出 SVG -> 重新导入"可以无损还原拓扑与两端次序。
3. 没有标记时退化为几何推断（见 topology.py），置信度如实下调。

★ 一条重要的实践结论：**SVG 里往往有真文字**（``<text>``），
所以元件数值可以直接读出来，不需要视觉模型。
这是 SVG 通道相比位图通道的额外优势 —— 位图那一路才真的需要 VLM 或人工。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from ..ir.model import CircuitError
from .topology import DEFAULT_MAX_GAP, Geometry, build_topology
from .values import parse_engineering, parse_resistor

#: 2x3 仿射矩阵，按 SVG 约定 [a c e; b d f]
Matrix = tuple[float, float, float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_mul(m1: Matrix, m2: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def apply_mat(m: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(s: str) -> Matrix:
    """解析 SVG 的 transform 属性（多个变换按出现次序左乘）。"""
    m = IDENTITY
    for name, args in _TRANSFORM_RE.findall(s or ""):
        vals = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        if name == "matrix" and len(vals) == 6:
            t: Matrix = tuple(vals)  # type: ignore[assignment]
        elif name == "translate":
            tx = vals[0] if vals else 0.0
            ty = vals[1] if len(vals) > 1 else 0.0
            t = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = vals[0] if vals else 1.0
            sy = vals[1] if len(vals) > 1 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate":
            a = math.radians(vals[0] if vals else 0.0)
            ca, sa = math.cos(a), math.sin(a)
            t = (ca, sa, -sa, ca, 0.0, 0.0)
            if len(vals) >= 3:                 # rotate(a, cx, cy) = 平移到原点再转回来
                cx, cy = vals[1], vals[2]
                t = mat_mul(mat_mul((1, 0, 0, 1, cx, cy), t), (1, 0, 0, 1, -cx, -cy))
        elif name == "skewX":
            t = (1.0, 0.0, math.tan(math.radians(vals[0])), 1.0, 0.0, 0.0)
        elif name == "skewY":
            t = (1.0, math.tan(math.radians(vals[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue
        m = mat_mul(m, t)
    return m


# ---------------------------------------------------------------- path d


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(" + _NUM + r")")


def parse_path_d(d: str) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    """把 ``d`` 拆成直线段；曲线一律用**弦**近似并记一条警告。

    为什么敢用弦近似：本工具自己画的 SVG 全是直线；真正需要弧长精度的场合
    （例如按圆弧判电感符号）在 SVG 通道里本来就够用 —— 圆弧的存在性就够了，
    不需要它的弯曲程度。
    """
    warnings: list[str] = []
    tokens: list[tuple[str, str]] = []
    for cmd, num in _CMD_RE.findall(d or ""):
        tokens.append((cmd, num) if cmd else ("", num))

    segs: list[tuple[float, float, float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl: tuple[float, float] | None = None
    i = 0
    cmd = ""

    def num(i: int) -> float:
        return float(tokens[i][1]) if i < len(tokens) and tokens[i][1] else 0.0

    while i < len(tokens):
        if tokens[i][0]:
            cmd = tokens[i][0]
            i += 1
            if cmd in "Zz":
                if cur != start:
                    segs.append((cur[0], cur[1], start[0], start[1]))
                cur = start
                last_ctrl = None
                continue
        rel = cmd.islower()
        C = cmd.upper()

        if C == "M":
            if i + 1 >= len(tokens):
                break
            x, y = num(i), num(i + 1)
            i += 2
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            start = cur
            last_ctrl = None
            cmd = "l" if rel else "L"       # 后续隐式 L
        elif C == "L":
            if i + 1 >= len(tokens):
                break
            x, y = num(i), num(i + 1)
            i += 2
            nx, ny = (cur[0] + x, cur[1] + y) if rel else (x, y)
            segs.append((cur[0], cur[1], nx, ny))
            cur = (nx, ny)
            last_ctrl = None
        elif C == "H":
            if i >= len(tokens):
                break
            x = num(i)
            i += 1
            nx = cur[0] + x if rel else x
            segs.append((cur[0], cur[1], nx, cur[1]))
            cur = (nx, cur[1])
            last_ctrl = None
        elif C == "V":
            if i >= len(tokens):
                break
            y = num(i)
            i += 1
            ny = cur[1] + y if rel else y
            segs.append((cur[0], cur[1], cur[0], ny))
            cur = (cur[0], ny)
            last_ctrl = None
        elif C in ("C", "S", "Q", "T"):
            nargs = {"C": 6, "S": 4, "Q": 4, "T": 2}[C]
            if i + nargs - 1 >= len(tokens):
                break
            vals = [num(i + k) for k in range(nargs)]
            i += nargs
            pts_abs = []
            for k in range(0, nargs, 2):
                px, py = vals[k], vals[k + 1]
                pts_abs.append((cur[0] + px, cur[1] + py) if rel else (px, py))
            end = pts_abs[-1]
            segs.append((cur[0], cur[1], end[0], end[1]))
            warnings.append(
                f"path 里有贝塞尔曲线命令 {C}，已用**弦**近似（端点相连的直线）。"
                "本工具回绘的 SVG 不含曲线，所以这条警告通常出现在外来文件里。"
            )
            cur = end
            last_ctrl = pts_abs[-2] if len(pts_abs) >= 2 else None
        elif C == "A":
            if i + 6 >= len(tokens):
                break
            vals = [num(i + k) for k in range(7)]
            i += 7
            x, y = vals[5], vals[6]
            end = (cur[0] + x, cur[1] + y) if rel else (x, y)
            segs.append((cur[0], cur[1], end[0], end[1]))
            warnings.append(
                "path 里有圆弧命令 A，已用**弦**近似。本层只需要「圆弧存在」这个事实"
                "（用于判电感符号），不需要弧的弯曲程度。"
            )
            cur = end
            last_ctrl = None
        else:
            i += 1
    return segs, warnings


# ---------------------------------------------------------------- 主解析


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


#: 本工具回绘的 SVG 会给自己画的**非电路图元**打上这些 class。
#: ``ca-bg``   背景底色块   ``ca-deco`` 纯装饰   ``ca-node-mark`` 度为 2 的装饰小圆
DECO_CLASSES = frozenset({"ca-bg", "ca-deco", "ca-node-mark"})

#: 会产生"导线/图形"的元素（text 不在其列 —— 文字用 fill，不受 stroke 影响）
_SHAPE_TAGS = frozenset({"line", "polyline", "polygon", "path", "rect", "circle", "ellipse"})


def _is_decoration(el, tag: str) -> bool:
    """判断这个图元是**装饰**而不是电路图形。

    判据按可靠性排序，两条都要小心用：

    1. class 命中 ``ca-*`` 装饰标记 —— 自家回绘 SVG 的显式声明，最可靠。
    2. ``stroke="none"`` —— 没有轮廓。导线和元件符号必须描边，所以**矩形/路径/折线**
       没描边就不是导线。本工具的白色底色块正是这么画的：不排掉它，回导时会被读成
       4 条首尾相接的导线，凭空造出一个闭合方框，还污染"线段数"这个读图依据。

    ★ ``circle``/``ellipse`` **必须豁免第 2 条**：结点圆点就是「有填充、无描边」的圆，
    按 stroke 判装饰会把**所有圆点删光** —— 那等于把"圆点数量"这个读图依据清零，
    比不判还糟。（我第一版就是这么错的，回导后圆点从 4 个变成 0 个。）
    圆的装饰性只能靠 class（``ca-node-mark``）声明，以及 ``is_dot`` 的尺寸下限来挡。
    """
    if set((el.get("class") or "").split()) & DECO_CLASSES:
        return True
    if tag in ("circle", "ellipse"):
        return False
    return (el.get("stroke") or "").strip().lower() in ("none", "transparent")


def parse_svg_text(text: str, *, name: str = "svg") -> Geometry:
    """解析 SVG 文本，产出 Geometry。"""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise CircuitError(f"不是合法 XML（{e}）") from None

    geo = Geometry(channel="svg")
    if root.get("data-ca-refnode"):
        geo.ref_hint = root.get("data-ca-refnode")
    vb = root.get("viewBox")
    if vb:
        try:
            vx, vy, w, h = (float(v) for v in re.split(r"[\s,]+", vb.strip()))
            geo.size = (w, h)
            geo.viewbox = (vx, vy, w, h)
        except ValueError:
            pass
    elif root.get("width") and root.get("height"):
        try:
            w = float(re.sub(r"[^\d.]", "", root.get("width")))
            h = float(re.sub(r"[^\d.]", "", root.get("height")))
            geo.size = (w, h)
            geo.viewbox = (0.0, 0.0, w, h)
        except ValueError:
            pass

    _walk(root, IDENTITY, geo, 0)
    if geo.deco_skipped:
        geo.warnings.append(
            f"有 {geo.deco_skipped} 个无描边（stroke=\"none\"）或带装饰标记的图元未计入几何。"
            "这些通常是背景色块或本工具的装饰圆点，不是导线。"
            "若图里的元件符号恰好没有描边，会被一并跳过 —— 请在界面上核对。"
        )
    return geo


def _num(el, key, default=0.0) -> float:
    v = el.get(key)
    if v is None:
        return default
    m = re.match(r"^\s*(" + _NUM + r")", v)
    return float(m.group(1)) if m else default


def _walk(el, m: Matrix, geo: Geometry, depth: int) -> None:
    if depth > 60:
        geo.warnings.append("SVG 嵌套超过 60 层，已停止向下解析（防御畸形文件）")
        return

    tag = _strip_ns(el.tag)
    m = mat_mul(m, parse_transform(el.get("transform", "")))

    # 装饰图元直接跳过（见 _is_decoration）。text 不受影响 —— 它用 fill 不用 stroke。
    if tag in _SHAPE_TAGS and _is_decoration(el, tag):
        geo.deco_skipped += 1
        for child in el:
            _walk(child, m, geo, depth + 1)
        return

    if tag == "g":
        # 语义提示组
        if el.get("data-ca-ref"):
            hint = {
                "ref": el.get("data-ca-ref"),
                "kind": el.get("data-ca-kind"),
                "a": el.get("data-ca-a"),
                "b": el.get("data-ca-b"),
                "value": None,
            }
            raw = el.get("data-ca-value")
            if raw:
                # ★ 走 ``parse_engineering`` 而不是裸 ``float()``。
                #   原本只认浮点字面量，于是界面上的手绘画布要写这个属性，
                #   就必须在 JS 里再实现一遍工程记法解析 —— 而 ``1M`` / ``1m``
                #   这种约定（兆 vs 毫）最容易在复制品里**静默**解析错，差 10^9 倍，
                #   三法互校还拦不住（三法共用同一份 IR，会一致地给出同一个错答案）。
                #   直接复用这一份，既不给复制品留机会，又顺带把它的警告留痕。
                #   浮点字面量照样解析（回绘 SVG 写的就是浮点）。
                v, vw = parse_engineering(raw)
                hint["value"] = v
                if vw:
                    ref = hint.get("ref") or "?"
                    geo.warnings.append(
                        f"数据标记 data-ca-value={raw!r}（{ref}）的解析提示："
                        + "；".join(vw))
            p1 = el.get("data-ca-p1")
            p2 = el.get("data-ca-p2")
            if p1:
                try:
                    hint["p1"] = tuple(float(v) for v in p1.split(","))
                except ValueError:
                    pass
            if p2:
                try:
                    hint["p2"] = tuple(float(v) for v in p2.split(","))
                except ValueError:
                    pass
            # ---- 受控源的**控制支路**
            # ★ 控制端也必须按**坐标**带过来，不能带节点名：
            #   画布/渲染器都不该替解析器把名字定死（见 MEMORY「画布绝不写 data-ca-a/b」）。
            #   控制端是"图上另一个位置"，那就老老实实给位置，让解析器自己算出
            #   它落在哪个结点上 —— 与元件自身两个端子的处理完全一致。
            ctl_mode = (el.get("data-ca-ctrl-mode") or "").strip().upper()
            ctl: dict[str, Any] = {}
            if ctl_mode in ("V", "I"):
                ctl["mode"] = ctl_mode
            for key, attr in (("cp1", "data-ca-ctrl-p1"), ("cp2", "data-ca-ctrl-p2")):
                raw = el.get(attr)
                if not raw:
                    continue
                try:
                    ctl[key] = tuple(float(v) for v in raw.split(","))
                except ValueError:
                    pass
            # 电流控制：被采样支路是**位号**（位号本来就是语义身份，不是节点名，
            # 与 data-ca-ref 同一性质，所以这里可以直接带）
            if el.get("data-ca-ctrl-ref"):
                ctl["ref"] = el.get("data-ca-ctrl-ref")
            # 自动插入的 0V 探针位号。带上它是为了**幂等**：丢了它，回导时会
            # 以为还没插过探针，于是又串一个 0V 源、又多一个内部节点 ——
            # 电学行为不变，但元件数每往返一次就涨一次。
            if el.get("data-ca-ctrl-sense"):
                ctl["sense"] = el.get("data-ca-ctrl-sense")
            if el.get("data-ca-ctrl-expr"):
                ctl["expr"] = el.get("data-ca-ctrl-expr")
            if ctl:
                hint["ctrl"] = ctl
            if "p1" in hint and "p2" in hint:
                geo.hints.append(hint)

    elif tag == "line":
        x1, y1 = _num(el, "x1"), _num(el, "y1")
        x2, y2 = _num(el, "x2"), _num(el, "y2")
        a = apply_mat(m, x1, y1)
        b = apply_mat(m, x2, y2)
        geo.segments.append((a[0], a[1], b[0], b[1]))

    elif tag == "polyline" or tag == "polygon":
        vals = [float(v) for v in re.split(r"[\s,]+", (el.get("points") or "").strip())
                if v and re.match(r"^" + _NUM + r"$", v)]
        pts = [apply_mat(m, vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if tag == "polygon" and len(pts) > 2:
            pts = pts + [pts[0]]
        geo.polys.append(pts)
        for i in range(len(pts) - 1):
            geo.segments.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))

    elif tag == "path":
        segs, w = parse_path_d(el.get("d") or "")
        for x1, y1, x2, y2 in segs:
            a = apply_mat(m, x1, y1)
            b = apply_mat(m, x2, y2)
            geo.segments.append((a[0], a[1], b[0], b[1]))
        for x in w:
            if x not in geo.warnings:
                geo.warnings.append(x)
        # 顺带把 path 里出现的圆弧命令记成"存在圆弧"（判电感符号用）
        if re.search(r"[Aa]", el.get("d") or ""):
            pts = []
            for x1, y1, x2, y2 in segs:
                pts.append(apply_mat(m, x1, y1))
            if pts:
                geo.arcs.append({"pts": pts, "approx": "chord"})

    elif tag == "rect":
        x, y = _num(el, "x"), _num(el, "y")
        w, h = _num(el, "width"), _num(el, "height")
        # 用四角 + 旋转，支持被 rotate 过的电阻框
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        abs_pts = [apply_mat(m, cx, cy) for cx, cy in corners]
        for i in range(4):
            a, b = abs_pts[i], abs_pts[(i + 1) % 4]
            geo.segments.append((a[0], a[1], b[0], b[1]))
        geo.rects.append({
            "cx": sum(p[0] for p in abs_pts) / 4,
            "cy": sum(p[1] for p in abs_pts) / 4,
            "w": w, "h": h, "pts": abs_pts,
        })

    elif tag in ("circle", "ellipse"):
        cx, cy = _num(el, "cx"), _num(el, "cy")
        r = _num(el, "r") or max(_num(el, "rx"), _num(el, "ry"))
        c = apply_mat(m, cx, cy)
        scale = math.hypot(m[0], m[1]) or 1.0
        rr = r * scale
        fill = (el.get("fill") or "").lower()
        cls = el.get("class") or ""
        # ★ 什么算"结点圆点"？三条都满足才行：
        #   1) 有填充（空心圆是元件符号，不是接线点）
        #   2) 半径够大（>= 2.8px）——本工具给度为 2 的节点画了 r=2.4 的
        #      **装饰性**小圆，尺寸上就排除了，不依赖 class 也能挡住
        #   3) 没被显式标成装饰（class 含 ca-node-mark）
        #   只写"有填充"一条会误判，而且误判的是"圆点数量"这个读图依据，
        #   等于在报告里报了假的证据。（真踩过。）
        is_dot = (fill not in ("none", "transparent", "")
                  or "junction" in cls) and rr >= 2.8 and "ca-node-mark" not in cls
        geo.circles.append({"cx": c[0], "cy": c[1], "r": rr,
                            "fill": fill, "is_dot": is_dot})
        if is_dot:
            geo.dots.append((c[0], c[1], rr))

    elif tag == "text":
        # SVG 里的真文字 —— 数值可以直接读出来，不需要 VLM
        content = "".join(el.itertext()).strip()
        if content:
            tx = el.get("x")
            ty = el.get("y")
            pos = None
            if tx and ty:
                try:
                    pos = apply_mat(m, float(re.sub(r"[^\d.\-]", "", tx)),
                                    float(re.sub(r"[^\d.\-]", "", ty)))
                except ValueError:
                    pos = None
            geo.texts.append({"text": content, "pos": pos})

    for child in el:
        _walk(child, m, geo, depth + 1)


# ---------------------------------------------------------------- 读数


def attach_text_values(geo: Geometry, slots: list[dict]) -> None:
    """把 ``<text>`` 里的数字贴到最近的元件槽位上。

    SVG 通道独有的便宜事：矢量图里的文字是**真文字**，能直接解析出 1k / 4.7uF，
    于是不需要视觉模型也不需要 OCR。位图通道没有这个待遇。
    """
    for sl in slots:
        mx = (sl["p"][0] + sl["q"][0]) / 2
        my = (sl["p"][1] + sl["q"][1]) / 2
        best, bestd, besttxt = None, 90.0, None
        for t in geo.texts:
            if not t.get("pos"):
                continue
            d = math.hypot(t["pos"][0] - mx, t["pos"][1] - my)
            if d < bestd:
                best, bestd, besttxt = t, d, t["text"]
        if best is None:
            continue
        sl["nearby_text"] = besttxt
        # 先试整串，再试串里的数字片段
        for cand in [besttxt] + re.findall(r"[0-9][0-9a-zA-Zµμ.]*", besttxt or ""):
            v, _ = parse_engineering(cand)
            if v is not None:
                sl["value"] = v
                break


def svg_to_ir(svg_text: str, *, name: str = "svg", tol: float = 6.0,
              max_gap: float = DEFAULT_MAX_GAP, ref_node: str | None = None):
    """SVG 文本 -> (IR, 读图依据报告)。

    ★ 元件占位（槽位）**无论有没有语义标记都要先算**：拓扑层要用它算出
    "元件本体圆柱"，把符号本体（电阻框这类闭环导体）从导线图里剔除掉。
    少了这一步，电阻框会把自己的两个端子短接，两个结点被误并成一个、**而且不报错**。
    """
    geo = parse_svg_text(svg_text, name=name)
    from .topology import detect_component_slots
    slots = detect_component_slots(geo, tol=tol, max_gap=max_gap)
    # SVG 独有便宜事：矢量图的文字是真文字，能直接解析出 1k / 4.7uF。
    # 有语义标记时也照读不误（最终取值以标记为准），顺带留出交叉印证的空间。
    attach_text_values(geo, slots)
    return build_topology(geo, tol=tol, max_gap=max_gap, name=name,
                          ref_node=ref_node, slots=slots)


def svg_file_to_ir(path: str | Path, **kw):
    p = Path(path)
    if not p.is_file():
        raise CircuitError(f"SVG 文件不存在：{p}")
    if p.suffix.lower() not in (".svg", ".svgz") and not p.name.lower().endswith(".svg"):
        raise CircuitError(f"{p.name} 看起来不是 SVG 文件（扩展名 {p.suffix}）")
    return svg_to_ir(p.read_text(encoding="utf-8", errors="replace"),
                     name=p.stem, **kw)
