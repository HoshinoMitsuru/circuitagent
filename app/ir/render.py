"""IR -> SVG 回绘渲染器。

两个用途，缺一不可：

1. **叠图对账**：把解析出来的结构按**原图像素坐标系**重画一遍，前端以半透明
   图层盖在照片上。错线、漏线、圆点判错、跨线判反，一眼就能看出来 ——
   这是"不许靠肉眼定连接"这条禁令唯一能落地的验证手段：不靠肉眼看原图，
   而是靠肉眼看**机器读出来的结果**与原图的差集。
2. **报告插图**：没有原图时，用自动布局画一张干净的电路图，随报告一起给出。

元件符号按 GB/T 4728（国标）习惯画：电阻用矩形框，电压源用圆圈带 +/−，
电流源用圆圈带箭头 —— 对齐国内《电路》教材的读图习惯。
"""

from __future__ import annotations

import math
from typing import Any

from .model import (CONTROL_NOTE, GAIN_SYMBOL, Circuit, CircuitError,
                    REF_NODE, CONTROLLED_KINDS, declared_direction)

# 画布与图元尺寸（像素）
PAD = 60
HL = 30.0          # 元件半长（两端引线到中心的距离）
BODY = 15.0        # 元件本体半长
BODY_W = 10.0      # 元件本体半宽
DOT_R = 5.0        # 结点圆点半径
LABEL_DY = -16.0


# ---------------------------------------------------------------- 布局


def layout_geom(circuit: Circuit) -> dict[str, tuple[float, float]]:
    """用元件自带的视觉坐标反推节点位置（图像解析路径）。

    对同一节点的多个端点取算术平均 —— 照片透视畸变会让同一节点的两个端点
    差十几像素，取平均比任取一个都稳。
    """
    acc: dict[str, list[tuple[float, float]]] = {}
    for c in circuit.components:
        p1, p2 = c.geom.get("p1"), c.geom.get("p2")
        if not p1 or not p2:
            continue
        acc.setdefault(c.nodes[0], []).append(tuple(p1))
        acc.setdefault(c.nodes[1], []).append(tuple(p2))
    if not acc:
        raise CircuitError("没有元件带视觉坐标，无法按原图坐标回绘")
    return {
        n: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
        for n, ps in acc.items()
    }


def layout_grid(circuit: Circuit) -> dict[str, tuple[float, float]]:
    """自动布局：从参考节点 BFS 分层，层号 -> 列，层内序号 -> 行。

    对串联/并联为主的教材电路效果不错。参考节点固定摆在左下区域，
    让"地"在视觉上沉底，符合读图直觉。
    """
    nodes = circuit.nodes
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for c in circuit.components:
        adj[c.nodes[0]].append(c.nodes[1])
        adj[c.nodes[1]].append(c.nodes[0])

    depth: dict[str, int] = {circuit.ref_node: 0}
    order: list[str] = [circuit.ref_node]
    queue = [circuit.ref_node]
    while queue:
        x = queue.pop(0)
        for y in adj[x]:
            if y not in depth:
                depth[y] = depth[x] + 1
                order.append(y)
                queue.append(y)
    for n in nodes:                      # 兜底（理论上过不了 validate）
        depth.setdefault(n, max(depth.values()) + 1)

    cols: dict[int, list[str]] = {}
    for n in order:
        cols.setdefault(depth[n], []).append(n)

    col_gap, row_gap = 190.0, 130.0
    pos: dict[str, tuple[float, float]] = {}
    for d, ns in sorted(cols.items()):
        for i, n in enumerate(ns):
            y = PAD + i * row_gap
            pos[n] = (PAD + d * col_gap, y)

    # 让参考节点尽量居中，避免整张图往一角挤
    if circuit.ref_node in pos:
        ys = [p[1] for p in pos.values()]
        pos[circuit.ref_node] = (pos[circuit.ref_node][0],
                                 PAD + (max(ys) + min(ys)) / 2 - PAD)
    return pos


def layout_radial(circuit: Circuit) -> dict[str, tuple[float, float]]:
    """环形布局：参考节点居中，其余按 BFS 深度分层放在同心圆上。

    为什么非要它不可：分层版式把 BFS 同深度的结点排在同一列，遇到**电桥**这种
    "同层三结点、彼此两两相连"的图，长支路的导线会**穿过**邻元件的符号框 ——
    视觉上糊成一团，更麻烦的是几何上会把两个结点误并（`layout_overlaps` 会报出来）。
    环形布局里同一层的边都是圆的**弦**，弦与弦不会互相穿过，天然避掉这个坑。

    参考节点摆在圆心，其余按深度分层，符合"地/参考点在中间"的读图直觉；
    同层结点按 BFS 次序等角分布，从正上方起顺时针排 —— 确定性，可测。
    """
    nodes = circuit.nodes
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for c in circuit.components:
        adj[c.nodes[0]].append(c.nodes[1])
        adj[c.nodes[1]].append(c.nodes[0])

    depth: dict[str, int] = {circuit.ref_node: 0}
    order: list[str] = [circuit.ref_node]
    queue = [circuit.ref_node]
    while queue:
        x = queue.pop(0)
        for y in adj[x]:
            if y not in depth:
                depth[y] = depth[x] + 1
                order.append(y)
                queue.append(y)
    for n in nodes:
        depth.setdefault(n, max(depth.values()) + 1)

    rings: dict[int, list[str]] = {}
    for n in order:
        rings.setdefault(depth[n], []).append(n)

    #: 同层相邻结点至少要留出这么长的弧，否则把圆撑大 —— 元件符号本身就有几十像素宽，
    #: 挤在一起会让不同支路的符号叠上。
    arc_need = 165.0
    pos: dict[str, tuple[float, float]] = {circuit.ref_node: (0.0, 0.0)}
    for d, ns in sorted(rings.items()):
        if d == 0:
            continue
        k = len(ns)
        r = max(175.0 * d, arc_need * k / (2 * math.pi))
        for i, n in enumerate(ns):
            th = math.radians(90.0 - 360.0 * i / k)
            pos[n] = (r * math.cos(th), -r * math.sin(th))   # SVG 的 y 向下
    return pos


def layout_auto(circuit: Circuit) -> tuple[dict[str, tuple[float, float]], str]:
    """按**自检结果**选版式，返回 ``(坐标, 版式名)``。

    先说走过的弯路：我一开始想按"疏密"一步定版式（边数 ≥ 节点数 → 环形），
    结果被一个单回路打脸 —— 单回路的边数正好等于节点数，但它分层画出来
    是一条漂亮的直线，环形反而把它摊成一个圆。**疏密不是正确判据。**

    正确判据其实就是"画出来有没有毛病"，而这件事已经有现成的检查器
    （``layout_overlaps``）。所以这里不猜，改成**生成-检验**：

    1. 先试**分层** —— 它是教材那种直线版式，好看，优先；
    2. 自检不干净就试**环形** —— 同层的边是弦、弦不相穿，专治分层会撞的那些图；
    3. 两个都不干净，就留重叠**较少**的那张，让上层的自检如实报出来
       （不静默地用一张烂图）。

    代价是每张图要算两遍布局 —— 图很小，这点开销换"版式一定自检过关"很值。
    """
    best: tuple[int, dict[str, tuple[float, float]], str] | None = None
    for name, fn in (("grid", layout_grid), ("radial", layout_radial)):
        pos = fn(circuit)
        n_bad = len(layout_overlaps(circuit, pos))
        if n_bad == 0:
            return pos, name
        if best is None or n_bad < best[0]:
            best = (n_bad, pos, name)
    assert best is not None
    return best[1], best[2]


# ---------------------------------------------------------------- 元件符号


def _glyph_resistor() -> str:
    """电阻：国标矩形框（**只画本体**，引线由 render_svg 从节点画过来）。"""
    return (f'<rect x="{-BODY}" y="{-BODY_W}" width="{2*BODY}" height="{2*BODY_W}" '
            f'fill="none" stroke-width="2"/>')


def _glyph_voltage() -> str:
    """电压源：圆圈 + 极性标注。+ 号画在 nodes[0] 一侧（即 + 端）。

    极性是本题最容易看错、也最致命的读数，所以把 +/− 画成**圆圈外侧的红色文字**，
    与 IR 的 nodes[0] 严格绑定，回绘图上能直接逐点核对。

    ★ 圈**内**原来还画了两笔"极性笔画"，已删除。两个原因：
    1. 那两笔的画法是错的：``M -6 -7 L -6 1`` + ``M -10 -3 L -2 -3`` 是竖+横=**十字**，
       正负两侧画出来都是加号，根本看不出极性。
    2. 更要命的是它们伸到局部 ±10，而端子（导线端点）在 ±13 —— 只差 3px，
       在容差 6px 之内。于是解析器把"导线端点"和"极性笔画端点"当成同一个点，
       导线端点因此**不算自由端点**，电压源的槽位永远挑不出来，
       连带着它的本体也没被剔除。（真踩过：整张图 6 个元件只挑出 5 个。）
    """
    r = 13.0
    p = [f'<circle cx="0" cy="0" r="{r}" fill="none" stroke-width="2"/>']
    p.append(f'<text x="{-r-3}" y="{-r-2}" text-anchor="end" class="pol">+</text>')
    p.append(f'<text x="{r+3}" y="{-r-2}" text-anchor="start" class="pol">-</text>')
    return "".join(p)


def _glyph_current() -> str:
    """电流源：圆圈内画箭头，箭头方向 = nodes[0] -> nodes[1]。

    箭头末端刻意收在局部 (4, 0)，离端子（局部 ±13）留 9px 余量 ——
    同电压源那个坑：图形伸到离端子 6px 以内，就把导线端点"粘"住了。
    """
    r = 13.0
    p = [f'<circle cx="0" cy="0" r="{r}" fill="none" stroke-width="2"/>']
    p.append('<path d="M -8 0 L 4 0 M 0 -4 L 4 0 L 0 4" stroke-width="1.8" fill="none"/>')
    return "".join(p)


def _glyph_capacitor() -> str:
    """电容：两片极板 + 一小段本体引线。

    ★ 本体引线不能省：极板只到 ±4，若导线也画到 ±4，剩下的空隙只有 8px，
    刚刚擦着容差边界（默认 6px），元件槽位会时灵时不灵。
    留一段本体引线让空隙稳定在 24px。
    """
    return ('<path d="M -12 0 L -4 0 M 4 0 L 12 0" stroke-width="2"/>'
            '<path d="M -4 -11 L -4 11 M 4 -11 L 4 11" stroke-width="2.4"/>')


def _glyph_inductor() -> str:
    """电感：四个半圆（本体引线连同圆弧一起画，自成一段可识别的弧串）。"""
    p = [f'<path d="M {-BODY-4} 0 L {-BODY} 0 M {BODY} 0 L {BODY+4} 0" stroke-width="2"/>']
    seg = (2 * BODY) / 4
    d = f"M {-BODY} 0"
    for i in range(4):
        x0 = -BODY + i * seg
        d += f" A {seg/2} {seg/2} 0 0 1 {x0+seg} 0"
    p.append(f'<path d="{d}" fill="none" stroke-width="2"/>')
    return "".join(p)


def _glyph_dependent(kind: str) -> str:
    """受控源：**菱形** + 增益记号（μ / gm / rm / α）。

    菱形是教材里受控源的标准画法，四类共用；区分靠里面的字，
    而那个字正是 IR 里 ``GAIN_SYMBOL[kind]`` 与网表卡上的增益符号 ——
    同一个记号贯穿"画出来 / 参数表 / 网表"，不给人换一次记号的机会。

    ★ 尺寸要当心：菱形顶点必须**正好落在端子** ``±BODY_HALF`` 上。
    画小了，导线的自由端点会落进符号内部；画大了，导线会伸进符号里 ——
    两种都会让几何回导把元件端点判错（与 ``_glyph_voltage`` 那次的
    极性笔画是同一个坑）。
    """
    hx = BODY_HALF[kind]                 # 沿轴线的半长
    hy = hx * 0.72                       # 半宽按比例，四类看起来一样"胖"
    p = [f'<polygon points="{-hx},0 0,{-hy} {hx},0 0,{hy}" fill="none" stroke-width="2"/>']
    p.append(f'<text x="0" y="4" text-anchor="middle" class="gainsym">'
             f'{_esc(GAIN_SYMBOL.get(kind, kind))}</text>')
    return "".join(p)


def _glyph_E() -> str:
    return _glyph_dependent("E")


def _glyph_G() -> str:
    return _glyph_dependent("G")


def _glyph_H() -> str:
    return _glyph_dependent("H")


def _glyph_F() -> str:
    return _glyph_dependent("F")


GLYPHS = {
    "R": _glyph_resistor,
    "V": _glyph_voltage,
    "I": _glyph_current,
    "C": _glyph_capacitor,
    "L": _glyph_inductor,
    # 受控源四类。它们进了 GLYPHS，前端画布的元件调色板就会自动多出这四个
    # （``/api/symbols`` 直接吃这张表）——"手动画图要能画受控源"就是这么来的，
    # 不用改前端一行。
    "E": _glyph_E,
    "G": _glyph_G,
    "H": _glyph_H,
    "F": _glyph_F,
}

#: 各符号沿支路轴线的**半长**（本体占多宽）。
#: 这个数与符号画法必须严格对齐：画大了导线会塞进符号里，
#: 画小了导线的自由端点会落在符号内部，两种都会让几何回导把元件端点判错。
BODY_HALF: dict[str, float] = {"R": 15.0, "V": 13.0, "I": 13.0, "C": 12.0, "L": 19.0,
                             "E": 17.0, "G": 17.0, "H": 17.0, "F": 17.0}

#: 中文名与数值单位。**画布的元件调色板直接吃这张表** ——
#: 后端 `GLYPHS` 里加一种元件，前端的调色板就自动多一个（不用改前端）。
KIND_LABEL: dict[str, str] = {
    "R": "电阻", "V": "电压源", "I": "电流源", "C": "电容", "L": "电感",
    # ★ 受控源的标签里必须**带上"受控"二字**。画布调色板、
    #   结构确认表、报告都读这张表 —— 标签一含糊，用户就没法一眼分清
    #   "电压源"和"电压控制电压源"，而那正是最容易读错、后果最大的一处。
    "E": "受控·压控压源", "G": "受控·压控流源",
    "H": "受控·流控压源", "F": "受控·流控流源",
}
KIND_UNIT: dict[str, str] = {"R": "Ω", "V": "V", "I": "A", "C": "F", "L": "H",
                             "E": "V/V", "G": "S", "H": "Ω", "F": "A/A"}

#: 受控源类型的完整说明（画布提示与报告都用它，文字来自 params.CONTROL_NOTE）
KIND_DETAIL: dict[str, str] = dict(CONTROL_NOTE)


def theme_colors(dark: bool = False) -> dict[str, str]:
    """主题配色。

    ★ **画布与渲染器必须共用这一份**：手绘面板里的元件就是用这里的画法
    （同一个 ``GLYPHS``）与这里的配色渲染的。两边一旦分家，
    "画的时候看到的"和"出图后看到的"就不是一个样子 ——
    而"画完看一眼对不对"正是这个面板唯一的验收方式。
    """
    if not dark:
        return {"bg": "#ffffff", "fg": "#1b2430", "wire": "#1b2430",
                "sub": "#5b6875", "accent": "#c0392b"}
    return {"bg": "#12161d", "fg": "#e8eef6", "wire": "#cfd8e3",
            "sub": "#9fb0c0", "accent": "#ff8a75"}


def style_block(dark: bool = False) -> str:
    """渲染器与手绘画布共用的 CSS。

    ★ 为什么要抽出来而不是各写一份：符号图形（``GLYPHS``）里**只写
    ``stroke-width``、不写颜色**，颜色全靠这段 CSS 给。画布如果要自己再写一份
    等价规则，就等于把"符号长什么样"这件事复制成两份 —— 改一处忘一处，
    表现是画布上元件变成透明或全黑，而渲染器那边一切正常。
    """
    c = theme_colors(dark)
    bg, fg, wire, sub, accent = c["bg"], c["fg"], c["wire"], c["sub"], c["accent"]
    return f"""<style>
.circuit-svg text{{font-family:Consolas,Menlo,"Microsoft YaHei",monospace;font-size:13px;fill:{fg}}}
.circuit-svg .lbl{{font-size:13px;font-weight:600}}
.circuit-svg .val{{font-size:12px;fill:{sub}}}
.circuit-svg .node{{font-size:12px;fill:{sub}}}
.circuit-svg .pol{{font-size:14px;font-weight:700;fill:{accent}}}
/* 受控源：菱形里的增益记号、以及符号旁那行"受谁控制"。
   两者都用 accent 色 —— 同一张图上"受控"这件事必须一眼可辨，
   它和"这是个普通电压源"是完全不同的两件事。 */
.circuit-svg .gainsym{{font-size:13px;font-weight:700;fill:{accent};stroke:none}}
.circuit-svg .ctl{{font-size:11px;fill:{accent};stroke:none}}
/* 控制连线本身是 ca-deco，它由下面的 .ca-deco 规则免描边。
   ★ 注意 CSS 特意性：上面那条 `.circuit-svg path,.circuit-svg line` 会把
   line 描成 wire 色，所以 ca-deco 的规则必须写在它**之后**（这里就是之后），
   否则虚线会被描黑、看起来像真的导线。 */
.circuit-svg path,.circuit-svg line,.circuit-svg rect,.circuit-svg circle{{stroke:{wire}}}
.circuit-svg .glyph path,.circuit-svg .glyph line,.circuit-svg .glyph rect,.circuit-svg .glyph circle{{stroke:{fg}}}
.circuit-svg .body{{fill:none}}
/* ★ 背景与装饰必须显式免描边。CSS 规则的特异性高于 presentation attribute，
   只写 stroke="none" 是**压不住**上面那条 .circuit-svg rect 规则的 ——
   表现就是白底矩形被描出一圈黑框，而且回导时被当成 4 条导线。真踩过。 */
.circuit-svg .ca-bg{{stroke:none;fill:{bg}}}
.circuit-svg .ca-deco{{stroke:none;fill:none}}
.circuit-svg .ca-node-mark{{stroke:none;fill:{wire}}}
</style>"""


# ---------------------------------------------------------------- 布局自检


def layout_overlaps(circuit: Circuit, node_pos: dict[str, tuple[float, float]],
                    *, tol: float = 8.0) -> list[dict[str, Any]]:
    """自动布局的自我体检：**哪条支路的导线穿过了不属于它的结点**。

    为什么要专门做这件事：自动布局画出来的图，一旦有元件首尾堆叠在同一条直线上，
    长支路的导线就会**穿过**短元件的符号。后果有两层，第二层才是真麻烦：

    1. 看起来乱（导线上叠着别人的电阻框）；
    2. **几何上两个结点被合并** —— 因为导线端点与符号角点在容差内重合，
       回导时并查集把它们并成了一个节点。于是"报告插图另存为 SVG 再导回来"
       会得到一张**缺元件、少结点**的电路，而且**不报错**。

    第 2 条意味着：布局不干净的图不能拿去做几何往返。与其让它静默出错，
    不如在这里查出来、报出来，让上层（报告 / WebUI）明示"此图仅供示意，不可回导"。

    判据：结点 n 到支路 c 的导线段的距离 ≤ tol，且 n 不是 c 的端点。
    """
    hits: list[dict[str, Any]] = []
    for c in circuit.components:
        p1 = node_pos.get(c.nodes[0])
        p2 = node_pos.get(c.nodes[1])
        if not p1 or not p2:
            continue
        for n, q in node_pos.items():
            if n in c.nodes:
                continue
            if _point_seg_dist(q[0], q[1], p1[0], p1[1], p2[0], p2[1]) <= tol:
                hits.append({
                    "component": c.ref, "node": n,
                    "where": [round(q[0], 1), round(q[1], 1)],
                    "why": f"{c.ref} 的导线穿过结点 {n} —— 该支路与 {n} 无关，"
                           "两者在几何上会被误并成同一个结点（回导时会出错）",
                })

    # ---- 元件之间：两条支路的轴线几乎重合（**并联支路被画在同一条线上**）
    #     这个不报出来的话，两张符号会完全叠在一起，界面上只看得见其中一个 ——
    #     而并联在电路题里太常见了，不能让它静默地糊掉。
    comps = [c for c in circuit.components
             if node_pos.get(c.nodes[0]) and node_pos.get(c.nodes[1])]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            ax1, ay1 = node_pos[a.nodes[0]]
            ax2, ay2 = node_pos[a.nodes[1]]
            bx1, by1 = node_pos[b.nodes[0]]
            bx2, by2 = node_pos[b.nodes[1]]
            amx, amy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            alen = math.hypot(ax2 - ax1, ay2 - ay1) or 1.0
            bmx, bmy = (bx1 + bx2) / 2, (by1 + by2) / 2
            blen = math.hypot(bx2 - bx1, by2 - by1) or 1.0
            if math.hypot(amx - bmx, amy - bmy) > max(12.0, tol):
                continue
            cross = abs(((ax2 - ax1) / alen) * ((by2 - by1) / blen)
                        - ((ay2 - ay1) / alen) * ((bx2 - bx1) / blen))
            if cross < 0.12:
                hits.append({
                    "component": f"{a.ref}/{b.ref}", "node": "-",
                    "where": [round(amx, 1), round(amy, 1)],
                    "why": f"{a.ref} 与 {b.ref} 的轴线几乎重合（并联支路被排在同一位置），"
                           "两个符号会叠在一起。需要给并联支路分配不同路径。",
                })
    return hits


def _point_seg_dist(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


# ---------------------------------------------------------------- 主渲染


def render_svg(
    circuit: Circuit,
    node_pos: dict[str, tuple[float, float]] | None = None,
    *,
    mode: str = "auto",
    width: float | None = None,
    height: float | None = None,
    dark: bool = False,
    show_labels: bool = True,
    viewbox: tuple[float, float, float, float] | None = None,
    background: bool = True,
    node_labels: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    """把 IR 画成 SVG。

    mode:
      ``geom``   —— 按元件自带视觉坐标（原图像素），用于叠图对账
      ``auto``   —— 有视觉坐标就走 geom，否则按图的疏密自动选 ``radial``/``grid``
      ``radial`` —— 环形布局（参考节点居中 + 同心圆分层），密集图专用
      ``grid``   —— 分层布局（BFS 深度->列），稀疏图/串联链专用

    ``viewbox`` 强制指定视口 ``(x0, y0, w, h)``。**叠图对账必须传它**：
    回绘层要和原图逐像素对齐，就得用原图自己的视口；不传则按结点位置自动外扩，
    两张图的坐标系不一致，叠上去是错位的 —— 而对账的全部意义就在于"能叠准"。

    ``background`` 关掉底色块。**叠图时必须关**：白色底块会把原图整张盖住。

    ``node_labels`` 是否画结点名。不传则按版式自动决定（分层/环形画，几何叠图不画，
    因为叠图的原图上本来就有名字）。叠图想额外标出机器读出来的节点名就传 True。

    ``diagnostics`` 传入一个 dict 时会**就地填入**：
    ``{"layout": 实际用的布局, "overlaps": [...], "roundtrip_safe": bool}``。
    ``overlaps`` 非空说明这张图有支路导线穿过无关结点、或并联符号叠在一起 ——
    **只能当示意图看，不能拿去几何回导**。

    返回**裸 SVG 片段**（不含 <html>），便于前端直接内联。
    """
    if mode not in ("geom", "grid", "radial", "auto"):
        raise CircuitError(f"不认识的绘图模式 {mode!r}（可选 geom/grid/radial/auto）")

    if node_pos is None:
        has_geom = all(c.geom.get("p1") and c.geom.get("p2") for c in circuit.components)
        if mode == "auto":
            use_geom = has_geom
            resolved = "geom" if use_geom else "auto"
        else:
            use_geom = (mode == "geom")
            resolved = mode
        if use_geom:
            node_pos = layout_geom(circuit)
        elif resolved == "radial":
            node_pos = layout_radial(circuit)
        elif resolved == "grid":
            node_pos = layout_grid(circuit)
        else:                       # auto：生成-检验，选一张自检过关的版式
            node_pos, resolved = layout_auto(circuit)
    else:
        resolved = mode if mode != "auto" else "manual"

    if not node_pos:
        raise CircuitError("无法为这张电路生成坐标")

    if diagnostics is not None:
        diagnostics["layout"] = resolved
        diagnostics["overlaps"] = layout_overlaps(circuit, node_pos)
        diagnostics["roundtrip_safe"] = not diagnostics["overlaps"]

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    if viewbox is not None:
        x0, y0, vw, vh = (float(v) for v in viewbox)
        x1, y1 = x0 + vw, y0 + vh
    else:
        x0, x1 = min(xs) - PAD - 60, max(xs) + PAD + 60
        y0, y1 = min(ys) - PAD, max(ys) + PAD
        vw, vh = (x1 - x0), (y1 - y0)
    if width:
        sx = width / vw
        height = height or vh * sx
    elif height:
        sy = height / vh
        width = vw * sy
    else:
        width, height = vw, vh

    # 浅色主题：白底深字。叠图时前端再降透明度。
    # 配色与 CSS 走共用函数 —— 手绘面板用的是同一份（见 theme_colors / style_block）。
    tc = theme_colors(dark)
    bg, fg, wire = tc["bg"], tc["fg"], tc["wire"]
    sub, accent = tc["sub"], tc["accent"]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} '
        f'{vw:.2f} {vh:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'class="circuit-svg" data-ca-refnode="{_esc(circuit.ref_node)}">'
    )
    parts.append(style_block(dark))
    if background:
        parts.append(f'<rect class="ca-bg" x="{x0:.2f}" y="{y0:.2f}" width="{vw:.2f}" '
                     f'height="{vh:.2f}" fill="{bg}" stroke="none"/>')

    # ---- 支路（导线 + 元件符号）
    for c in circuit.components:
        p1 = node_pos.get(c.nodes[0])
        p2 = node_pos.get(c.nodes[1])
        if not p1 or not p2:
            continue
        L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        bh = BODY_HALF.get(c.kind, 15.0)

        # ★ 导线必须从**节点**画到**符号本体边缘**，而不是只画符号周围一小段。
        #   早先的实现只画了 ±30px 的引线，节点与引线之间那段根本没画 ——
        #   短元件看不出来，长元件（网格布局下动辄 200px）在报告插图里是**断的**，
        #   而且几何回导时会多出一堆莫名其妙的自由端点。
        ux, uy = (p2[0] - p1[0]) / (L or 1), (p2[1] - p1[1]) / (L or 1)
        if L > 2 * bh + 4:
            a_end = (mx - bh * ux, my - bh * uy)
            b_beg = (mx + bh * ux, my + bh * uy)
            parts.append(f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" '
                         f'x2="{a_end[0]:.2f}" y2="{a_end[1]:.2f}" stroke-width="2"/>')
            parts.append(f'<line x1="{b_beg[0]:.2f}" y1="{b_beg[1]:.2f}" '
                         f'x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke-width="2"/>')
        else:
            # 节点太近，导线与符号会叠在一起；直接画整条线，符号盖上去
            parts.append(f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" '
                         f'x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke-width="2"/>')

        glyph = GLYPHS.get(c.kind)
        # 语义标记：让本工具画出来的 SVG 能被 **精确无损地再次导入**。
        # 这是"可逆"的落地 —— 回绘图既能给人看，也能给机器读，
        # 于是"导出 SVG -> 重新导入 -> 三法对账"成了一条真正的回归测试链路。
        # 受控源的**控制支路**也要一起带上，否则"导出 → 再导入"会丢控制关系：
        # 而 IR 里受控源没有控制支路是硬错，于是回导会直接失败（或者更糟：
        # 被当成独立源）。控制端给**坐标**而不是节点名 —— 理由见
        # topology._control_from_hint：结点名是解析层算出来的，不该由画图的一方定死。
        ctl_attr = ""
        if c.kind in CONTROLLED_KINDS and c.ctrl is not None:
            mode = c.ctrl.mode
            ctl_attr = f' data-ca-ctrl-mode="{_esc(mode)}"'
            if mode == "V" and c.ctrl.nodes:
                cp = [node_pos.get(n) for n in c.ctrl.nodes]
                if all(cp):
                    ctl_attr += (f' data-ca-ctrl-p1="{cp[0][0]:.3f},{cp[0][1]:.3f}"'
                                 f' data-ca-ctrl-p2="{cp[1][0]:.3f},{cp[1][1]:.3f}"')
            elif mode == "I" and c.ctrl.ref:
                ctl_attr += f' data-ca-ctrl-ref="{_esc(c.ctrl.ref)}"'
                # ★ 自动插入的探针位号必须一起带上，否则"导出 → 再导入"会**再插一遍**：
                #   回导时 sense_ref 丢了，ensure_sense_sources 会以为还没插过，
                #   于是又串一个 0V 源、又多一个内部节点。电学行为不变（0V 串联
                #   不改变任何东西），但元件数/节点数会**每往返一次涨一次**，
                #   而参数表与报告里都会多出"题目里没有的东西"。
                if c.ctrl.sense_ref:
                    ctl_attr += f' data-ca-ctrl-sense="{_esc(c.ctrl.sense_ref)}"'
            if c.ctrl.expr:
                ctl_attr += f' data-ca-ctrl-expr="{_esc(c.ctrl.expr)}"'
        parts.append(f'<g class="glyph ca-component" transform="translate({mx:.2f},{my:.2f}) '
                     f'rotate({ang:.2f})" '
                     f'data-ca-ref="{_esc(c.ref)}" data-ca-kind="{_esc(c.kind)}" '
                     f'data-ca-a="{_esc(c.nodes[0])}" data-ca-b="{_esc(c.nodes[1])}" '
                     f'data-ca-p1="{p1[0]:.3f},{p1[1]:.3f}" '
                     f'data-ca-p2="{p2[0]:.3f},{p2[1]:.3f}"'
                     + (f' data-ca-value="{c.value}"' if c.value is not None else "")
                     + ctl_attr
                     + '>')
        if glyph:
            parts.append(glyph())
        else:
            # 未知类型：画个方框并打问号，绝不静默美化
            parts.append(f'<rect x="{-BODY}" y="{-BODY_W}" width="{2*BODY}" '
                         f'height="{2*BODY_W}" fill="none" stroke-width="2" '
                         'stroke-dasharray="4 3"/>')
            parts.append('<text x="0" y="4" text-anchor="middle">?</text>')
        if show_labels:
            parts.append(f'<g transform="rotate({-ang:.2f})">'
                         f'<text x="0" y="{LABEL_DY}" text-anchor="middle" class="lbl">'
                         f'{_esc(c.ref)}</text>'
                         f'<text x="0" y="{LABEL_DY+15}" text-anchor="middle" class="val">'
                         f'{_esc(_value_label(c))}</text></g>')
        parts.append('</g>')

        # ---- 受控源：控制连线 + 控制关系文字（都在旋转坐标系之外）
        #   ★ 放在 </g> 之后：控制连线连的是**绝对坐标**（控制端结点位置），
        #     跟着元件一起旋转就会指到错的地方 —— 而那正是"图看着没问题、
        #     控制关系其实是错的"这类最难发现的错位。
        if c.kind in CONTROLLED_KINDS:
            parts.append(_control_annotation(c, circuit, node_pos, (mx, my), accent))
            if show_labels:
                parts.append(
                    f'<text x="{mx:.2f}" y="{my + 30:.2f}" text-anchor="middle" '
                    f'class="ctl">{_esc(_control_label(c))}</text>')

    # ---- 结点圆点（度数 >= 3 才画，与读图判据一致）
    for n, (x, y) in node_pos.items():
        deg = circuit.degree(n)
        if deg >= 3:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{DOT_R}" fill="{wire}" '
                         f'stroke="none"/>')
        elif deg == 2:
            # ★ 这是**装饰性**的节点标记，不是接线圆点。必须显式打上 class，
            #   否则回导解析时它会因为"有填充的小圆"被当成结点圆点，
            #   污染"圆点数量"这个读图依据指标。（真踩过。）
            parts.append(f'<circle class="ca-node-mark" cx="{x:.2f}" cy="{y:.2f}" '
                         f'r="2.4" fill="{wire}" stroke="none"/>')
        want_names = (resolved in ("grid", "radial")) if node_labels is None else node_labels
        if show_labels and want_names:
            parts.append(f'<text x="{x+7:.2f}" y="{y-8:.2f}" class="node">{_esc(n)}</text>')

    parts.append('</svg>')
    return "".join(parts)


def _value_label(c) -> str:
    if c.value is None:
        if c.kind in CONTROLLED_KINDS:
            # 受控源可以没有增益 —— 它可能用自定义表达式描述控制关系
            return "表达式" if (c.ctrl and c.ctrl.expr) else "缺增益"
        return "缺数值"
    kind = c.kind
    if kind == "R":
        return f"{c.value_str()}Ω"
    if kind == "V":
        return f"{c.value_str()}V"
    if kind == "I":
        return f"{c.value_str()}A"
    if kind == "C":
        return f"{c.value_str()}F"
    if kind == "L":
        return f"{c.value_str()}H"
    if kind in CONTROLLED_KINDS:
        # ★ 增益的记号与单位必须与参数表/网表卡一致：同一个 μ 在
        #   图上、在参数表里、在 ngspice 卡上都是同一个符号。
        return f"{GAIN_SYMBOL.get(kind, '')}={c.value_str()}{KIND_UNIT.get(kind, '')}"
    return c.value_str()


def _control_label(c) -> str:
    """受控源的**控制关系**一行文字（画在符号旁）。

    为什么要画出来：菱形符号本身只说明"这是受控源"，**说明不了它受谁控制**。
    而受控源最容易读错、后果最大的正是这一处 —— 控制端接在别的节点上，
    整题答案就换个样，而三法互校与功率守恒都拦不住（三条路径共用同一份 IR）。
    所以图上必须直接写出控制关系，让人一眼能核。
    """
    ctrl = getattr(c, "ctrl", None)
    if ctrl is None:
        return "控制关系未给出"
    if ctrl.expr:
        return f"控制={ctrl.expr}"
    if ctrl.mode == "V" and ctrl.nodes:
        return f"控制 V({ctrl.nodes[0]})−V({ctrl.nodes[1]})"
    if ctrl.mode == "I":
        return f"控制 i({ctrl.sampling})"
    return "控制关系未给出"


def _control_annotation(c, circuit: Circuit,
                        node_pos: dict[str, tuple[float, float]],
                        pos: tuple[float, float], accent: str) -> str:
    """受控源的控制连线（**虚线 + 空心圆点 + 文字**）。

    ★★ 整段都打 ``class="ca-deco"``，这是**必须的**，不是装饰性选择。
      回导解析器把 ``<line>``/``<path>`` 一律当导线；控制连线若不带装饰标记，
      "导出 SVG → 再导入"就会**多出两条导线**、把控制端节点和输出端节点
      额外接起来 —— 拓扑被改掉，而且**不报错**。
      ``_is_decoration`` 认 ``ca-deco``，所以这里的每一笔都带着它。

    控制端坐标拿不到时返回空串 —— 此时图上仍有文字说明
    （见 :func:`_control_label`），不会出现"看起来是受控源、却完全没提
    受谁控制"这种更容易误读的状态。
    """
    ctrl = getattr(c, "ctrl", None)
    if ctrl is None:
        return ""
    mx, my = pos
    out: list[str] = []
    targets: list[tuple[float, float]] = []

    if ctrl.mode == "V" and ctrl.nodes:
        for n in ctrl.nodes:
            q = node_pos.get(n)
            if q:
                targets.append((q[0], q[1]))
    else:
        # 电流控制型：把虚线引到**被采样支路的中点**（那才是"电流所在的地方"。
        # 引到端点会让人以为控制量取自那个节点 —— 电流取的是整条支路）。
        name = ctrl.sampling or ctrl.ref
        try:
            target = circuit.by_ref(name)
        except CircuitError:
            target = None
        if target is not None:
            p1 = node_pos.get(target.nodes[0])
            p2 = node_pos.get(target.nodes[1])
            if p1 and p2:
                targets.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))

    for (qx, qy) in targets:
        out.append(f'<line class="ca-deco" x1="{mx:.2f}" y1="{my:.2f}" '
                   f'x2="{qx:.2f}" y2="{qy:.2f}" fill="none" stroke="{accent}" '
                   'stroke-width="1.3" stroke-dasharray="5 4"/>')
        out.append(f'<circle class="ca-deco" cx="{qx:.2f}" cy="{qy:.2f}" r="3.2" '
                   f'fill="none" stroke="{accent}" stroke-width="1.3"/>')
    return "".join(out)


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_overlay_payload(
    circuit: Circuit,
    image_size: tuple[int, int],
    *,
    viewbox: tuple[float, float, float, float] | None = None,
    dark: bool = False,
) -> dict:
    """给前端的叠图载荷：回绘 SVG + 原图尺寸 + 对齐说明。

    ``viewbox`` 传原图自己的视口，前端把两层放进同一个盒子就能逐像素对齐。
    默认（位图/无 viewBox 的图）用 ``(0, 0, W, H)``。
    """
    if viewbox is None:
        viewbox = (0.0, 0.0, float(image_size[0]), float(image_size[1]))
    dg: dict = {}
    svg = render_svg(circuit, mode="geom", viewbox=viewbox, dark=dark,
                     background=False, node_labels=True, diagnostics=dg)
    return {
        "svg": svg,
        "image_size": {"w": image_size[0], "h": image_size[1]},
        "viewbox": list(viewbox),
        "mode": "geom",
        "overlaps": dg["overlaps"],
        "roundtrip_safe": dg["roundtrip_safe"],
    }
