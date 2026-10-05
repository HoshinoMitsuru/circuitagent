"""几何 -> 连通性 -> IR。SVG 通道与位图通道共用这一层。

这是整个项目**最要紧的一段代码**，因为它落实技能里的第一条禁令：
"不许靠肉眼定连接"。判定规则严格照技能文档的电路图惯例表：

| 几何现象 | 判定 |
|---|---|
| ≥3 条线汇聚处画了**实心圆点** | 相连（最可靠） |
| 十字交叉但**没有**圆点 | 不相连（跨线） |
| T 接无圆点 | 相连（教科书惯例） |
| 端点两者都不满足 | 悬空线头 |

★ 实现上的关键一招，把上面四条变成了几乎不可能写错的形式：

**只把「线段端点」和「圆点圆心」当作候选结点，绝不把几何交点当结点。**
于是——

- 一根导线天然是"导体"：凡是落在它上面的候选结点，全部并到一起。
- **纯 X 跨线**：交点既不是端点、也没有圆点 → **根本不会生成结点** →
  两根线各并各的，天然不相连。不需要写任何"跨线特殊处理"。
- **T 接**：交点恰好是另一根线的端点 → 是候选结点 → 落在长线上 → 并到一起 → 相连。
- **带圆点的交叉**：圆点圆心是候选结点 → 两根线都并到它 → 相连。

四条规则里有三条是"构造的自然结果"，唯一需要显式写的是"候选结点从哪来"。
这类实现比"先求交点再逐点判断"少一个数量级的出错面。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..ir.model import (ALLOWED_KINDS, CONTROLLED_KINDS, CONTROL_MODE,
                        CONTROL_NOTE, Control, Circuit, Component, CircuitError,
                        Evidence)

DEFAULT_TOL = 6.0
#: 端点配对的上限距离。因为槽位还要求"图形必须横跨两个端子"（见 detect_component_slots），
#: 空隙长度已经被图形自身的尺度框住了，这里只作兜底，不必再卡得很死。
#: 早先取 220，结果一根画成 270px 的电阻整个被漏掉 —— 漏掉的后果是它的符号本体
#: 没被剔除、把自己的两个端子短路，而且**不报错**。
DEFAULT_MAX_GAP = 500.0


# ---------------------------------------------------------------- 数据结构


@dataclass
class Geometry:
    """一张电路图里抽出来的全部几何图元。"""

    segments: list[tuple[float, float, float, float]] = field(default_factory=list)
    dots: list[tuple[float, float, float]] = field(default_factory=list)
    rects: list[dict[str, Any]] = field(default_factory=list)
    circles: list[dict[str, Any]] = field(default_factory=list)
    arcs: list[dict[str, Any]] = field(default_factory=list)
    polys: list[list[tuple[float, float]]] = field(default_factory=list)
    texts: list[dict[str, Any]] = field(default_factory=list)
    #: 语义提示：来源方直接给出的元件信息（自家 SVG 会带，外来 SVG 通常没有）
    hints: list[dict[str, Any]] = field(default_factory=list)
    #: 被判为装饰而**没有**计入几何的图元数量（背景块等）。计数是为了让"跳过"可见，
    #: 而不是静默丢弃 —— 假如图里有元件被误判成装饰，这里能看出苗头。
    deco_skipped: int = 0
    #: 参考节点提示：可以是 "x,y" 坐标，也可以是 IR 里的节点名
    ref_hint: str | None = None
    #: 来源通道名，写进 IR 的 origin 便于溯源
    channel: str = "geometry"
    size: tuple[float, float] | None = None
    #: 原图自己的视口 ``(x0, y0, w, h)``。叠图对账要靠它把回绘层与原图对齐。
    viewbox: tuple[float, float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "线段": len(self.segments), "圆点": len(self.dots),
            "矩形": len(self.rects), "圆": len(self.circles),
            "圆弧": len(self.arcs), "折线": len(self.polys),
            "文字": len(self.texts), "语义提示": len(self.hints),
            "装饰跳过": self.deco_skipped,
            "警告": list(self.warnings),
        }


# ---------------------------------------------------------------- 点聚类


class _PointUF:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def add(self, i: int) -> None:
        self.p.setdefault(i, i)

    def find(self, i: int) -> int:
        self.p.setdefault(i, i)
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _cluster_points(points: list[tuple[float, float]], tol: float) -> list[int]:
    """把邻近的点聚成一簇，返回每点的簇号（网格加速，避免 O(n²)）。"""
    uf = _PointUF()
    for i in range(len(points)):
        uf.add(i)
    cell = max(tol, 1e-6)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(points):
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    px, py = points[j]
                    if math.hypot(px - x, py - y) <= tol:
                        uf.union(i, j)
        grid.setdefault((gx, gy), []).append(i)
    # 归一化簇号
    remap: dict[int, int] = {}
    out = []
    for i in range(len(points)):
        r = uf.find(i)
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out


def _cluster_centroids(points, labels) -> list[tuple[float, float]]:
    acc: dict[int, list[tuple[float, float]]] = {}
    for p, l in zip(points, labels):
        acc.setdefault(l, []).append(p)
    return [
        (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
        for _, ps in sorted(acc.items())
    ]


def _point_seg_distance(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


# ---------------------------------------------------------------- 元件本体区域
#
# ★★ 这一段是全项目第二要紧的代码，因为它挡住的是**最阴险的一类错误：短路**。
#
# 问题长这样：电阻符号是一个矩形框，两根导线分别画到框的左右两条边中间。
# 于是——
#   · 左导线的端点**正好落在**框左边缘线段上（距离 0）；
#   · 右导线的端点**正好落在**框右边缘线段上；
#   · 框的四条边首尾相接、彼此相连，是一个闭环导体。
# 一旦连通层把"线段碰到的点都并起来"照单全收，就得到
#   左导线 → 框左边缘 → 框角 → 框右边缘 → 右导线
# 也就是**元件用自己的符号把两端短接了**，两个电学结点被并成一个。
#
# 后果比"报错"严重得多：不报错、静默地少一个结点、解出来的数字还**看着挺合理**。
# 照片里的电阻框、电容的双极板、电感的弧串，全都是这个形状。
#
# 解法不是打补丁，而是把模型摆正：**符号本体不是导线**。
# 既然元件占位（槽位）已经能挑出来，就顺手把槽位轴线周围的"本体圆柱"算出来，
# 凡是**完整落在圆柱内**的线段，一律不进导线图。
#
# 判据用"两端点都落在圆柱内"而不是"碰到圆柱"——这样导线不会被误删：
# 导线的远端伸到结点那边（在圆柱外），只有贴着符号的那一头在圆柱里。


def _slot_axis(slot: dict[str, Any]):
    """槽位轴线：起点 p、单位方向 u、长度 L。"""
    p, q = slot["p"], slot["q"]
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy) or 1.0
    return p, (dx / L, dy / L), L


def _perp_dist(pt, p, u) -> float:
    """点到轴线的**垂直**距离（把 pt-p 投影到法线方向）。"""
    vx, vy = pt[0] - p[0], pt[1] - p[1]
    return abs(vx * (-u[1]) + vy * u[0])


def slot_region(geo: Geometry, slot: dict[str, Any],
                *, margin: float = 6.0) -> dict[str, Any]:
    """算出槽位对应的**元件本体圆柱** ``{p, q, u, L, radius}``。

    半径不是拍脑袋给的，而是**从空隙里那些图形本身量出来的**：
    矩形角点、圆心+半径、圆弧点、垂直于轴线的短极板 —— 取它们到轴线的最大垂直距离。
    这样"本体有多大"是读出来的，不是猜的；也就不会为了保险把半径开得过大，
    顺手把旁边真正的导线吞掉。
    """
    p, u, L = _slot_axis(slot)
    x0, y0, x1, y1 = slot["gapbox"]

    def inside(pt) -> bool:
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    r = 0.0
    r_source = ""
    for rc in geo.rects:
        if inside((rc.get("cx", 0), rc.get("cy", 0))):
            for pt in rc.get("pts", []):
                d = _perp_dist(pt, p, u)
                if d > r:
                    r, r_source = d, "矩形框角点"
    for c in geo.circles:
        if inside((c.get("cx", 0), c.get("cy", 0))):
            d = _perp_dist((c.get("cx", 0), c.get("cy", 0)), p, u) + c.get("r", 0.0)
            if d > r:
                r, r_source = d, "圆形符号外缘"
    for a in geo.arcs:
        pts = [pt for pt in a.get("pts", []) if inside(pt)]
        if pts:
            d = max(_perp_dist(pt, p, u) for pt in pts)
            if d > r:
                r, r_source = d, "圆弧串"
    # 电容的两片极板是 path 线段，不在 rects/circles/arcs 里，得直接量线段端点
    for (ax, ay, bx, by) in geo.segments:
        mxx, myy = (ax + bx) / 2, (ay + by) / 2
        if not inside((mxx, myy)):
            continue
        vl = math.hypot(bx - ax, by - ay)
        if vl <= 1e-9:
            continue
        # 只认"垂直于轴线且明显短于本体"的线段（电容极板）
        if abs((bx - ax) / vl * u[0] + (by - ay) / vl * u[1]) > 0.3:
            continue
        if vl > L * 0.7:
            continue
        d = max(_perp_dist((ax, ay), p, u), _perp_dist((bx, by), p, u))
        if d > r:
            r, r_source = d, "电容极板"

    return {"p": p, "q": slot["q"], "u": u, "L": L,
            "radius": r + margin, "rx": r, "radius_source": r_source or "无图形（保守取 margin）"}


def _inside_region(seg, region: dict[str, Any], tol: float) -> bool:
    """线段是否**完整落在**本体圆柱内（两端点都在）。

    ``t`` 允许向外放宽 ``tol``：符号本体的引线段常常正好从轴线端点起步，
    端点坐标与圆柱端面重合，不吃这点余量会被判成"露出去了"。
    但**绝不能**放得很宽 —— 放太宽就会把贴着本体的真导线也吞掉。
    """
    p, u, L = region["p"], region["u"], region["L"]
    rmax = region["radius"]
    for pt in (seg[0:2], seg[2:4]):
        vx, vy = pt[0] - p[0], pt[1] - p[1]
        t = vx * u[0] + vy * u[1]
        if t < -tol or t > L + tol:
            return False
        if abs(vx * (-u[1]) + vy * u[0]) > rmax:
            return False
    return True



# ---------------------------------------------------------------- 连通性


def group_nodes(
    geo: Geometry,
    *,
    tol: float = DEFAULT_TOL,
    ref_node: str | None = None,
    exclude: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """几何 -> 电学节点分组。**不含元件**，因此可以独立测试。

    抽出来的理由很实际：跨线/圆点/T 接这套判定是全项目最要紧的逻辑，
    它必须能用几个手画的小 SVG 单独验收，而不该被"必须识别出元件"
    这道前置条件挡住 —— 否则核心规则会一直处于"顺带被测"的模糊状态。

    ``exclude`` 是**元件本体圆柱**列表（见 ``slot_region``）。完整落在里面的线段
    会被当作符号本体剔除 —— 不剔的话，电阻框会把自己的两个端子短接起来。
    剔除的线段会如实记在返回值的 ``excluded`` 里，不静默。

    返回的字典包含：
    - ``roots``       电学节点（并查集根）列表
    - ``node_names``  根 -> 节点名（参考节点恒为 "0"）
    - ``root_of``     候选簇 -> 根
    - ``centroids``   候选簇中心坐标
    - ``degree``      每个电学节点接了几条线段
    - ``crossings``   被判为"不相连"的纯交叉点（读图依据）
    - ``dangling``    悬空端点
    - ``excluded``    被当作符号本体剔除的线段
    """
    if ref_node is not None and geo.ref_hint is None:
        geo.ref_hint = str(ref_node)

    segs = [s for s in geo.segments
            if math.hypot(s[2] - s[0], s[3] - s[1]) > tol * 0.5]
    if not segs:
        raise CircuitError(
            "没有从图里提取到任何有效线段。若这是位图，可能是二值化阈值不合适；"
            "若这是 SVG，可能是图形都由曲线/文本构成。"
        )

    # ---- 0. 剔除符号本体（**这一步必须在连通之前**，否则短路已经发生了）
    excluded: list[tuple[float, float, float, float]] = []
    if exclude:
        kept = []
        for s in segs:
            if any(_inside_region(s, rg, tol) for rg in exclude):
                excluded.append(s)
            else:
                kept.append(s)
        segs = kept
        if not segs:
            raise CircuitError(
                "剔除元件本体后一条导线都不剩了。说明元件占位（槽位）把自己的导线"
                "也圈了进去 —— 这是本体半径估大了，需要人工核对元件位置。"
            )

    # ---- 1. 候选结点：只取「线段端点」与「圆点圆心」。
    #      几何交点**一律不生成结点** —— 这是整套判定的命门：
    #      纯 X 跨线处没有候选结点，于是两根线各并各的，天然不相连，
    #      不需要写任何"跨线特判"。
    endpoints: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in segs:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    dot_pts = [(d[0], d[1]) for d in geo.dots]
    all_pts = endpoints + dot_pts
    labels = _cluster_points(all_pts, tol)
    end_cluster = labels[:len(endpoints)]
    dot_cluster = labels[len(endpoints):]
    centroids = _cluster_centroids(all_pts, labels)
    n_clusters = len(centroids)

    has_dot: set[int] = set(dot_cluster)
    is_endpoint_of: dict[int, set[int]] = {}
    for si, cl in enumerate(end_cluster):
        is_endpoint_of.setdefault(cl, set()).add(si)

    # ---- 2. 导体并集：每条线段把它上面所有的候选簇并到一起
    uf = _PointUF()
    for ci in range(n_clusters):
        uf.add(ci)
    on_seg: list[list[int]] = []
    for si, (x1, y1, x2, y2) in enumerate(segs):
        hits = [
            ci for ci, (cx, cy) in enumerate(centroids)
            if _point_seg_distance(cx, cy, x1, y1, x2, y2) <= tol
        ]
        on_seg.append(hits)
        for ci in hits[1:]:
            uf.union(hits[0], ci)
    root_of = [uf.find(ci) for ci in range(n_clusters)]

    node_segs: dict[int, set[int]] = {}
    for ci in range(n_clusters):
        node_segs.setdefault(root_of[ci], set())
    for si, hits in enumerate(on_seg):
        for ci in hits:
            node_segs.setdefault(root_of[ci], set()).add(si)

    # ---- 3. 跨线识别（用于**报告**，不用于判定）
    crossings: list[dict[str, Any]] = []
    for ci, (cx, cy) in enumerate(centroids):
        if ci in has_dot:
            continue
        touching = on_seg_of_cluster(ci, on_seg)
        if len(touching) < 2:
            continue
        if not [s for s in touching if ci in is_endpoint_of.get(ci, ())]:
            crossings.append({
                "x": round(cx, 1), "y": round(cy, 1),
                "segments": sorted(touching),
                "note": "两线交叉且无圆点 -> 判为不相连（跨线）。"
                        "这是「交点不生成结点」的自然结果，不是特判。",
            })

    # ---- 4. 命名电学节点
    roots = sorted(node_segs.keys(), key=lambda r: (-len(node_segs[r]), r))
    if not roots:
        raise CircuitError("几何解析后没有得到任何电学节点")

    node_names: dict[int, str] = {}
    ref_root: int | None = None
    ref_source = "auto_max_degree"

    # 4a. 语义提示带的 IR 节点名（自家 SVG 回导走这条，可无损还原名字与参考节点）
    if geo.hints:
        for h in geo.hints:
            for irname, pos in ((h.get("a"), h.get("p1")), (h.get("b"), h.get("p2"))):
                if irname is None or not pos:
                    continue
                r = _root_at(pos, centroids, root_of, tol)
                if r is None:
                    continue
                if r in node_names and node_names[r] != str(irname):
                    geo.warnings.append(
                        f"节点名冲突：几何上同一个结点同时被提示为 "
                        f"{node_names[r]!r} 与 {irname!r}，说明图上的导线被合并了"
                        "（常见于两次回绘叠加）。已保留先出现的名字。"
                    )
                else:
                    node_names[r] = str(irname)
        if "0" in node_names.values():
            ref_root = next(r for r, n in node_names.items() if n == "0")
            ref_source = "svg_semantic_hints"

    # 4b. 显式提示（坐标 "x,y" 或节点名）
    if ref_root is None and geo.ref_hint:
        r = _root_at_hint(geo.ref_hint, node_names, roots, centroids, root_of, tol)
        if r is not None:
            ref_root, ref_source = r, "hint"

    if ref_root is None:
        ref_root = roots[0]
        ref_source = "auto_max_degree"
        geo.warnings.append(
            "没有找到地/参考节点的标记，已把连接数最多的节点当作参考节点。"
            "**请在界面上确认参考节点** —— 选错会让所有电压读数整体平移一个常数。"
        )

    # 4c. 补齐其余节点名，避开已占用的名字
    taken = set(node_names.values())
    counter = 1
    for r in roots:
        if r in node_names and r != ref_root:
            continue
        if r == ref_root:
            continue
        while f"N{counter}" in taken:
            counter += 1
        node_names[r] = f"N{counter}"
        taken.add(f"N{counter}")
        counter += 1
    # 参考节点必须叫 "0"；若已被别的结点占用，就把那个改掉
    for r in [r for r, n in node_names.items() if n == "0" and r != ref_root]:
        while f"N{counter}" in taken:
            counter += 1
        node_names[r] = f"N{counter}"
        taken.add(f"N{counter}")
        counter += 1
    node_names[ref_root] = "0"

    # ---- 5. 度数 + 悬空端点
    degree: dict[int, int] = {}
    for si, hits in enumerate(on_seg):
        seen = {root_of[ci] for ci in hits}
        for r in seen:
            degree[r] = degree.get(r, 0) + 1
    dangling = [
        {"node": node_names[r], "degree": degree.get(r, 0),
         "x": round(_centroid_of_root(r, root_of, centroids)[0], 1),
         "y": round(_centroid_of_root(r, root_of, centroids)[1], 1)}
        for r in roots if degree.get(r, 0) < 2 and r != ref_root
    ]

    return {
        "segments": segs, "roots": roots, "node_names": node_names,
        "root_of": root_of, "centroids": centroids, "degree": degree,
        "crossings": crossings, "dangling": dangling,
        "ref_root": ref_root, "ref_source": ref_source,
        "n_dots": len(geo.dots), "on_seg": on_seg,
        "excluded": excluded,
    }


def build_topology(
    geo: Geometry,
    *,
    tol: float = DEFAULT_TOL,
    max_gap: float = DEFAULT_MAX_GAP,
    name: str = "circuit",
    ref_node: str | None = None,
    slots: list[dict[str, Any]] | None = None,
) -> tuple[Circuit, dict[str, Any]]:
    """几何 -> (IR, 读图依据报告)。

    ``slots`` 可由上游预先算好并**填好数值**（SVG 通道会先从 ``<text>`` 里读数），
    不传则在此内部按几何特征现算。

    ★ 顺序很关键：**先挑元件占位（槽位）→ 算出本体圆柱 → 再建连通图**。
    反过来做（先连通再挑元件）会让符号本体参与导电，电阻框把两个端子短接，
    两个结点被并成一个，而且不报错。详见 ``slot_region`` 那段注释。
    """
    # ---- 甲：元件占位先算（元件识别与"本体剔除"都要用它）
    if slots is None:
        slots = detect_component_slots(geo, tol=tol, max_gap=max_gap)
    regions = [slot_region(geo, sl) for sl in slots]

    # ---- 乙：连通性（符号本体已剔除）
    g = group_nodes(geo, tol=tol, ref_node=ref_node, exclude=regions)
    segs = g["segments"]
    node_names = g["node_names"]
    centroids = g["centroids"]
    root_of = g["root_of"]
    roots = g["roots"]

    report: dict[str, Any] = {
        "geometry": geo.describe(),
        "tol": tol,
        "max_gap": max_gap,
        "hints_used": bool(geo.hints),
        "crossings": g["crossings"],
        "dangling_endpoints": g["dangling"],
        "node_count": len(roots),
        "segment_count": len(segs),
        "ref_node_source": g["ref_source"],
        "dot_count": g["n_dots"],
        "symbol_body_segments_removed": len(g["excluded"]),
        "slot_count": len(slots),
        "slot_body_radius": [round(rg["rx"], 1) for rg in regions],
        #: 原图视口，前端叠图要靠它把回绘层与原图对齐
        "viewbox": list(geo.viewbox) if geo.viewbox else None,
    }
    if g["excluded"]:
        geo.warnings.append(
            f"有 {len(g['excluded'])} 条线段落在元件本体内，已按**符号本体**处理、"
            "不计入导线。这一步是为了防止电阻框/电容极板/电感弧串把自己的两个端子"
            "短接起来（符号本体是闭环导体，会让两个结点被并成一个且不报错）。"
        )


    # ---- 5. 悬空端点分类的细节已在 group_nodes 里算好（report 上方已取用）
    if g["ref_source"] == "auto_max_degree":
        geo.warnings.append(
            "没有找到地/参考节点的标记，已把连接数最多的那个节点当作参考节点"
            f"（{node_names[g['ref_root']]}）。**请在界面上确认**。"
        )

    # ---- 6. 元件
    comps: list[Component] = []
    slot_report: list[dict[str, Any]] = []
    used_refs: list[str] = []
    #: 被跳过的元件（末端落不到结点 / 两端落在同一结点=短路），用于把失败原因讲清楚
    skipped: list[dict[str, Any]] = []

    if geo.hints:
        for h in geo.hints:
            kind = str(h.get("kind", "")).upper()
            # ★ 这里**不能**写死成 R/V/I/C/L。
            #   写死之后的症状极其隐蔽：受控源被一条 warning 悄悄跳过，
            #   于是网表里少一个元件、剩下的电路照样可解，三法互校也照样全过
            #   （三条路径共用同一份"少了元件"的 IR）—— 用户拿到一个**看似合理**
            #   的答案。所以判据必须跟着 IR 的权威定义走，加一种元件这里自动跟上。
            if kind not in ALLOWED_KINDS:
                geo.warnings.append(f"语义提示里的元件类型 {kind!r} 不认识，已跳过")
                continue
            # ★ 用**坐标**去定位几何结点，用**IR 节点名**去命名 —— 两者分工明确。
            #   直接拿 IR 节点名当几何结点名会错，因为几何层的名字是本层自己起的。
            a = _name_at(h.get("p1"), centroids, root_of, node_names, tol)
            b = _name_at(h.get("p2"), centroids, root_of, node_names, tol)
            if a is None or b is None:
                skipped.append({"ref": h.get("ref"), "why": "endpoint_off_geometry",
                                "detail": f"端点坐标 {h.get('p1')} / {h.get('p2')} "
                                          "落不到任何几何结点上"})
                geo.warnings.append(
                    f"语义提示 {h.get('ref')} 的端点坐标落不到任何几何结点上，已跳过"
                )
                continue
            if a == b:
                # ★ 这是**短路的症状**：元件两端被判定为同一个结点。
                #   最可能的成因是符号本体（电阻框/极板/弧串）参与了导电。
                skipped.append({"ref": h.get("ref"), "why": "short_circuit",
                                "detail": f"两端都落在结点 {a} 上"})
                geo.warnings.append(
                    f"**短路嫌疑**：语义提示 {h.get('ref')} 的两端被判为同一个结点 {a}，"
                    "已跳过。通常说明该元件的符号本体被当成了导线（本体把自己的两个"
                    "端子连起来了），请人工核对这个元件的两个端点位置。"
                )
                continue

            # ---- 受控源：控制支路必须一起还原，否则这个元件**根本建不出来**
            #   （IR 里受控源没有控制支路是硬错，见 Component.__post_init__）。
            #   所以控制关系缺失时不能"建一个半成品"，只能明确跳过并说清怎么办 ——
            #   比整个解析崩掉好，也比静默建出一个方程不对的元件好。
            ctrl, ctl_why = (
                _control_from_hint(h, kind, centroids, root_of, node_names, tol)
                if kind in CONTROLLED_KINDS else (None, None))
            if kind in CONTROLLED_KINDS and ctrl is None:
                skipped.append({"ref": h.get("ref"), "why": "control_undefined",
                                "detail": ctl_why})
                geo.warnings.append(
                    f"**{h.get('ref')}（{kind}，{CONTROL_NOTE.get(kind, kind)}）"
                    f"的控制关系没有读出来**：{ctl_why}。"
                    "受控源和独立源的电路方程完全不同，控制支路不敢猜 —— "
                    "请把它的控制端（或它采样的那条支路）补上再重新解析。"
                )
                continue

            ref = str(h.get("ref") or Circuit.auto_ref(kind, used_refs))
            if ref in used_refs:
                ref = Circuit.auto_ref(kind, used_refs)
            used_refs.append(ref)
            comps.append(Component(
                ref=ref, kind=kind, nodes=(a, b),
                value=h.get("value"),
                ctrl=ctrl,
                evidence=Evidence(
                    source="exact", confidence=1.0,
                    detail="来自 SVG 语义标记（data-ca-*）：拓扑与两端次序均无损读取"),
                geom={"p1": h.get("p1"), "p2": h.get("p2")},
            ))
        report["component_source"] = "svg_semantic_hints"
    else:
        for sl in slots:
            kind, conf, detail = classify_slot(geo, sl)
            ref = Circuit.auto_ref(kind, used_refs)
            used_refs.append(ref)
            na = _name_at(sl["p"], centroids, root_of, node_names, tol)
            nb = _name_at(sl["q"], centroids, root_of, node_names, tol)
            if na is None or nb is None or na == nb:
                skipped.append({
                    "ref": ref,
                    "why": ("endpoint_off_geometry" if (na is None or nb is None)
                            else "short_circuit"),
                    "detail": f"两端结点 {na} / {nb}",
                })
                continue
            comps.append(Component(
                ref=ref, kind=kind, nodes=(na, nb),
                value=sl.get("value"),
                evidence=Evidence(source="cv", confidence=conf, detail=detail),
                geom={"p1": list(sl["p"]), "p2": list(sl["q"])},
            ))
            slot_report.append({**{k: v for k, v in sl.items() if k != "gapbox"},
                                "kind": kind, "confidence": conf, "detail": detail})
        report["component_source"] = "geometric_inference"
        report["slots"] = slot_report

    report["skipped_components"] = skipped
    n_short = sum(1 for s in skipped if s["why"] == "short_circuit")

    if not comps:
        if n_short:
            raise CircuitError(
                f"识别出的 {n_short} 个元件**两端都被判成了同一个结点**（短路），"
                "因此一个都没能建立起来。最可能的原因是符号本体被当成了导线 —— "
                "电阻框、电容极板、电感弧串都是闭环导体，一旦参与连通就会把元件两端接在一起。"
                "本层已经把落入元件本体的线段剔除，所以出现这种情况说明本体的位置/大小"
                "估得不准。**请在界面上核对元件图形与导线端点位置。**"
            )
        raise CircuitError(
            "从图里没有识别出任何元件。可能原因：图里没有元件符号、"
            "或者元件符号不是本工具认识的画法。"
            "**这种情况需要人工在界面上补元件，不能靠猜。**"
        )

    # ---- 被跳过的元件：**每一条原因都要报出来**
    #   ★ 这一段以前只报 `short_circuit` 一种。其余几种（端点落不到结点、
    #     受控源控制关系没读出来、几何槽位挑不出来）被写进 `report["skipped_components"]`
    #     就没人再读了 —— 而那个字段**全项目没有任何消费方**。
    #     症状是：图里少了一个元件，网表照样生成、三法照样一致、报告照样"通过"，
    #     因为三条求解路径共用的是同一份**残缺**的 IR。用户拿到的是一个
    #     基于残缺电路算出来的、看起来完全合理的答案。这是本项目最不能出现的一类错。
    #   所以这里按原因分类逐条列出，并且**明说后果**。
    if skipped:
        REASON_NOTE = {
            "short_circuit":
                "两端被判为同一个结点（短路嫌疑）—— 多半是符号本体被当成了导线",
            "endpoint_off_geometry":
                "端点坐标落不到任何结点上 —— 该元件那一端没接到导线/别的端子上",
            "control_undefined":
                "受控源的控制关系没读出来 —— 受控源与独立源的方程完全不同，控制支路不敢猜",
            "no_slot":
                "图上找不到这个元件的符号槽位（画法不认识，或者符号被别的东西压住了）",
        }
        by_reason: dict[str, list[str]] = {}
        for s in skipped:
            by_reason.setdefault(s["why"], []).append(str(s.get("ref") or "?"))
        lines = []
        for why, refs in sorted(by_reason.items()):
            lines.append(f"{len(refs)} 个（{', '.join(refs)}）：{REASON_NOTE.get(why, why)}")
        geo.warnings.append(
            f"**有 {len(skipped)} 个元件没能建立起来，它们不在网表里** ——\n  "
            + "\n  ".join(lines)
            + "\n这几条支路一律**没有被偷偷换掉或补上**，就是缺了。"
              "因此下面算出来的结果对应的是**比原图少了几条支路**的电路："
              "三法互校与功率守恒都拦不住这类错（三条路径共用同一份残缺的 IR）。"
              "**请先在图/表上把这几个元件补齐，再采信任何数值。**"
        )

    circ = Circuit(
        name=name, components=comps, ref_node="0",
        diagnostics=[{"kind": "topology_report", "text": f"几何解析：{len(segs)} 条线段、"
                      f"{len(geo.dots)} 个圆点、{len(roots)} 个电学节点；"
                      f"元件识别来源：{report['component_source']}"}],
        origin={"channel": geo.channel,
                "geometry_report": report, "warnings": list(geo.warnings)},
    )
    return circ, report


def on_seg_of_cluster(ci: int, on_seg: list[list[int]]) -> list[int]:
    """哪些线段在该候选结点上（含穿过与终止）。"""
    return [si for si, hits in enumerate(on_seg) if ci in hits]


def node_at(g: dict[str, Any], x: float, y: float, tol: float = DEFAULT_TOL) -> str | None:
    """查"坐标 (x,y) 处属于哪个电学节点"。

    这是验收跨线判定的正确姿势：不要比节点**总数**（容易数错，
    毕竟一根导线自己就把两个端点并成一个节点），而要比
    "某个坐标上的点，和另一个坐标上的点，是不是同一个节点"。
    """
    best, bestd = None, tol
    for ci, (cx, cy) in enumerate(g["centroids"]):
        d = math.hypot(cx - x, cy - y)
        if d <= bestd:
            best, bestd = ci, d
    if best is None:
        return None
    return g["node_names"].get(g["root_of"][best])


def _root_at(pos, centroids, root_of, tol: float) -> int | None:
    """把坐标 ``(x, y)`` 落到最近的几何结点上（容差外返回 None）。"""
    if not pos:
        return None
    try:
        px, py = float(pos[0]), float(pos[1])
    except (TypeError, ValueError, IndexError):
        return None
    best, bestd = None, tol
    for ci, (cx, cy) in enumerate(centroids):
        d = math.hypot(cx - px, cy - py)
        if d <= bestd and (best is None or d < bestd):
            best, bestd = ci, d
    return root_of[best] if best is not None else None


def _name_at(pos, centroids, root_of, node_names, tol) -> str | None:
    r = _root_at(pos, centroids, root_of, tol)
    return node_names.get(r) if r is not None else None


def _control_from_hint(h, kind, centroids, root_of, node_names, tol
                       ) -> tuple["Control | None", str]:
    """把语义标记里的 ``ctrl`` 还原成 ``Control``。返回 ``(control, 失败原因)``。

    ★ **控制端走坐标、被采样支路走位号**，这不是随手定的：
      * 电压控制的控制端是"图上另外两个位置"→ 只有位置是几何事实，
        结点名是本层算出来的（与元件自己两个端子完全同一套处理）。
        让上游直接写结点名就等于把命名权收走了，一旦图上的接法改了，
        标记里的名字还停在旧值上 —— 而它长得完全正常，没人会发现。
      * 电流控制的控制量是"某条支路上的电流"，位号本来就是**语义身份**
        （与 ``data-ca-ref`` 同性质），不是几何算出来的，所以可以直接带。

    失败一律返回原因字符串，**不抛异常**：一个元件的控制关系缺失
    不该让整张图的解析崩掉 —— 但也不能默默少一个元件（那会给出一个
    看起来合理的错答案），所以由调用方明确跳过并留痕。
    """
    ctl = h.get("ctrl") or {}
    mode = str(ctl.get("mode") or CONTROL_MODE.get(kind) or "").strip().upper()
    if mode not in ("V", "I"):
        return None, "标记里没有说明它是电压控制还是电流控制（data-ca-ctrl-mode）"
    if mode != CONTROL_MODE.get(kind):
        return None, (f"标记说它是{'电压' if mode == 'V' else '电流'}控制，"
                      f"但 {kind} 这种受控源按定义是"
                      f"{'电压' if CONTROL_MODE.get(kind) == 'V' else '电流'}控制 —— "
                      "两者矛盾，说明标记写错了")
    expr = str(ctl.get("expr") or "").strip()

    if mode == "V":
        na = _name_at(ctl.get("cp1"), centroids, root_of, node_names, tol)
        nb = _name_at(ctl.get("cp2"), centroids, root_of, node_names, tol)
        if na is None or nb is None:
            return None, ("控制端坐标 " + repr(ctl.get("cp1")) + " / " + repr(ctl.get("cp2"))
                          + " 落不到任何几何结点上（漏画了控制端，或者控制端没接在图上的结点）")
        if na == nb:
            return None, f"控制端的两点都落在结点 {na} 上 —— 控制量恒为 0，多半是画错了"
        return Control(mode="V", nodes=(na, nb), expr=expr), ""

    ref = str(ctl.get("ref") or "").strip()
    if not ref:
        return None, "没有给出被采样的支路位号（data-ca-ctrl-ref）"
    sense = str(ctl.get("sense") or "").strip()
    return Control(mode="I", ref=ref, expr=expr, sense_ref=sense), ""


def _centroid_of_root(r, root_of, centroids) -> tuple[float, float]:
    pts = [c for ci, c in enumerate(centroids) if root_of[ci] == r]
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _root_at_hint(hint, node_names, roots, centroids, root_of, tol) -> int | None:
    """参考节点提示：``"x,y"`` 坐标，或 IR 里的节点名。"""
    s = str(hint)
    if "," in s:
        try:
            x, y = (float(v) for v in s.split(","))
        except ValueError:
            return None
        return _root_at((x, y), centroids, root_of, tol)
    for r, n in node_names.items():
        if n == s:
            return r
    return None


# ---------------------------------------------------------------- 元件槽位


def detect_component_slots(geo: Geometry, *, tol: float = DEFAULT_TOL,
                           max_gap: float = DEFAULT_MAX_GAP) -> list[dict[str, Any]]:
    """找"元件槽位"：两个**面对面**的自由端点，中间隔着一件**横跨两端**的图形。

    四条判据（缺一不可）：

    1. 两点距离落在 ``(tol, max_gap]``；
    2. 两点各自所属的线段都**朝着远离对方的方向延伸**（点积 < -0.7），
       即两根线是"头对头"而不是"并排"；
    3. 空隙里得有图形（矩形/圆/弧/垂直极板），否则只是断线；
    4. ★ **那件图形必须同时够得着两个端点**（``_glyph_touch``）。

    第 4 条是后加的，加它的原因很实在：原先只有第 3 条时，判据退化成
    "两点之间的大框里碰巧有个图形"，于是 ``max_gap`` 必须卡得很死（220px），
    否则两根不相干的线头也能配上。而元件在图上画多长根本没准 ——
    真实电路里一根 270px 的画法就足以让 ``max_gap`` 卡掉整个元件，
    **元件没被挑出来的后果不是报错，是它的符号本体没被剔除、把自己的两端短路**。
    加上"图形要够得着两端"之后，空隙长度被图形本身的尺度自然框住了：
    图形撑到哪里，槽位就到哪里。于是 ``max_gap`` 可以放宽到纯粹兜底的大小。

    这样挑出来的槽位，正是元件符号在电路图里"吃掉"的那一段。
    """
    segs = geo.segments
    if not segs:
        return []

    # 自由端点：没有被别的线段端点"接上"的端点
    ends: list[tuple[int, int, tuple[float, float]]] = []
    for si, (x1, y1, x2, y2) in enumerate(segs):
        ends.append((si, 0, (x1, y1)))
        ends.append((si, 1, (x2, y2)))
    n = len(ends)
    connected = [False] * n
    for i in range(n):
        si, ei, (x, y) = ends[i]
        for j in range(n):
            if i == j:
                continue
            sj, ej, (xx, yy) = ends[j]
            if math.hypot(xx - x, yy - y) <= tol:
                connected[i] = True
                break
    free = [ends[i] for i in range(n) if not connected[i]]

    def into_seg(si: int, ei: int) -> tuple[float, float]:
        """该端点**朝线段内部**的单位方向，也就是"这根线往哪边跑"。

        ★★ 这里踩过一个大坑，写下来免得再踩：**两端的判据符号是相反的。**

        设槽位轴线方向 ``u`` 为 p→q。物理事实是：
        · p 端的导线往 **−u** 方向跑（背离 q）；
        · q 端的导线往 **+u** 方向跑（背离 p）。

        我第一版把两端都写成"内向方向与 u 相反"（都要求点积 < −0.7），
        结果——
        · 真槽位（两端背向）**必然被拒**：q 端算出来的点积是 +1；
        · 假槽位（两根线同向、都在同一侧，比如 R1 的端子和 R3 的端子）
          **反而被接受**：两端的点积都是 −1。

        于是槽位检测一直在挑"最不像元件的那种配对"，把真槽位的端点占掉，
        真元件反而因为本体没被剔除而继续短路。教训是：判据符号写反不一定会
        报错或挑不出东西，**它可能挑出一堆看起来合理的垃圾**。
        """
        x1, y1, x2, y2 = segs[si]
        if ei == 0:
            vx, vy = x2 - x1, y2 - y1
        else:
            vx, vy = x1 - x2, y1 - y2
        L = math.hypot(vx, vy) or 1.0
        return vx / L, vy / L

    slots: list[dict[str, Any]] = []
    used: set[int] = set()
    for i in range(len(free)):
        if i in used:
            continue
        si, ei, p = free[i]
        op = into_seg(si, ei)
        for j in range(i + 1, len(free)):
            if j in used:
                continue
            sj, ej, q = free[j]
            if si == sj:
                continue
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if not (tol < d <= max_gap):
                continue
            ux, uy = (q[0] - p[0]) / d, (q[1] - p[1]) / d
            oq = into_seg(sj, ej)
            # 两根线都要**背离空隙**：p 端朝 −u、q 端朝 +u。
            # 注意两端的符号相反（详见 into_seg 的注释，这里写错过一次）。
            if (op[0] * ux + op[1] * uy) > -0.7:
                continue
            if (oq[0] * ux + oq[1] * uy) < 0.7:
                continue
            box = (min(p[0], q[0]) - 6, min(p[1], q[1]) - 6,
                   max(p[0], q[0]) + 6, max(p[1], q[1]) + 6)
            items = _glyph_items(geo, box, (ux, uy))
            # ★ 判据是"**同一件**图形横跨两个端子"，不是"两个端子各碰到一件图形"。
            #   后者会放进这种假槽位：R1 的端子碰到 R1 的框、R3 的端子碰到 R3 的框边 ——
            #   两个端子各碰各的，配对成功，于是 R1 和 R3 之间被凭空造出一个元件，
            #   而且把两个**真**槽位的端点都占掉，导致 R1/R3 的本体没能被剔除、继续短路。
            #   （这个假槽位真的出现过，一次吃掉两个真元件。）
            if not _one_glyph_spans(items, p, q, tol):
                continue
            used.add(i)
            used.add(j)
            slots.append({"p": p, "q": q, "gap": d,
                          "p_seg": si, "q_seg": sj, "gapbox": box})
            break
    return slots


def _glyph_samples(kind: str, g: Any) -> list[tuple[float, float]]:
    """取图形上的**样本点**，只用于判断"两件图形挨不挨着"。

    用样本点而不是精确最近距离，是因为这里只需要一个**量级上**的判断
    （同一符号内部的图形一定挨得很近，不同元件的图形一定离得很远），
    而精确的图形-图形最近距离要写四类图形两两组合，不值当。
    """
    if kind == "seg":
        return [(g[0], g[1]), (g[2], g[3])]
    if kind == "circle":
        cx, cy, r = g.get("cx", 0), g.get("cy", 0), g.get("r", 0)
        return [(cx, cy)] + [(cx + r * math.cos(a), cy + r * math.sin(a))
                             for a in (0, 1.57, 3.14, 4.71)]
    if kind == "rect":
        return [(float(pt[0]), float(pt[1])) for pt in (g.get("pts") or [])]
    return [(float(pt[0]), float(pt[1])) for pt in (g.get("pts") or [])]


def _one_glyph_spans(items: list[tuple[str, Any]], p, q, tol: float,
                     *, chain: float | None = None) -> bool:
    """是否**同一件（连通的）图形**同时够得着 p 和 q。

    先把互相挨得很近的图形并成一组（电阻框的 4 条边、电感的 4 段弧、
    电容的两片极板，各自都会并成一件），再看哪一组能同时碰到两个端子。

    并组阈值 ``chain`` 之所以能取到 ``3*tol`` 这种"宽松"的数，是因为它比的是
    **图形与图形**的距离，而同一符号内部的图形间隙（电容两极板 8px 上下）
    远小于不同元件之间的距离（动辄上百像素）。要出错的余地很小。
    """
    if not items:
        return False
    thr = chain if chain is not None else max(3.0 * tol, 12.0)
    n = len(items)
    uf = _PointUF()
    for i in range(n):
        uf.add(i)
    samples = [_glyph_samples(k, g) for k, g in items]
    for i in range(n):
        for j in range(i + 1, n):
            if min(math.hypot(a[0] - b[0], a[1] - b[1])
                   for a in samples[i] for b in samples[j]) <= thr:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    for idx in groups.values():
        group = [items[i] for i in idx]
        if _glyph_touch(group, p, tol) and _glyph_touch(group, q, tol):
            return True
    return False


def _glyph_items(geo: Geometry, box, u) -> list[tuple[str, Any]]:
    """空隙里的**图形元素**，也就是可能是元件本体的那些东西。

    只有四类，且第 4 类要滤掉与轴线**平行**的线段 —— 平行于轴线的线段就是导线本身，
    把它算成图形的话，"空隙里有图形"这条就恒真了。

    第 4 类必不可少：电容的两片极板是 ``path`` 画出来的线段，
    它既不是 rect 也不是 circle 也不是 arc。早先只看前三类，
    电容的槽位永远挑不出来（而且不报错，就是静默地少一个元件）。
    """
    x0, y0, x1, y1 = box

    def inside(pt) -> bool:
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    items: list[tuple[str, Any]] = []
    for r in geo.rects:
        if inside((r.get("cx", 0), r.get("cy", 0))):
            items.append(("rect", r))
    for c in geo.circles:
        if inside((c.get("cx", 0), c.get("cy", 0))):
            items.append(("circle", c))
    for a in geo.arcs:
        if any(inside(pt) for pt in a.get("pts", [])):
            items.append(("arc", a))
    for s in geo.segments:
        if not inside(((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)):
            continue
        vl = math.hypot(s[2] - s[0], s[3] - s[1])
        if vl <= 1e-9:
            continue
        if abs((s[2] - s[0]) / vl * u[0] + (s[3] - s[1]) / vl * u[1]) < 0.3:
            items.append(("seg", s))
    return items


def _glyph_touch(items: list[tuple[str, Any]], pt, tol: float) -> bool:
    """某件图形是否**够得着**这个端点（距离 ≤ tol）。"""
    px, py = pt
    for kind, g in items:
        if kind == "seg":
            if _point_seg_distance(px, py, g[0], g[1], g[2], g[3]) <= tol:
                return True
        elif kind == "circle":
            if math.hypot(px - g.get("cx", 0), py - g.get("cy", 0)) <= g.get("r", 0) + tol:
                return True
        elif kind == "rect":
            pts = g.get("pts") or []
            for i in range(len(pts)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                if _point_seg_distance(px, py, a[0], a[1], b[0], b[1]) <= tol:
                    return True
        elif kind == "arc":
            pts = g.get("pts") or []
            for i in range(len(pts) - 1):
                if _point_seg_distance(px, py, pts[i][0], pts[i][1],
                                       pts[i + 1][0], pts[i + 1][1]) <= tol:
                    return True
            if any(math.hypot(px - p[0], py - p[1]) <= tol for p in pts):
                return True
    return False


def classify_slot(geo: Geometry, slot) -> tuple[str, float, str]:
    """按空隙里的图形特征给元件定种类，同时**如实给出置信度**。

    置信度不是装饰：低于 model.CONFIDENCE_GATE 的会被 WebUI 强制走人工确认。
    """
    x0, y0, x1, y1 = slot["gapbox"]
    p, q = slot["p"], slot["q"]
    ux, uy = (q[0] - p[0]), (q[1] - p[1])
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux / L, uy / L

    inside = lambda pt: x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    rects = [r for r in geo.rects if inside((r.get("cx", 0), r.get("cy", 0)))]
    circles = [c for c in geo.circles if inside((c.get("cx", 0), c.get("cy", 0)))]
    arcs = [a for a in geo.arcs if any(inside(pt) for pt in a.get("pts", []))]
    # 垂直于槽位轴的短线段（电容的两个极板）
    perp = []
    for (ax, ay, bx, by) in geo.segments:
        mx, my = (ax + bx) / 2, (ay + by) / 2
        if not inside((mx, my)):
            continue
        vx, vy = bx - ax, by - ay
        vl = math.hypot(vx, vy) or 1.0
        if abs((vx / vl) * ux + (vy / vl) * uy) < 0.3 and vl < L * 0.7:
            perp.append((ax, ay, bx, by))

    if rects:
        return "R", 0.9, "空隙里有矩形框 —— 国标电阻符号（置信度较高，但不同画法可能用矩形表示其他器件）"
    if len(arcs) >= 2:
        return "L", 0.8, f"空隙里有 {len(arcs)} 段圆弧 —— 电感符号"
    if len(perp) == 2:
        return "C", 0.8, "空隙里有两片垂直于轴线的短极板 —— 电容符号"
    if circles:
        # 圆里既有 +/− 也可能是箭头，本层无法可靠区分，如实降置信度
        return "V", 0.55, (
            "空隙里有圆形符号。圆形既可能是电压源（内有 +/−）也可能是电流源（内有箭头），"
            "本层按电压源处理但置信度只有 0.55 —— **必须人工确认**。")
    return "R", 0.35, "空隙里有图形但无法归类，先按电阻占位，置信度低 —— **必须人工确认**"
