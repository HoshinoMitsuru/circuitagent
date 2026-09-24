"""位图通道的导线追踪：从墨迹里找出线段、结点、以及元件接在哪两个结点上。

**这一段是整个位图通道里最容易出错、也最容易被糊弄过去的部分。**
矢量通道有明确的线段端点可以依赖；位图只有一坨像素，"哪根线连到哪"
必须靠几何算出来。所以这里严格照搬矢量通道那两条铁律：

1. **只把「线段端点」和「圆点圆心」当候选结点，几何交点不生成结点。**
   于是两根线十字交叉时天然不相连 —— 这是构造的自然结果，不是特判。
   一个十字交叉在几何上是"两条线段内部相交"，它不产生新结点，
   所以四条线段各走各的，交叉点两侧并不导通。
2. **T 形接入（一端点落在另一条线的内部）判为相连**，即使没有圆点。
   这是教科书制图惯例。

再加上第三条（从矢量通道的血泪教训）：

3. **符号本体不参与连通。** 元件符号的闭环导体（电阻框、电容极板、
   电感弧串）若参与连通，就会把元件两端短接。所以顺序固定为
   **先挑槽位 → 先剔除落在本体区域内的线段 → 再建连通图**。
   本模块用 ``slots`` 传入的 ``slot_region()`` 来做这件事。

线段怎么来：用霍夫变换（``cv2.HoughLinesP``）在"剔除符号本体后的墨迹"上
抽直线段。选它而不是骨架化，是因为骨架化会把交叉点变成 4 度顶点，
于是"十字交叉不相连"这条规则就得靠特判恢复；而霍夫给的是**直线段**，
线段与线段的内部相交天然不产生结点，规则是免费得到的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .symbols import (SymbolReport, SymbolSlot, ink_mask,  # noqa: F401
                      estimate_stroke_half,
                      probe as _sym_probe)

# ---------------------------------------------------------------- 参数

#: 霍夫抽线的参数。取值偏"宁可多抽短段、由后面的合并去收"：
#: 抽漏一根线会直接丢掉一条连接（拓扑错），多抽一段只是多算一点。
HOUGH_THRESHOLD = 18
HOUGH_MIN_LINE = 14
HOUGH_MAX_GAP = 6

#: 线段合并：夹角小于此值（度）且垂距小于 tol 的两段视为同一条直线。
MERGE_ANGLE_DEG = 8.0
MERGE_PERP_TOL = 6.0
#: 两段在同一直线上、但投影间隔小于此值也算同一条（中间被符号/文字打断）。
MERGE_JOIN_GAP = 12.0

#: 端点吸附半径：两个端点在这个距离内视为同一个结点。
#: 与矢量通道的 tol 默认值（6px）同量级，但位图有 1~2px 的骨架抖动，
#: 放宽到 8px。
NODE_SNAP = 8.0

#: 圆点（连接点）检测：墨迹内的局部最粗处半径超过它才可能是圆点。
#: ★ 这只是**下界**，真正管事的是下面的 ``DOT_MIN_STROKE_FACTOR`` ——
#: 绝对半径无法区分"圆点"和"两根导线交叉处的正方形"。
DOT_MIN_RADIUS = 2.6
#: 圆点的最大半径。超过它就不是连接点，是符号本体或涂黑块。
DOT_MAX_RADIUS = 9.0

#: 圆点半径相对**笔画半宽**的最小倍数。判据是比值而不是绝对像素，因为
#: 交叉处的粗度天然就是线宽的函数。本机实测（线宽 4 的导线，笔画半宽 2.0）：
#:   单根导线中段      dt = 2.00 → 比值 1.00
#:   两根导线十字交叉  dt = 2.83 → 比值 1.41   ← 交叉处是个 4x4 方块，内切圆 2√2
#:   半径 3 的圆点     dt = 3.61 → 比值 1.80
#:   半径 6 的圆点     dt = 6.40 → 比值 3.20
#: 取 1.6 卡在 1.41 与 1.80 的几何中点：交叉处的假圆点必被排除，
#: 而比导线只粗一点点的"小圆点"会被列为可疑并**报出来**，不静默当成圆点。
#: 这条非有不可：假圆点会让"十字交叉 = 不相连"这条铁律失效，
#: 把两条本来独立的回路并成一个结点。
DOT_MIN_STROKE_FACTOR = 1.6

#: 判定"元件端子在哪一侧"时，沿线方向把靠近本体的端点分成两簇的最小间隔。
TERMINAL_AXIS_TOL = 0.35


@dataclass
class WireSeg:
    x1: float
    y1: float
    x2: float
    y2: float
    #: 合并进来的原始霍夫段数，用来判断可靠性
    n_src: int = 1
    #: 结点名（连通后回填）
    node_a: str | None = None
    node_b: str | None = None

    @property
    def length(self) -> float:
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5

    @property
    def angle(self) -> float:
        """线段的**无向**倾角，值域 ``[0, 180)``。

        ★ 这个数只适合用来比较"两条线是否平行"（配合 ``min(diff, pi-diff)``），
        **不要**拿 ``cos/sin`` 去当方向向量 —— ``% 180`` 抹掉了朝向，
        一条从左往右的画线可能返回指向左边的角度。要方向就用
        ``(x2-x1, y2-y1)/length``。
        """
        import math
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0

    def to_dict(self) -> dict[str, Any]:
        return {"x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1),
                "length": round(self.length, 1), "n_src": self.n_src,
                "node_a": self.node_a, "node_b": self.node_b}


@dataclass
class Junction:
    """一个结点。``kind`` 说明它为什么是结点 —— 这是可核对性的关键。"""

    name: str
    x: float
    y: float
    kind: str                     # "endpoint" | "T" | "dot" | "merged"
    evidence: str = ""
    degree: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "x": round(self.x, 1), "y": round(self.y, 1),
                "kind": self.kind, "degree": self.degree, "evidence": self.evidence}


@dataclass
class Terminal:
    """某个元件位的两个端子接在哪个结点上。"""

    x: float
    y: float
    node: str | None
    side: str                     # "a" | "b"

    def to_dict(self) -> dict[str, Any]:
        return {"x": round(self.x, 1), "y": round(self.y, 1),
                "node": self.node, "side": self.side}


@dataclass
class WireGraph:
    segments: list[WireSeg] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    dots: list[dict[str, Any]] = field(default_factory=list)
    #: 比导线粗、但没粗到能算圆点的位置。**必须交人工看一眼** ——
    #: 若原图那里真有圆点，这个交叉其实是相连的，而这里按"不相连"处理了。
    suspect_dots: list[dict[str, Any]] = field(default_factory=list)
    #: 估出来的导线笔画半宽（像素）。圆点门槛由它推出。
    stroke_half: float = 0.0
    #: 每个符号位的两个端子（side="a" 是本体轴线负方向那一侧）
    terminals: dict[int, list[Terminal]] = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": {"w": self.image_size[0], "h": self.image_size[1]},
            "segments": [s.to_dict() for s in self.segments],
            "junctions": [j.to_dict() for j in self.junctions],
            "dots": list(self.dots),
            "suspect_dots": list(self.suspect_dots),
            "stroke_half": round(self.stroke_half, 2),
            "terminals": {str(k): [t.to_dict() for t in v]
                          for k, v in self.terminals.items()},
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------- 抽线段


def _wire_mask(image, slots: list[SymbolSlot]):
    """墨迹 - 符号本体区域 = 导线墨迹。

    ★ 这个"先剔除本体再连通"的顺序不能变，见模块头注释第 3 条。
    剔除区域用 ``slot_region()``，它只比本体外扩一点点 —— 外扩多了会把
    紧邻本体的那截**导线**也剃掉，元件两端就没线可接了。
    """
    import numpy as np

    mask = ink_mask(image)
    keep = mask.copy()
    for s in slots:
        x0, y0, x1, y1 = s.slot_region(margin=2.0)
        a = max(0, int(np.floor(y0)))
        b = min(mask.shape[0], int(np.ceil(y1)))
        c = max(0, int(np.floor(x0)))
        d = min(mask.shape[1], int(np.ceil(x1)))
        if b > a and d > c:
            keep[a:b, c:d] = False
    return keep


def _hough_segments(mask) -> list[WireSeg]:
    """霍夫抽直线段。**cv2 不可用时返回空表 + 由上层给出警告**。"""
    try:
        import cv2
    except Exception:                          # noqa: BLE001
        return []
    import numpy as np

    img = (mask.astype(np.uint8)) * 255
    lines = cv2.HoughLinesP(img, 1, np.pi / 360.0, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE,
                            maxLineGap=HOUGH_MAX_GAP)
    if lines is None:
        return []
    # ★ 不要假设 cv2.HoughLinesP 的返回形状。文档写的是 (N,1,4)，
    # 但实测本机这个构建给的是 (N,4)：写 `x1,y1,x2,y2 = row[0]` 会得到
    # `TypeError: 'numpy.int32' object is not iterable`，而且报错信息完全指不到形状上。
    # 直接展平成 (-1,4) 对两种形状都成立。
    arr = np.asarray(lines).reshape(-1, 4)
    out: list[WireSeg] = []
    for x1, y1, x2, y2 in arr:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if (x1, y1) == (x2, y2):
            continue
        out.append(WireSeg(x1, y1, x2, y2))
    return out


def _merge_collinear(segs: list[WireSeg]) -> list[WireSeg]:
    """把在同一直线上、彼此相接或重叠的段合并成一条。

    不合并的后果很直接：一根导线会残留成七八段，
    每一段的端点都成了候选结点，于是"同一根线"上凭空多出好几个结点，
    元件端子会被接到其中一个碎片结点上，而另一边接在另一个碎片上 ——
    拓扑看起来是通的，实际全错，而且**不会有任何报错**。
    """
    import math
    import numpy as np

    if not segs:
        return []
    parent = list(range(len(segs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            d1 = math.radians(a.angle)
            d2 = math.radians(b.angle)
            diff = abs(d1 - d2) % math.pi
            diff = min(diff, math.pi - diff)
            if math.degrees(diff) > MERGE_ANGLE_DEG:
                continue
            ux, uy = math.cos(d1), math.sin(d1)
            # ★★ 上面算出来的 (ux,uy) 只能用来判**角度差**（直线是无向的），
            # 绝不能拿来当"从 x1 指向 x2"的方向。``WireSeg.angle`` 是
            # ``atan2(dy,dx) % 180``，值域 [0,180) —— 一条从左往右画的线，
            # 它的 angle 可能指回左边。把这个反方向拿去算投影，整段的 ta/tb 会一起取反，
            # 于是**相距几十像素的两段会被算成"重叠"而误并**。
            # 实测代价：一张普通串联回路图里，电阻两侧的两段上导线
            # （(342,122)-(480,121) 与 (228,122)-(258,122)，真实间隔 84px）
            # 被算成 gap=-54 而并成一条 363px 的假线，正好跨过电阻的缺口，
            # 把电阻两端接到同一个结点上 —— 等效于把电阻短路，且不报任何错。
            # 方向一律用几何向量，不用角度。
            alen = a.length
            fx, fy = ((a.x2 - a.x1) / alen, (a.y2 - a.y1) / alen) if alen > 1e-9 else (ux, uy)
            # b 的两个端点相对 a 所在直线的垂直距离
            perp = max(abs(-(b.x1 - a.x1) * fy + (b.y1 - a.y1) * fx),
                       abs(-(b.x2 - a.x1) * fy + (b.y2 - a.y1) * fx))
            if perp > MERGE_PERP_TOL:
                continue
            # 沿 a 方向的投影间隔
            ta = sorted([0.0, alen])
            tb = sorted([(b.x1 - a.x1) * fx + (b.y1 - a.y1) * fy,
                         (b.x2 - a.x1) * fx + (b.y2 - a.y1) * fy])
            gap = max(tb[0] - ta[1], ta[0] - tb[1])
            if gap > MERGE_JOIN_GAP:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(segs)):
        groups.setdefault(find(i), []).append(i)

    out: list[WireSeg] = []
    for members in groups.values():
        pts: list[tuple[float, float]] = []
        nsrc = 0
        for i in members:
            pts.extend([(segs[i].x1, segs[i].y1), (segs[i].x2, segs[i].y2)])
            nsrc += segs[i].n_src
        arr = np.array(pts, dtype=float)
        # 在这组点里取相距最远的一对端点作为合并后的线段
        best = (0.0, 0, 1)
        for ii in range(len(arr)):
            d = np.hypot(arr[:, 0] - arr[ii, 0], arr[:, 1] - arr[ii, 1])
            k = int(d.argmax())
            if d[k] > best[0]:
                best = (float(d[k]), ii, k)
        _, i0, i1 = best
        out.append(WireSeg(float(arr[i0, 0]), float(arr[i0, 1]),
                           float(arr[i1, 0]), float(arr[i1, 1]), n_src=nsrc))
    out.sort(key=lambda s: -s.length)
    return out


# ---------------------------------------------------------------- 圆点


@dataclass
class DotReport:
    """圆点检测结果。``suspects`` 是"粗得可疑但不够格当圆点"的位置。"""

    dots: list[dict[str, Any]] = field(default_factory=list)
    #: 半径超过绝对下界、但没有明显超过笔画半宽的位置。
    #: **必须报出来** —— 一个真圆点被判成"不是圆点"，等于把一条真实连接判成不连。
    suspects: list[dict[str, Any]] = field(default_factory=list)
    stroke_half: float = 0.0
    warnings: list[str] = field(default_factory=list)


def find_dots(image, slots: list[SymbolSlot]) -> list[dict[str, Any]]:
    """找连接圆点（junction dot）。返回圆点表（兼容旧签名）。"""
    return find_dots_detail(image, slots).dots


def find_dots_detail(image, slots: list[SymbolSlot]) -> DotReport:
    """找连接圆点（junction dot），并把可疑但不够格的位置一并报出来。

    做法：在**导线墨迹**上取距离变换，找局部粗处。导线本身很细
    （半径 1~2px），圆点会明显更粗（半径 3~8px）。
    上限 ``DOT_MAX_RADIUS`` 用来排除符号本体和涂黑块，
    相对下限 ``DOT_MIN_STROKE_FACTOR`` 用来排除"两根导线交叉处的正方形"。

    为什么要专门找它：**圆点会否决"十字交叉不相连"这条默认规则。**
    有圆点的交叉是相连的，所以漏检一个圆点就会把一条真实连接判成不连；
    反过来，误判一个假圆点会把两条独立回路并成一个结点。两个方向都必须防。
    """
    import numpy as np
    from scipy import ndimage

    rep = DotReport()
    try:
        import cv2                                # noqa: F401
    except Exception:                             # noqa: BLE001
        rep.warnings.append("没装 opencv，无法做圆点检测；"
                            "十字交叉一律按「不相连」处理（教科书默认规则）。")
        return rep
    wire = _wire_mask(image, slots)
    if not wire.any():
        return rep
    rep.stroke_half = estimate_stroke_half(wire)
    need = max(DOT_MIN_RADIUS, DOT_MIN_STROKE_FACTOR * rep.stroke_half)

    dt = ndimage.distance_transform_edt(wire)
    # 局部极大 + 半径落区间
    mx = ndimage.maximum_filter(dt, size=9)
    peaks = (dt >= mx - 1e-6) & (dt >= DOT_MIN_RADIUS) & (dt <= DOT_MAX_RADIUS)
    lbl, n = ndimage.label(peaks, structure=np.ones((3, 3), dtype=int))
    raw: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(dt[ys, xs].max())
        # 排掉"宽线段的中间"：圆点周围各方向都有较粗的墨迹，
        # 而一根粗线的粗处只沿一个方向延伸。用"半径 ≥ 峰值 70% 的像素
        # 是否在各方向都接近"来粗筛 —— 简单做法是看粗处的形状是否近圆。
        sel = dt >= 0.7 * r
        yy, xx = np.where(sel)
        if len(xx) < 3:
            continue
        w = float(xx.max() - xx.min() + 1)
        h = float(yy.max() - yy.min() + 1)
        if max(w, h) / max(1.0, min(w, h)) > 2.5:
            continue                              # 拉长 → 是粗线，不是圆点
        raw.append({"x": cx, "y": cy, "radius": round(r, 1)})

    # 去重：挨得很近的峰值合成一个
    merged: list[dict[str, Any]] = []
    for d in sorted(raw, key=lambda t: -t["radius"]):
        if all((d["x"] - m["x"]) ** 2 + (d["y"] - m["y"]) ** 2 > (2 * d["radius"]) ** 2
               for m in merged):
            merged.append(d)

    for d in merged:
        if d["radius"] >= need:
            rep.dots.append(d)
        else:
            d = dict(d)
            d["ratio"] = round(d["radius"] / max(1e-9, rep.stroke_half), 2)
            d["reason"] = (f"半径 {d['radius']} 相对笔画半宽 {rep.stroke_half:.2f} "
                           f"只有 {d['ratio']} 倍，低于门槛 {DOT_MIN_STROKE_FACTOR}")
            rep.suspects.append(d)

    # ★ 这里**只如实记录**"这些位置比导线粗、但不够格当圆点"，不做任何解释。
    # 解释交给 build_wire_graph：那时才知道这些位置里哪些落在**内部交叉点**上。
    # 第一版在这里就写了"已按不相连处理"，而实际上拐角处的可疑点与拓扑毫无关系，
    # 文案与事实相反，看日志的人会被带偏。
    return rep


# ---------------------------------------------------------------- 连通


def _point_on_seg(px: float, py: float, s: WireSeg, tol: float) -> tuple[bool, float]:
    """点是否落在线段内部（含端点）。返回 ``(是否落在内部, 沿线参数 t)``。

    "落在内部"用投影参数判断：t 在 (0,1) 开区间内、且垂距在 tol 以内。
    端点附近（t≈0 或 1）不算内部，那种情况按"端点对端点"处理。
    """
    import math

    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return False, 0.0
    t = ((px - s.x1) * dx + (py - s.y1) * dy) / L2
    if t <= 0.0 or t >= 1.0:
        return False, t
    projx, projy = s.x1 + t * dx, s.y1 + t * dy
    dist = math.hypot(px - projx, py - projy)
    return dist <= tol, t


def _interior_crossing(a: WireSeg, b: WireSeg) -> tuple[float, float] | None:
    """两条线段的**内部×内部**交点；不是这种相交就返回 ``None``。

    ★ 单独把它提出来，是因为"粗得可疑的墨迹"只有落在这类交点上才有意义。
    第一版把所有"比导线粗"的位置都当可疑圆点报出来，结果**每一张图**都会
    在导线的每个**拐角**上报两条 —— 拐角是"端点接端点"，压根不涉及
    "交叉要不要连"这个判断，而两根 4px 导线成 90° 相接处的方块
    dt = 2.83（= 2√2），天然就比单根导线的 2.0 粗。
    这么一报，正常电路图每次都会触发"结构缺失 → 升级给视觉模型"，
    把"本地优先"直接架空。
    """
    import math

    dx1, dy1 = a.x2 - a.x1, a.y2 - a.y1
    dx2, dy2 = b.x2 - b.x1, b.y2 - b.y1
    den = dx1 * dy2 - dy1 * dx2
    if abs(den) < 1e-9:
        return None
    t = ((b.x1 - a.x1) * dy2 - (b.y1 - a.y1) * dx2) / den
    u = ((b.x1 - a.x1) * dy1 - (b.y1 - a.y1) * dx1) / den
    if not (0.0 < t < 1.0 and 0.0 < u < 1.0):
        return None
    return (a.x1 + t * dx1, a.y1 + t * dy1)


def _segments_connected(a: WireSeg, b: WireSeg, dots: list[dict[str, Any]]
                        ) -> tuple[str, str] | None:
    """两条线段是否电气相连。返回 ``(连接类型, 人类可读依据)``，不相连则 ``None``。

    ★ 这里的判断顺序就是那两条铁律的落地：
    1. 端点挨端点 → 相连；
    2. 一端点落在另一条线的**内部** → T 接，相连；
    3. 两条线**内部**互相穿过 → 十字交叉，**不相连**（除非那里有圆点）；
    4. 有圆点落在两线交点上 → 相连（圆点否决默认规则）。

    返回连接类型而不是只返回文案，是因为上层要拿它给结点标 ``kind``：
    早先上层是"只要合并了 >1 条线段就标 ``T``"，于是"四角相接的拐弯"
    也被标成 T —— 文案写着"端点对端点"、类型却是 T，自相矛盾。
    靠解析中文文案去判类型是会碎的，直接给类型。
    类型 ∈ ``{"endpoint", "t", "dot"}``。
    """
    import math

    def near(p, q, tol):
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= tol

    a_pts = [(a.x1, a.y1), (a.x2, a.y2)]
    b_pts = [(b.x1, b.y1), (b.x2, b.y2)]

    for p in a_pts:
        for q in b_pts:
            if near(p, q, NODE_SNAP):
                return "endpoint", "端点对端点"

    for p in a_pts:
        ok, t = _point_on_seg(p[0], p[1], b, NODE_SNAP)
        if ok:
            return "t", "T 形接入（第 1 条的端点落在第 2 条的内部）"
    for q in b_pts:
        ok, t = _point_on_seg(q[0], q[1], a, NODE_SNAP)
        if ok:
            return "t", "T 形接入（第 2 条的端点落在第 1 条的内部）"

    # 内部 × 内部：十字交叉。默认**不连** —— 除非交点附近有圆点。
    for p in a_pts:
        for q in b_pts:
            # 用两条直线的交点
            dx1, dy1 = a.x2 - a.x1, a.y2 - a.y1
            dx2, dy2 = b.x2 - b.x1, b.y2 - b.y1
            den = dx1 * dy2 - dy1 * dx2
            if abs(den) < 1e-9:
                continue
            t = ((b.x1 - a.x1) * dy2 - (b.y1 - a.y1) * dx2) / den
            u = ((b.x1 - a.x1) * dy1 - (b.y1 - a.y1) * dx1) / den
            if not (0.0 < t < 1.0 and 0.0 < u < 1.0):
                continue
            ix, iy = a.x1 + t * dx1, a.y1 + t * dy1
            for d in dots:
                if math.hypot(d["x"] - ix, d["y"] - iy) <= max(NODE_SNAP, d["radius"] + 2):
                    return "dot", (f"十字交叉处检出连接圆点（{d['x']:.0f},{d['y']:.0f}，"
                                   f"半径 {d['radius']}）→ 圆点否决「交叉不相连」的默认规则，判为相连")
            # 落到这里就是"内部相交且无圆点"，明确判为不连
            return None
    return None


def build_wire_graph(image, symbols: SymbolReport,
                     *, report_warnings: bool = True) -> WireGraph:
    """从图 + 符号位建出导线图。**永不抛异常**。"""
    import math

    from PIL import Image

    g = WireGraph()
    try:
        im = image
        if not hasattr(im, "convert"):
            im = Image.open(im)
        if hasattr(im, "load"):
            im.load()
    except Exception as e:                     # noqa: BLE001
        g.warnings.append(f"图片读取失败：{type(e).__name__}: {e}")
        return g
    g.image_size = (im.size[0], im.size[1])

    try:
        import cv2                               # noqa: F401
    except Exception as e:                       # noqa: BLE001
        g.warnings.append(
            f"缺少 OpenCV，无法用霍夫变换抽直线段：{e}。"
            "位图通道的导线追踪不可用 —— 请 pip install opencv-python-headless，"
            "或改用视觉模型重读这张图。")
        return g

    wire = _wire_mask(im, symbols.slots)
    raw = _hough_segments(wire)
    dotrep = find_dots_detail(im, symbols.slots)
    g.dots = dotrep.dots
    g.suspect_dots = dotrep.suspects
    g.stroke_half = dotrep.stroke_half
    if report_warnings:
        g.warnings.extend(dotrep.warnings)
    segs = _merge_collinear(raw)
    # 丢掉过短的碎片：它们几乎都是符号笔画残留或霍夫噪声，
    # 留着只会凭空造出结点。长度门槛设得比"本体内扩量"大一点。
    segs = [s for s in segs if s.length >= HOUGH_MIN_LINE * 0.6]
    g.segments = segs

    if not segs:
        g.warnings.append(
            "没有抽出任何导线线段。可能原因：图里只有元件没有连线、"
            "线太细在二值化时断了、或者霍夫参数不适合这张图。"
            "这种情况无法得到拓扑，请改用视觉模型重读。")
        return g

    # ---- 并查集连通
    parent = list(range(len(segs)))
    reasons: dict[tuple[int, int], str] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cross_ignored = 0
    crossings: list[tuple[float, float]] = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            hit = _segments_connected(segs[i], segs[j], g.dots)
            if hit:
                union(i, j)
                reasons[(i, j)] = hit
            else:
                cross_ignored += 1
                ix = _interior_crossing(segs[i], segs[j])
                if ix is not None:
                    crossings.append(ix)

    # ★ 可疑圆点只在**内部交叉处**才值得报出来（拐角处的粗块没有任何含金量，
    #   见 _interior_crossing 的注释）。不在交叉点附近的一律降级为说明性提示，
    #   否则每张图都会因为导线拐角而触发"结构缺失 → 升级给模型"。
    if g.suspect_dots:
        keep, dismissed = [], []
        for d in g.suspect_dots:
            if any(math.hypot(d["x"] - cx, d["y"] - cy) <= 2 * NODE_SNAP
                   for cx, cy in crossings):
                keep.append(d)
            else:
                dismissed.append(d)
        g.suspect_dots = keep
        if dismissed:
            g.warnings.append(
                f"另有 {len(dismissed)} 处墨迹比导线粗、但不在任何内部交叉点上"
                "（最典型的是导线拐角 —— 两根线成 90° 相接处的方块天然比单根线粗），"
                "已按「不是圆点」处理，不影响拓扑。"
                "位置：" + "、".join(f"({d['x']:.0f},{d['y']:.0f})"
                                   for d in dismissed[:6]))
        if keep:
            g.warnings.append(
                f"★ 有 {len(keep)} 处可疑墨迹**正好落在导线交叉点上**（笔画半宽 "
                f"{g.stroke_half:.2f}）：如果那里原本画了连接圆点，这个交叉就是相连的，"
                "而现在按不相连处理了 —— 拓扑可能少一处连接，请人工确认。"
                "位置：" + "、".join(f"({d['x']:.0f},{d['y']:.0f})" for d in keep))

    # ---- 结点命名：先定组的代表点，再按阅读顺序编号
    groups: dict[int, list[int]] = {}
    for i in range(len(segs)):
        groups.setdefault(find(i), []).append(i)
    reps = []
    for root, members in groups.items():
        pts = []
        for i in members:
            pts.extend([(segs[i].x1, segs[i].y1), (segs[i].x2, segs[i].y2)])
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        # kind 由**实际的连接类型**决定，不靠"合并了几条"猜。优先级：
        # 有圆点 > T 接 > 端点相接（拐弯）。一根独立的线段是 "endpoint"。
        kinds = {reasons[k][0] for k in reasons if find(k[0]) == root}
        if "dot" in kinds:
            kind = "dot"
        elif "t" in kinds:
            kind = "T"
        else:
            kind = "corner" if len(members) > 1 else "endpoint"
        reps.append((cy, cx, root, members, kind))
    # 阅读顺序：先上后下、先左后右
    reps.sort(key=lambda t: (round(t[0] / 40.0), t[1]))

    name_of_root: dict[int, str] = {}
    for idx, (cy, cx, root, members, kind) in enumerate(reps):
        name = str(idx)
        name_of_root[root] = name
        deg = len(members)
        ev = [reasons[k][1] for k in reasons if find(k[0]) == root]
        evidence = (f"由 {deg} 条线段合并而成；" + ("；".join(sorted(set(ev))[:2])
                    if ev else "独立线段"))
        g.junctions.append(Junction(name=name, x=cx, y=cy, kind=kind,
                                    evidence=evidence, degree=deg))

    # ★ 两个独立结点坐标重合 → 后面按"最近"给元件端子找结点时会挑错。
    # 这不是臆想的风险：十字交叉被判"不相连"时正是这种情形 ——
    # 两条独立的网线在同一像素上交叉，两个结点会在同一个坐标上。
    # 这个不确定性必须报出来，而不是让 _assign_terminals 悄悄挑一个。
    dup = []
    for ii in range(len(reps)):
        for jj in range(ii + 1, len(reps)):
            if math.hypot(reps[ii][1] - reps[jj][1], reps[ii][0] - reps[jj][0]) <= NODE_SNAP:
                dup.append((name_of_root[reps[ii][2]], name_of_root[reps[jj][2]],
                            reps[ii][1], reps[ii][0]))
    if dup:
        g.warnings.append(
            f"有 {len(dup)} 对**互不相连**的结点坐标几乎重合"
            "（典型来源：十字交叉未画圆点，两条网线在同一处交叉）。"
            "这种地方元件端子是按「最近」配对结点的，可能配错 —— "
            "请核对这几处："
            + "、".join(f"结点{a}/结点{b} @({x:.0f},{y:.0f})" for a, b, x, y in dup[:5]))

    for i, s in enumerate(segs):
        nm = name_of_root.get(find(i))
        s.node_a, s.node_b = nm, nm

    if cross_ignored:
        g.warnings.append(
            f"有 {cross_ignored} 对线段判为**不相连**（几何上相交但相交点在两条线"
            "的内部、且那里没有连接圆点）。这是「只把线段端点与圆点圆心当结点、"
            "几何交点不生成结点」这条规则的直接结果。如果原图里那个交叉其实"
            "是相连的（画了圆点但没被检出），拓扑会少一处连接 —— 请核对。")

    # ---- 元件端子
    # ★ 传"每条的根"而不是并查集数组本身：并查集里存的可能是指向别处的父指针，
    #   直接 parent[i] 取到的是父而不是根，节点名会张冠李戴。
    #   （第一版就是这么写的，它会静默给出错的节点名 —— 不报错，但拓扑全错。）
    root_of = [find(i) for i in range(len(segs))]
    g.terminals = _assign_terminals(im, symbols.slots, segs, name_of_root, root_of)

    n_unassigned = sum(1 for ts in g.terminals.values()
                       for t in ts if t.node is None)
    if n_unassigned:
        g.warnings.append(
            f"有 {n_unassigned} 个元件端子附近找不到导线 —— 这个元件在图上可能是"
            "悬空的，也可能是那段导线被符号本体区域误剃掉了。两种情况都会导致"
            "拓扑不完整，请逐个人工核对。")
    return g


def _assign_terminals(im, slots: list[SymbolSlot], segs: list[WireSeg],
                      name_of_root: dict[int, str],
                      root_of: list[int]) -> dict[int, list[Terminal]]:
    """给每个元件位定两个端子，各接在哪个结点上。

    做法：找"端点落在本体外边缘附近"的线段，把这些端点按**沿本体轴线**
    的方向分成两簇（正方向 / 负方向），两簇分别就是元件的两个端子。

    ★ 关于极性：``side="a"`` 是轴线负方向那一侧，``side="b"`` 是正方向侧。
    这与 IR 里"电压源 nodes[0] 是 + 端"的约定**不能自动对应** ——
    位图里哪个端子画了 + 号是图上的信息，必须靠识别 +/− 标记才知道。
    所以这里只给几何分侧，极性由上层标 `needs_human`，
    除非在上层真的检出了 + 号。
    """
    import math

    out: dict[int, list[Terminal]] = {}
    for si, s in enumerate(slots):
        cx, cy = s.cx, s.cy
        reach = max(s.w, s.h) / 2.0 + NODE_SNAP * 3
        hits: list[tuple[float, float, float, str, float]] = []
        for k, seg in enumerate(segs):
            nm = name_of_root.get(root_of[k], "?")
            for (px, py) in ((seg.x1, seg.y1), (seg.x2, seg.y2)):
                dist = math.hypot(px - cx, py - cy)
                if dist > reach:
                    continue
                hits.append((px, py, 0.0, nm, dist))

        # ---- 轴线方向：由**导线接过来的方向**定，而不是由外接矩形定。
        #
        # ★ 这里踩过一个坑：电压源和电流源画成圆，外接矩形近似正方形，
        #   ``w >= h`` 这个判据在圆上完全是抛硬币 —— 一个接在竖直导线上的
        #   电压源会被当成"水平朝向"，于是去左右两侧找端子，一个都找不到，
        #   两个端子都成了 None，整个元件从电路里消失（而且不报错）。
        #   所以必须看导线：如果接过来的端点主要在竖直方向铺开，轴就是竖直的。
        axis_x = None
        if len(hits) >= 2:
            xs = [h[0] for h in hits]
            ys = [h[1] for h in hits]
            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)
            if max(spread_x, spread_y) >= NODE_SNAP:
                axis_x = spread_x >= spread_y
        if axis_x is None:
            axis_x = s.orientation == "h"      # 没导线可依据时退回外接矩形
        ax, ay = (1.0, 0.0) if axis_x else (0.0, 1.0)

        # 沿轴线的投影原点用**接点群的中心**，不用本体中心：
        # 本体检测有 ±3px 的漂移，用接点群中心更稳。
        if hits:
            ox = sum(h[0] for h in hits) / len(hits)
            oy = sum(h[1] for h in hits) / len(hits)
        else:
            ox, oy = cx, cy
        for i in range(len(hits)):
            px, py, _, nm, dist = hits[i]
            hits[i] = (px, py, (px - ox) * ax + (py - oy) * ay, nm, dist)

        terms: list[Terminal] = []
        for side in ("a", "b"):
            sign = -1.0 if side == "a" else 1.0
            sel = [h for h in hits if (h[2] > 0) == (sign > 0)] if hits else []
            sel = [h for h in sel if abs(h[2]) > 1.0]
            if not sel:
                terms.append(Terminal(
                    x=cx + sign * ax * max(s.w, s.h) / 2,
                    y=cy + sign * ay * max(s.w, s.h) / 2, node=None, side=side))
                continue
            sel.sort(key=lambda t: t[4])          # 取离本体最近的那个
            px, py, _, nm, _dist = sel[0]
            terms.append(Terminal(x=px, y=py, node=nm, side=side))
        out[si] = terms
    return out


def probe() -> dict[str, Any]:
    """给 /api/health 用：位图通道能不能跑。"""
    out: dict[str, Any] = {"opencv": False, "scipy": False, "numpy": False,
                           "pillow": False}
    try:
        import cv2
        out["opencv"] = getattr(cv2, "__version__", True)
    except Exception:                          # noqa: BLE001
        pass
    try:
        import scipy
        out["scipy"] = getattr(scipy, "__version__", True)
    except Exception:                          # noqa: BLE001
        pass
    try:
        import numpy
        out["numpy"] = numpy.__version__
    except Exception:                          # noqa: BLE001
        pass
    try:
        import PIL
        out["pillow"] = PIL.__version__
    except Exception:                          # noqa: BLE001
        pass
    out["ready"] = bool(out["opencv"])
    return out
