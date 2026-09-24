"""位图通道的符号定位：找出"这里有一个元件"以及它是哪一类。

两条独立的判据，故意都用上，因为它们**错的时机不同**：

1. **闭合空洞法**（主力）。把墨迹做 ``binary_fill_holes``，再与原墨迹相减，
   剩下的就是所有"被墨迹围起来的空白"。这是一次形态学运算，一次就能把整张图
   的封闭形状全找出来 —— 不需要滑动窗口，也不需要阈值调参。
   它对**风格**不敏感（手绘抖、印刷字体不同、线宽不同都不影响"有没有围起来"），
   所以是主力。
2. **模板匹配**（辅助）。模板从项目自己的渲染器常量生成（见 templates.py）。
   它对**形状相近的符号**才有用：本项目的电压源是**不带内部直径线的空圆**，
   电流源是圆里加箭头 —— 两者都是"1 个洞"，洞法分不开，只有模板能分。
   而它是逐像素的，对透视和风格敏感，所以只当辅助、不当主判据。

**为什么两种都要，而不能只留一种**：真实图里既有"印刷体矩形电阻"
（洞法一眼认出），也有"手绘圆圈加箭头"（模板能分但洞法分不开），
还有电池样式的电压源（两根长短线，**完全没有洞**，两种方法都很勉强）。
留一个"识别不出"的出口比强行猜一个更符合这个项目的立场。

**★ 这一层与 ``app/ingest/topology.py`` 的对应关系**（顺序必须一致）：
本模块负责第一步"挑槽位"并给出**本体区域**；上层拿到本体区域后
必须**先剔除落在本体内部的线段，再建连通图**。顺序反了，符号本体的
闭环导体就会参与连通、把元件两端短接 —— 那个坑在矢量通道里已经踩过一次，
位图通道不能再踩一遍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import templates as T

#: 判定"这里是墨迹"的灰度阈值。白纸黑线，160 够宽裕；
#: 偏保守（宁可少算墨迹，不可把阴影当线）—— 多算墨迹会凭空造出连接。
INK_THRESH = 160

#: 候选连通块的尺寸门槛（像素）。小于这个的不是符号，是文字笔画或噪点；
#: 大于这个的是图框/标题栏/整块阴影，不该当成元件。
SYMBOL_MIN_SIDE = 12
SYMBOL_MAX_SIDE = 400

#: 长宽比超过这个值就认为是导线而不是符号本体。
WIRE_ASPECT = 6.0

#: 洞的填充率区间（洞面积 / 洞外接矩形面积）。
#:
#: ★ 这两个数是**实测**出来的，不是估的：矩形框的洞填满整个外接矩形
#: （实测 1.00），而圆的洞在外接矩形四角留空（实测 0.79）。
#: 所以**填充率才是区分"矩形 vs 圆"的主判据，长宽比是辅助** ——
#: 长宽比只在"方框 vs 长条"上有效，一个接近正方形的电阻框长宽比也是 1.1，
#: 跟圆撞在一起。
FILL_RECT_MIN = 0.93
FILL_CIRCLE_MAX = 0.90

#: 矩形洞的长宽比区间。下限取 1.15 是实测需要：一个 120×100 的电阻框
#: 内部洞是 112×92，长宽比只有 1.22 —— 第一版把下限写成 1.25，
#: 于是这种"偏方的电阻框"被判成认不出。
RECT_ASPECT = (1.15, 5.0)

#: 圆形洞的长宽比上限（外接矩形接近正方形）。
CIRCLE_ASPECT_MAX = 1.12

#: 两个洞要被认为"同一个圆被直径线切成两半"，它们的外接矩形要满足：
#: 尺寸相近、彼此紧贴。实测这个特征**与朝向无关** ——
#: 水平直径线切出上下两半，竖直直径线切出左右两半，两种都要认。
PAIR_SIZE_TOL = 0.35        # 尺寸相对差
PAIR_GAP_RATIO = 0.45       # 间隙 / 较短边

#: 模板匹配的归一化尺寸（等比缩放后居中留边到这个画布）。
MATCH_SIZE = (64, 64)

#: ★ 实心判据：**符号是笔画图形，不是实心块**。
#:
#: 两条同时满足才算"实心块"，两条都是实测校准的：
#:
#: 1. ``max(内切圆半径) / 较长边 > SOLID_RATIO_MAX``
#: 2. ``墨迹面积 / 外接矩形面积 > SOLID_INK_MAX``
#:
#: ★ 第 1 条的分母必须是**较长边**，不能是较短边。第一版用了较短边，
#: 结果是：任何扁平图形的内切圆半径都受高度限制，比值天然接近 0.5，
#: 于是**电感被当成实心块丢掉**（实测它的比值是 5/13 = 0.38）。
#: 换成较长边之后：实心圆 60/121 = 0.50 仍被拦，电感 5/121 = 0.04 通过。
#:
#: 第 2 条用来兜第 1 条漏掉的"扁平实心条"（101×13 的实心块，
#: 分母换成较长边后比值只有 0.054，会溜过去）。
#:
#: 实测一组：
#:   实心黑圆            → 半径比 0.50、占比 0.785  → 拦掉 ✓
#:   扁平实心条 101×13   → 半径比 0.054、占比 1.00  → 拦掉 ✓（靠第 2 条）
#:   矩形框（笔画 4px）  → 半径比 0.013、占比 0.13  → 通过
#:   空圆                → 半径比 0.017、占比 0.11  → 通过
#:   电容极板            → 半径比 0.025、占比 0.13  → 通过
#:   电感（120×10）      → 半径比 0.041、占比 0.24  → 通过
SOLID_RATIO_MAX = 0.25
SOLID_INK_MAX = 0.55


def estimate_stroke_half(mask) -> float:
    """估笔画**半宽**（像素）。取墨迹内所有"局部最粗处"半径的中位数。

    为什么取中位数而不是最大值：最大值会被圆点、被符号本体、被涂黑的异物污染；
    而导线在整张图里占绝大多数像素，局部最粗处的中位数就是线宽的一半。
    实测（线宽 → dt 最大值）：2→1.0、3→2.0、4→2.0、6→3.0、8→4.0、12→6.0、16→8.0。

    ★ 这个值在两层都要用，所以放在 ``symbols``（``wires`` 已经依赖它，反向依赖不行）：
    - ``wires`` 用它定"圆点半径要明显大于笔画半宽"的门槛；
    - ``symbols`` 用它把"洞的外接矩形"外扩成"符号本体的外接矩形"。
      这一处原来写死 ``stroke = 3``，而实测图里线宽是 4px ——
      外扩不够就取不到完整的本体轮廓，模板得分被压到 0.22（本该 0.67），
      于是"模板判据沉默"、"类型置信度 0.80 低于闸门"、
      **每一张图里的每一个电阻都会触发一次升级给视觉模型**。
      这是测量缺陷被误当成"两条判据真的不一致"，很难从日志看出来。
    """
    import numpy as np
    from scipy import ndimage

    if not mask.any():
        return 0.0
    dt = ndimage.distance_transform_edt(mask)
    mx = ndimage.maximum_filter(dt, size=5)
    ridge = (dt >= mx - 1e-6) & (dt > 0)
    if not ridge.any():
        return float(dt.max())
    return float(np.median(dt[ridge]))


def solidity(mask) -> float:
    """实心度的半径比 = 最大内切圆半径 / 图形**较长边**。

    ``distance_transform_edt`` 给每个墨迹像素到最近背景的距离，
    最大值就是最大内切圆半径。实心块的半径接近图形尺寸的一半，
    任何笔画图形的半径都远小于它 —— 这个判据与线宽、与缩放都无关。
    """
    import numpy as np
    from scipy import ndimage

    if not mask.any():
        return 1.0
    ys, xs = np.where(mask)
    long_side = max(1, max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)))
    # ★ 必须补一圈背景再算距离变换。外接矩形被填满的子图里**一个背景像素都没有**，
    # scipy 的 distance_transform_edt 在这种输入上会退化，返回一个无意义的巨大值
    # （实测 41x4 实心条得到 dt.max=40.2 → 半径比 0.98，而几何上应是 2/41≈0.05）。
    # 判"实心"的结论碰巧还是对的，但数值是错的，一旦有人拿这个数当特征用就会踩坑。
    # 补一圈 0 之后距离相对真实边界度量，结果与几何直觉一致。
    padded = np.pad(mask, 1, constant_values=False)
    dt = ndimage.distance_transform_edt(padded)
    return float(dt.max()) / float(long_side)


def solid_blob_reason(mask) -> str | None:
    """是实心块就返回原因（一句人话），否则 ``None``。"""
    import numpy as np

    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    ink = float(mask.sum())
    ratio = ink / float(max(1, w * h))
    rad = solidity(mask)
    if rad > SOLID_RATIO_MAX:
        return (f"这块墨迹是实心的（最大内切圆半径占较长边的 {rad:.2f}，"
                f"超过上限 {SOLID_RATIO_MAX}）—— 笔画图形不会长这样。")
    if ratio > SOLID_INK_MAX:
        return (f"这块墨迹几乎填满了它的外接矩形（占比 {ratio:.2f}，"
                f"超过上限 {SOLID_INK_MAX}）—— 笔画图形不会长这样。")
    return None

#: 开放图形（无闭合空洞的符号）靠模板认领的门槛，分两档。
#: 测得电容的模板得分在 0.37~0.62 之间浮动（不同画法的极板长宽比差异很大），
#: 所以 0.35 认"有个元件"、0.55 才敢给类型 —— 中间那档留 ``?`` 交人工。
OPEN_ACCEPT = 0.35
OPEN_KIND_MIN = 0.55

#: 判断"这块墨迹其实是导线网络"用的填充率下限：
#: 无洞且填充率低于此值 → 是细长的网线，不是元件。
#:
#: ★ 取值的硬约束来自两侧实测值，不是拍的：
#:   必须**排除**的最大实测值 = 0.057（一张普通串联回路图的整圈导线墨迹）
#:   必须**保留**的最小实测值 = 0.074（电池式电压源的极板 + 引线）
#: 所以阈值只能落在 (0.057, 0.074) 之间，取中点 0.065。
#: 这条余量很窄，是这一步的固有难点：一张"元件很少、导线很长"的图，
#: 它的整圈墨迹填充率会逼近电容的量级。逼到极限时的表现是
#: "把导线网络报成未解释墨迹"（多一条提醒，不会算错），方向是安全的。
WIRE_NETWORK_FILL = 0.065

#: 判断"这个洞其实是闭合导线回路围出来的空白"用的比值上限：
#: ``洞面积 / 洞所在的那个墨迹连通块的外接矩形面积``。超过它就是导线回路，不是元件。
#:
#: ★ 这条判据是**量出来的**，而且是被数据逼出来的：
#: 闭合导线回路与电阻框（甚至与放大 6 倍的电阻框）**几何上完全同形** ——
#: 都是一个矩形轮廓，线宽一样时连"洞面积 / 墨迹面积"都分不开
#: （实测导线回路 7.43~15.63、放大 6 倍的电阻 9.92，区间重叠）。
#: 唯一稳定分开的是"洞占了所在墨迹块外接矩形的多大比例"：
#:   真符号的**引线会把外接矩形撑出去**，闭合回路没有 ——
#:   孤立电阻 0.226、孤立电压源 0.179、回路图里的真 R/V 只有 0.017/0.015；
#:   而导线回路 0.820、无元件细线回形 0.881。
WIRE_LOOP_HOLE_RATIO = 0.55

#: 只有当图形的**绝对规模**达到这个边长，才敢把上面那条判据用出来。
#:
#: ★ 为什么必须叠加这道绝对规模的闸门 —— 因为比值本身不够：
#: 一个**独立画着、没有被引线撑开外接矩形**的符号，比值会高得离谱。
#: 本机实测（每个符号单独画）：
#:   电阻框 180x60 → 0.828
#:   空圆电压源 120x120 → 0.874   ← 比导线回路（0.820）还高！
#: 也就是说只靠比值会把电压源和电阻一起误杀。分开它们只剩规模这一条路。
#:
#: 取 200px 是**有意识的选择，代价明确写在下面**：
#:   - 误判方向一（把导线回路认成元件）：导线追踪会剔掉全图墨迹，
#:     得到 0 条线段、0 个结点，且**不报任何错** —— 灾难性。
#:   - 误判方向二（把一个大符号认成导线回路）：本地层报"结构缺失"，
#:     按 ``escalate_on="structural"`` 升级给视觉模型 —— 只是多花一次调用。
#: 所以宁可错杀方向二。已知代价：画得特别大的符号
#: （实测 480x120 的电阻框，比值 0.690）会被判成导线回路。
#: 在真实电路图里这几乎不会发生（元件本体远小于整张图），
#: 真发生了也只是升级给视觉模型。
WIRE_LOOP_MIN_SIDE = 200.0

#: 置信度低于它就强制人工确认。与 ir.model.CONFIDENCE_GATE 同值，
#: 但这一层自己导一遍是为了能独立测试（不想让 vision 依赖 solver/ir 的内部状态）。
CONFIDENCE_GATE = 0.85


# ---------------------------------------------------------------- 洞


@dataclass
class Hole:
    """一个闭合空洞（被墨迹围起来的空白区域）。"""

    x: int
    y: int
    w: int
    h: int
    area: int
    #: 洞的填充率 = 洞面积 / 外接矩形面积。越接近 1 越"整"。
    fill: float
    #: 长宽比（长边/短边）
    aspect: float
    label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "area": self.area, "fill": round(self.fill, 3),
                "aspect": round(self.aspect, 2)}


def ink_mask(image: Any):
    """灰度取墨迹。返回 bool 数组（True = 墨迹）。"""
    import numpy as np
    from PIL import Image

    im = image
    if not hasattr(im, "convert"):
        im = Image.open(im)
    g = np.array(im.convert("L"))
    if g.ndim == 3:                            # 万一还是彩色
        g = g[:, :, 0]
    return g < INK_THRESH


def find_holes(mask) -> list[Hole]:
    """闭合空洞法。一次形态学运算把整张图的封闭形状全找出来。

    做法：``binary_fill_holes(ink) & ~ink``。填洞后多出来的那些像素，
    正好就是所有被墨迹围住的空白。
    """
    import numpy as np
    from scipy import ndimage

    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    lbl, n = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    out: list[Hole] = []
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        area = int((lbl[sl] == i).sum())
        if w <= 0 or h <= 0 or area < 8:
            continue
        out.append(Hole(x=xs.start, y=ys.start, w=w, h=h, area=area,
                        fill=area / float(w * h),
                        aspect=max(w, h) / float(max(1, min(w, h))),
                        label=i))
    out.sort(key=lambda t: -t.area)
    return out


def _paired(h1: Hole, h2: Hole) -> bool:
    """两个洞是不是"同一个圆被一条直径线切成两半"。

    实测特征：**两个等大且紧贴的洞**。注意这与朝向无关 ——
    竖直直径线切出左右两半，水平直径线切出上下两半，两种都算。
    （技能文档 §9.2 曾写成"一对左右对称半圆"，那是把朝向当成了特征，
    是错的：横着画的圆内直径线给出的是一上一下两个洞。）
    """
    size1, size2 = min(h1.w, h1.h), min(h2.w, h2.h)
    if size1 <= 0 or size2 <= 0:
        return False
    if abs(size1 - size2) / float(max(size1, size2)) > PAIR_SIZE_TOL:
        return False
    # 竖直贴：上下相邻，x 大致对齐
    gap_v = max(h1.y, h2.y) - min(h1.y + h1.h, h2.y + h2.h)
    x_off = abs((h1.x + h1.w / 2) - (h2.x + h2.w / 2))
    if 0 <= gap_v <= PAIR_GAP_RATIO * size1 and x_off < 0.5 * max(h1.w, h2.w):
        return True
    # 水平贴：左右相邻，y 大致对齐
    gap_h = max(h1.x, h2.x) - min(h1.x + h1.w, h2.x + h2.w)
    y_off = abs((h1.y + h1.h / 2) - (h2.y + h2.h / 2))
    if 0 <= gap_h <= PAIR_GAP_RATIO * size1 and y_off < 0.5 * max(h1.h, h2.h):
        return True
    return False


def group_holes(holes: list[Hole]) -> list[list[Hole]]:
    """把属于同一个符号的洞归成一组（主要是把那对半圆配成对）。"""
    used = [False] * len(holes)
    groups: list[list[Hole]] = []
    for i, h1 in enumerate(holes):
        if used[i]:
            continue
        mate = None
        for j in range(i + 1, len(holes)):
            if used[j]:
                continue
            if _paired(h1, holes[j]):
                mate = j
                break
        if mate is not None:
            used[i] = used[mate] = True
            groups.append([h1, holes[mate]])
        else:
            used[i] = True
            groups.append([h1])
    return groups


# ---------------------------------------------------------------- 模板


def _mask_bbox(mask):
    """墨迹外接矩形 ``(x, y, w, h)``；空图返回 ``None``。"""
    import numpy as np

    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _aligned_hole_box(mask):
    """**对齐用的洞参照框**：最大闭合空洞的外接矩形；若是"一对配对的洞"则取两者的并集。

    ★ 为什么要单独处理"一对洞"：电压源画成"圆 + 一条直径线"时，
    它的内部被切成两个半圆洞。这时拿"最大的那个洞"去对齐，
    参照框只有半个圆，而模板 V 的参照框是整圆 —— 两边对不上，
    实测该图形的模板得分从应有的 0.62 掉到 0.19（还不如误拿到的 C）。
    一对配对洞的**并集**才是那个圆，才是正确的参照。
    """
    import numpy as np

    holes = find_holes(mask)
    holes = [h for h in holes if h.area >= 8]
    if not holes:
        return None
    holes.sort(key=lambda t: -t.area)
    if len(holes) >= 2 and _paired(holes[0], holes[1]):
        a, b = holes[0], holes[1]
        x0 = min(a.x, b.x)
        y0 = min(a.y, b.y)
        x1 = max(a.x + a.w, b.x + b.w)
        y1 = max(a.y + a.h, b.y + b.h)
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    h = holes[0]
    return (int(h.x), int(h.y), int(h.w), int(h.h))


def _pad_box(box, pad: float):
    """外接矩形外扩 ``pad``。"""
    x, y, w, h = box
    p = int(round(pad))
    return (x - p, y - p, w + 2 * p, h + 2 * p)


def _normalize_box(mask, box):
    """按**指定的**外接矩形裁切，再等比缩放居中放进 ``MATCH_SIZE``。

    与 ``_normalize`` 的区别只有一个：参考框由调用方给，不由"墨迹外接矩形"决定。
    这个区别是决定性的 —— 见 ``match_kind``。
    """
    import numpy as np
    from PIL import Image

    W, H = MATCH_SIZE
    out = np.zeros((H, W), dtype=bool)
    if box is None:
        return out
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1 = min(mask.shape[1], x + w)
    y1 = min(mask.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return out
    crop = mask[y0:y1, x0:x1]
    if not crop.any():
        return out
    ch, cw = crop.shape
    scale = min(W / float(cw), H / float(ch))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    small = np.array(im) > 96
    ox, oy = (W - nw) // 2, (H - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = small
    return out


def _normalize(mask):
    """把图块缩放到 ``MATCH_SIZE`` 里做形状比对。**按墨迹外接矩形归一等比缩放。**

    ★ **必须等比缩放 + 居中留边，不能直接拉伸到正方形。**
    第一版就是直接 resize 到 64×64，后果有两个：
    1. 长宽比信息被抹掉 —— 细长的电感被拉成方形，跟别的形状看起来一样；
    2. 更危险的是下面那道"填充率闸门"失去依据（见 ``_ink_ratio`` 的注释）。

    这样归一之后，"墨迹占画布的比例"在两个图块之间才是可比的。

    ★ 但它**不适合用来对齐"带洞的符号"** —— 见 ``match_kind``。
    """
    return _normalize_box(mask, _mask_bbox(mask))


def match_kind(patch, *, kinds: tuple[str, ...] = ("R", "V", "I", "C", "L")
               ) -> dict[str, float]:
    """把一个小图块和每个符号模板比对，返回 ``{kind: 得分0~1}``。

    得分用 F1（精确率与召回率的调和平均），不用 IoU：
    线稿的笔画很细，一个像素的错位就能让 IoU 掉一大截，
    而 F1 对细笔画的错位宽容得多。

    ★ 但 F1 有一个**结构性缺陷**，必须靠 ``solidity`` 那道闸门补：
    当候选是一坨实心墨（涂黑的圆、水印、阴影）时，它把**每一条模板**的
    笔画都包住了，于是每条模板的召回率都是 1，F1 退化成只比精确率，
    也就是"谁的笔画最少谁赢"。

    实测：实心黑圆拿到 ``V 0.46 / I 0.53 / C 0.53 / L 0.65`` 并**被认成电感**。
    这比认不出严重得多 —— 电感在直流化简里是**短路**，一个幻影电感会把
    两个节点合并掉，而且这个错误会一路传进求解器，三法还会一致地给出
    同一个错答案，互校拦不住。

    第一版试过用"墨迹占比超模板 3.5 倍就否决"来堵，**堵不住**：
    模板之间的墨迹占比本身就随线宽从 0.06 漂到 0.29，
    实心圆（0.79）在 V/I/C 那几条粗笔画模板面前照样过关。
    换成实心度之后才真正拦住：实心圆 0.50，全部笔画模板都在 0.05 以下。

    ★★ 归一的参照系必须**两边一致**，这是第二个踩出来的坑，而且后果很隐蔽：

    原来图块和模板各自按"墨迹外接矩形"归一等比缩放。可图块是**从整张图上
    连着一截外接导线抠下来的**，模板里却没有引线 —— 同一个电阻，两边参考框
    不一样，缩放比例就不一样，细笔画一错开几个像素，F1 就腰斩。
    实测：一张正常电路图里的电阻，模板得分从该有的 0.67 掉到 **0.24**
    （还不如误拿到的 C=0.44），于是"模板判据沉默" →
    "类型置信度 0.80 低于闸门" → **每一个电阻都触发一次升级给视觉模型**。
    日志上只看到"置信度不足"，完全看不出根因是参考框不一致。

    修法：**有洞的符号一律用"最大闭合空洞"对齐**（电阻框的洞就是它的内部、
    电压源圆的洞就是圆内 —— 那是同一个物理参照，与引线长短无关），
    两边都取"洞的外接矩形 + 各自一个笔画宽度"再归一。没有洞的符号
    （电容、电感）没有可用的共同参照，仍按墨迹外接矩形归一。
    """
    import numpy as np

    if solid_blob_reason(patch) is not None:
        return {k: 0.0 for k in kinds}

    p_ink = _normalize(patch)
    hb_p = _aligned_hole_box(patch)
    p_hole = None
    if hb_p is not None:
        # 外扩"两个笔画宽"：一个笔画宽刚好是描写本体的那一圈，再留一点余量。
        p_hole = _normalize_box(patch, _pad_box(hb_p, 2.0 * estimate_stroke_half(patch)))

    scores: dict[str, float] = {}
    for k in kinds:
        best = 0.0
        for v in (False, True):
            tm_raw = T.template_mask(k, v)
            hb_t = _aligned_hole_box(tm_raw)
            if hb_t is not None and p_hole is not None:
                p = p_hole
                tm = _normalize_box(tm_raw,
                                    _pad_box(hb_t, 2.0 * estimate_stroke_half(tm_raw)))
            else:
                p = p_ink
                tm = _normalize(tm_raw)
            inter = float((p & tm).sum())
            if inter == 0:
                continue
            prec = inter / max(1.0, float(p.sum()))
            rec = inter / max(1.0, float(tm.sum()))
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            best = max(best, f1)
        scores[k] = best
    return scores


# ---------------------------------------------------------------- 槽位


@dataclass
class SymbolSlot:
    """一个候选元件位。``kind`` 是猜测，``evidence`` 是猜的依据。"""

    x: int
    y: int
    w: int
    h: int
    kind: str = "?"
    orientation: str = "h"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    #: 洞法给的候选（可能为空 —— 开放图形没有洞）
    holes: list[Hole] = field(default_factory=list)
    hole_kind: str | None = None
    #: 模板匹配的完整得分表，便于人工核对"为什么没选另一个"
    template_scores: dict[str, float] = field(default_factory=dict)
    template_kind: str | None = None
    #: 两种判据是否一致。不一致必须人工确认。
    agreed: bool = False

    @property
    def needs_human(self) -> bool:
        return self.confidence < CONFIDENCE_GATE or self.kind == "?"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def body_box(self) -> tuple[float, float, float, float]:
        """本体包围盒 ``(x0, y0, x1, y1)``。"""
        return (float(self.x), float(self.y),
                float(self.x + self.w), float(self.y + self.h))

    def slot_region(self, *, margin: float = 3.0) -> tuple[float, float, float, float]:
        """★ 要**从导线追踪中剔除**的圆柱区域。

        这是矢量通道那个坑的直接对应物：符号本体是闭环导体，
        若参与连通就会把元件两端短接。所以上层拿到这个区域后，
        必须先剔除落在其中、且长度足够短的线段，再建连通图。

        ``margin`` 只放一点点（默认 3px）：放大了会把紧邻本体的那截**导线**
        也剔掉，元件两端就没线可接了，拓扑直接断。
        放小了符号笔画会残留成一小段孤立墨迹，被当成导线 ——
        上层用"长度 < margin*2 的碎段直接丢"来兜住这种情况。
        """
        return (self.x - margin, self.y - margin,
                self.x + self.w + margin, self.y + self.h + margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "orientation": self.orientation,
            "confidence": round(self.confidence, 3),
            "needs_human": self.needs_human,
            "agreed": self.agreed,
            "hole_kind": self.hole_kind,
            "template_kind": self.template_kind,
            "template_scores": {k: round(v, 3) for k, v in
                                sorted(self.template_scores.items(), key=lambda t: -t[1])},
            "holes": [h.to_dict() for h in self.holes],
            "evidence": list(self.evidence),
        }


@dataclass
class SymbolReport:
    slots: list[SymbolSlot] = field(default_factory=list)
    holes: list[Hole] = field(default_factory=list)
    #: 检出但**判定为"闭合导线回路围出来的空白"**、因而没有当成元件的洞。
    #: 单独留一栏是为了不静默丢弃 —— 它恰恰是"这里有一圈导线"的证据。
    wire_holes: list[Hole] = field(default_factory=list)
    #: 既不像导线、也没有被任何槽位解释掉的大块墨迹。
    #: **必须报出来** —— 它可能就是一个没认出来的元件。
    unexplained: list[dict[str, Any]] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": {"w": self.image_size[0], "h": self.image_size[1]},
            "slot_count": len(self.slots),
            "slots": [s.to_dict() for s in self.slots],
            "holes": [h.to_dict() for h in self.holes],
            "wire_holes": [h.to_dict() for h in self.wire_holes],
            "unexplained": list(self.unexplained),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------- 分类


def _classify_group(group: list[Hole], mask, box: tuple[int, int, int, int]
                    ) -> tuple[str | None, float, list[str], dict[str, float], str | None]:
    """给出 (洞法结论, 洞法置信度, 依据, 模板得分表, 模板结论)。"""
    import numpy as np

    evidence: list[str] = []
    hole_kind: str | None = None
    hole_conf = 0.0

    if len(group) == 2:
        h1, h2 = group
        hole_kind = "V"
        hole_conf = 0.9
        evidence.append(
            f"检出两个等大紧贴的洞（{h1.w}×{h1.h} / {h2.w}×{h2.h}）"
            "—— 这是「圆被一条直径线切开」的特征，按教材画法判为电压源。"
            "注意此特征与朝向无关：竖切给左右两半、横切给上下两半。")
    elif len(group) == 1:
        h = group[0]
        # ★ 判序：**填充率优先，长宽比辅助**。
        # 实测矩形框的洞把外接矩形填满（1.00），圆的洞在四角留空（0.79），
        # 这才是稳定的区分点。长宽比单独用不行 —— 一个接近正方形的电阻框
        # 长宽比只有 1.2，跟圆撞在一起。
        if h.fill >= FILL_RECT_MIN and RECT_ASPECT[0] <= h.aspect <= RECT_ASPECT[1]:
            hole_kind = "R"
            hole_conf = 0.85
            evidence.append(
                f"检出单个矩形洞（{h.w}×{h.h}，填充率 {h.fill:.2f} 接近填满、"
                f"长宽比 {h.aspect:.2f}）—— 符合矩形框电阻的特征。")
        elif h.fill <= FILL_CIRCLE_MAX and h.aspect <= CIRCLE_ASPECT_MAX:
            hole_kind = "?"
            hole_conf = 0.4
            evidence.append(
                f"检出单个近正方、四角留空的洞（{h.w}×{h.h}，长宽比 {h.aspect:.2f}，"
                f"填充率 {h.fill:.2f}）—— 是圆形符号的内部。"
                "但本项目的电压源是空圆、电流源是圆内带箭头，"
                "两者都是「1 个洞」，洞法分不开，交给模板判。")
        else:
            evidence.append(
                f"检出单个洞，但形状既不像矩形框也不像圆"
                f"（{h.w}×{h.h}，填充率 {h.fill:.2f}，长宽比 {h.aspect:.2f}）。")

    # ---- 模板：在洞的外接矩形（外扩一点，把笔画含进来）上比对
    x0, y0, x1, y1 = box
    pad = 3
    patch = mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    scores = match_kind(patch) if patch.size else {}
    template_kind = None
    if scores:
        best = max(scores.items(), key=lambda t: t[1])
        # 得分太低的"第一名"不算结论 —— 否则任何图块都会被迫认领一个符号
        if best[1] >= 0.45:
            template_kind = best[0]
        evidence.append("模板得分：" + "、".join(
            f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda t: -t[1])))
    return hole_kind, hole_conf, evidence, scores, template_kind


def _fuse(hole_kind: str | None, hole_conf: float,
          template_kind: str | None, scores: dict[str, float]
          ) -> tuple[str, float, bool, list[str]]:
    """把两种判据合起来。返回 ``(kind, 置信度, 是否一致, 追加依据)``。

    融合规则刻意保守：**两种判据都给出明确结论且一致**才给高置信度；
    相冲突或有一方沉默，一律降到门槛以下，逼人工看一眼。
    这不是"不够聪明"，而是这个项目的立场 —— 猜错一个元件类型，
    后面三法会一致地给出同一个错答案，互校拦不住。
    """
    extra: list[str] = []
    t_conf = max(scores.values()) if scores else 0.0

    if hole_kind and hole_kind != "?" and template_kind:
        if hole_kind == template_kind:
            extra.append(f"洞法与模板一致（{hole_kind}），互相印证。")
            return hole_kind, min(0.95, 0.5 * hole_conf + 0.5 * t_conf + 0.35), True, extra
        extra.append(
            f"★ 两种判据冲突：洞法说 {hole_kind}、模板说 {template_kind}。"
            f"已按洞法（对风格不敏感）取 {hole_kind}，"
            "但置信度压到门槛以下，请人工确认。")
        return hole_kind, 0.5, False, extra

    if hole_kind and hole_kind != "?":
        extra.append("只有洞法给出结论，模板未能确认（或没有模板可匹配），"
                     "置信度按洞法保守估计。")
        return hole_kind, min(0.8, hole_conf), False, extra

    if template_kind:
        extra.append(
            f"洞法无结论（这类符号是开放图形，本来就没有闭合空洞），"
            f"改由模板判为 {template_kind}，得分 {t_conf:.2f}。"
            "模板是逐像素判据，对透视与风格敏感，故置信度不设高。")
        return template_kind, min(0.6, t_conf), False, extra

    extra.append("两种判据都没给出结论。")
    return "?", 0.0, False, extra


# ---------------------------------------------------------------- 主入口


def _loop_like(hole_area: int, cw: int, ch: int) -> bool:
    """这个洞是不是"闭合导线回路围出来的空白"。

    判据与实测数据见 ``WIRE_LOOP_HOLE_RATIO`` / ``WIRE_LOOP_MIN_SIDE``。
    抽成一个函数是为了让**两个调用点用同一条判据**：
    ``_split_wire_loop_holes``（别把回路当元件）与
    ``_unexplained_clusters``（别把回路报成"未解释墨迹"）。
    两边各写一遍的后果是：回路被剔除之后，同一条回路又冒出来当"未解释墨迹"，
    用户会以为图上多了一块不明物体。
    """
    bb = max(1, cw * ch)
    return (hole_area / float(bb) > WIRE_LOOP_HOLE_RATIO
            and max(cw, ch) >= WIRE_LOOP_MIN_SIDE)


def _split_wire_loop_holes(mask, holes: list[Hole]
                           ) -> tuple[list[Hole], list[tuple[Hole, float]]]:
    """把"闭合导线回路围出来的空白"从真符号的洞里剔出去。

    返回 ``(保留的洞, [(剔掉的洞, 比值)])``。判据与实测数据见
    ``WIRE_LOOP_HOLE_RATIO`` —— 这里只解释为什么必须做这一步：

    ★ 整张电路图的墨迹**本来就是一个连通块**（元件本体接着引线、引线接着导线），
    所以一圈导线围出来的空白与一个电阻框的内部空白，在只看几何时**完全同形**；
    它们的差别只在"这个洞是不是几乎占满了它所在的那块墨迹"。
    不做这一步的实测后果很严重：一张普通串联回路图会被定位出一个
    ``R 362x241`` 的假槽位，它的 ``slot_region`` 覆盖整张图，
    于是导线追踪阶段会把全图墨迹当"元件本体"剔掉，得到**0 条线段、0 个结点** ——
    而且不报任何错。
    """
    import numpy as np
    from scipy import ndimage

    if not holes:
        return [], []
    h_img, w_img = mask.shape
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return list(holes), []
    objs = ndimage.find_objects(lbl)

    kept: list[Hole] = []
    dropped: list[tuple[Hole, float]] = []
    for hole in holes:
        x0, y0 = max(0, hole.x - 2), max(0, hole.y - 2)
        x1, y1 = min(w_img, hole.x + hole.w + 2), min(h_img, hole.y + hole.h + 2)
        ring = lbl[y0:y1, x0:x1]
        vals, cnt = np.unique(ring[ring > 0], return_counts=True)
        if len(vals) == 0:
            kept.append(hole)
            continue
        sl = objs[int(vals[int(np.argmax(cnt))]) - 1]
        if sl is None:
            kept.append(hole)
            continue
        ys, xs = sl
        cw, ch = xs.stop - xs.start, ys.stop - ys.start
        # ★ 用的是**被检验的这个洞**的面积，不是"所在块里最大的洞"。
        # 这一条很容易写错且不报错：一张回路图里，电阻的洞与回路的洞
        # 处于同一个连通块，一旦拿"块里最大的洞"去判，
        # 电阻的洞会被连坐剔除 —— 电阻凭空消失。
        if _loop_like(hole.area, cw, ch):
            dropped.append((hole, hole.area / float(max(1, cw * ch))))
        else:
            kept.append(hole)
    return kept, dropped


def detect_symbols(image: Any, *, correct: bool = True) -> SymbolReport:
    """找出图里所有候选元件位。**永不抛异常**。

    ``correct=True`` 时对图做一次轻度归一（灰度 + 中值去噪 + 自适应二值化），
    这对手机照片很有用；对已经很干净的合成图没有坏处。
    """
    import numpy as np
    from PIL import Image

    rep = SymbolReport()
    try:
        im = image
        if not hasattr(im, "convert"):
            im = Image.open(im)
        if hasattr(im, "load"):
            im.load()
        if im.mode != "L":
            im = im.convert("L")
    except Exception as e:                     # noqa: BLE001
        rep.warnings.append(f"图片读取失败：{type(e).__name__}: {e}")
        return rep

    rep.image_size = im.size
    work = _normalize_image(im) if correct else im
    mask = ink_mask(work)
    # ★ 用**实测**笔画宽度，不用写死的常数。见 estimate_stroke_half 的长注释：
    # 写死 3 而实际线宽 4 时，本体外接矩形取不全，模板得分被压到 0.22，
    # 于是每个电阻都判成"模板沉默"、置信度卡在 0.80、每次都升级给视觉模型。
    stroke = max(2, int(round(estimate_stroke_half(mask))))

    holes_all = find_holes(mask)
    holes, wire_holes = _split_wire_loop_holes(mask, holes_all)
    rep.holes = holes
    rep.wire_holes = [h for h, _ in wire_holes]
    if wire_holes:
        biggest, big_ratio = wire_holes[0]
        rep.warnings.append(
            f"有 {len(wire_holes)} 个洞被判为「闭合导线回路围出来的空白」，没有当成元件。"
            f"最大那个 {biggest.w}×{biggest.h} 占了所在墨迹块的 {big_ratio:.2f} —— "
            "真符号的引线会把外接矩形撑出去，闭合回路不会。"
            "这一步不能省：不做的话整张回路会被认成一个大电阻，"
            "导线追踪会把全图墨迹当元件本体剔掉，最后得到 0 条线段、0 个结点，且不报错。")
    groups = group_holes(holes)

    slots: list[SymbolSlot] = []
    for group in groups:
        gx0 = min(h.x for h in group)
        gy0 = min(h.y for h in group)
        gx1 = max(h.x + h.w for h in group)
        gy1 = max(h.y + h.h for h in group)
        # 洞是"被围起来的空白"，本体比它多一圈笔画。外扩量取**实测**笔画宽度。
        box = (gx0 - stroke, gy0 - stroke, gx1 + stroke, gy1 + stroke)
        kind, conf, ev, scores, tk = _classify_group(group, mask, box)
        kind2, conf2, agreed, extra = _fuse(kind, conf, tk, scores)
        ev.extend(extra)
        w, h = box[2] - box[0], box[3] - box[1]
        if min(w, h) < SYMBOL_MIN_SIDE or max(w, h) > SYMBOL_MAX_SIDE:
            continue
        slots.append(SymbolSlot(
            x=int(box[0]), y=int(box[1]), w=int(w), h=int(h),
            kind=kind2, orientation=("h" if w >= h else "v"),
            confidence=conf2, evidence=ev, holes=group, hole_kind=kind,
            template_scores=scores, template_kind=tk, agreed=agreed))

    # ---- 没有洞的符号（C / L / 电池式电压源）只能靠模板
    slots.extend(_detect_open_symbols(mask, slots, rep))

    slots.sort(key=lambda s: (s.y, s.x))
    rep.slots = slots

    # ---- 兜底：既不是导线、又没被任何槽位解释掉的大块墨迹，必须报出来
    rep.unexplained = _unexplained_clusters(mask, slots)

    n_unknown = sum(1 for s in slots if s.kind == "?")
    n_unsure = sum(1 for s in slots if s.needs_human)
    if slots:
        rep.warnings.append(
            f"定位到 {len(slots)} 个候选元件位，其中 {n_unsure} 个置信度不足"
            f"需人工确认（{n_unknown} 个类型完全无法判定）。")
    else:
        rep.warnings.append(
            "一个候选元件位都没定位到。可能原因：图里没有闭合符号图形、"
            "线宽过细导致形态学运算断线、或图片对比度太低。"
            "建议改用视觉模型重读这张图。")
    if rep.unexplained:
        rep.warnings.append(
            f"另有 {len(rep.unexplained)} 块墨迹既不像导线、也没被任何候选位解释"
            "（可能是没认出来的元件、也可能是图框或标题栏）。已逐块列出坐标，请核对。")
    return rep


def _normalize_image(im):
    """轻度归一：去噪 + 自适应二值化。返回 L 模式图。

    只做**不会改变几何**的操作 —— 不旋转、不透视校正、不缩放。
    理由是这一层的输出要拿去定位元件、进而算导线端点，
    任何几何变换都会让坐标和原图对不上，叠图核对就没意义了。
    """
    import numpy as np
    from PIL import Image, ImageFilter

    den = im.filter(ImageFilter.MedianFilter(size=3))
    a = np.array(den).astype(np.float32)
    # 局部均值作阈值（盒式滤波近似自适应二值化），比全局阈值抗阴影
    local = np.array(Image.fromarray(a.astype(np.uint8))
                     .filter(ImageFilter.BoxBlur(15))).astype(np.float32)
    out = np.where(a < local - 8, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _group_components(mask, *, gap_ratio: float = 0.75):
    """把邻近的连通块归成一组，返回 ``[(union_box, union_mask, member_count)]``。

    ★ 为什么必须有这一步：**有些元件天然由多个互不相连的墨迹块组成**。
    电容的两片极板电气上就是分开的（这正是电容的定义），
    每片极板加各自那段引线各自成块；电池式电压源的长短线也是两块。
    第一版逐块做模板匹配，结果一个电容被拆成两个"?"槽位 ——
    比认不出更糟，因为它凭空多出两个元件。

    分组判据：外接矩形之间的间隙小于较短边的一定比例。
    只做"最近邻"式的贪心合并，不做传递闭包 —— 否则沿着导线一路并下去，
    整张图会被并成一组。
    """
    import numpy as np
    from scipy import ndimage

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    boxes: list[tuple[int, int, int, int, int]] = []   # x0,y0,x1,y1,label
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        ys, xs = sl
        boxes.append((xs.start, ys.start, xs.stop, ys.stop, i))
    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            ax0, ay0, ax1, ay1, _ = boxes[a]
            bx0, by0, bx1, by1, _ = boxes[b]
            gap_x = max(ax0, bx0) - min(ax1, bx1)
            gap_y = max(ay0, by0) - min(ay1, by1)
            gap = max(gap_x, gap_y)
            if gap < 0:
                gap = 0
            size = min(min(ax1 - ax0, ay1 - ay0), min(bx1 - bx0, by1 - by0))
            if size <= 0:
                continue
            if gap <= gap_ratio * size:
                union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        sel = [boxes[i] for i in members]
        x0 = min(b[0] for b in sel)
        y0 = min(b[1] for b in sel)
        x1 = max(b[2] for b in sel)
        y1 = max(b[3] for b in sel)
        sub = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for bx0, by0, bx1, by1, lab in sel:
            sub[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] |= (lbl[by0:by1, bx0:bx1] == lab)
        out.append(((x0, y0, x1, y1), sub, len(sel)))
    return out


def _detect_open_symbols(mask, found: list[SymbolSlot],
                         rep: SymbolReport) -> list[SymbolSlot]:
    """在还没被解释的墨迹里找开放图形（电容、电感、电池式电压源）。

    ★ **刻意不按长宽比预筛**。第一版在这里写了"长宽比 > 6 就当导线跳过"，
    结果电感被直接丢掉 —— 可电感本来就是细长的（弧串加引线，
    长宽比轻松过 6），电容的极板也常被拉长。那条预筛在矢量通道或许合适，
    在这里是错的判据。

    模板得分分两档：
    - 得分 >= ``OPEN_KIND_MIN``：给出具体类型，但仍标需人工确认；
    - 得分在 ``OPEN_ACCEPT`` 与 ``OPEN_KIND_MIN`` 之间：只认"这里有个元件"，
      类型留 ``?`` 并把候选与得分都列出来。
      实测这个区间最有用：电容两片极板的间距与极板长度之比在不同画法下
      差异很大，同一条模板会给出 0.37~0.62 的分数，硬判类型会错。
    """
    import numpy as np

    out: list[SymbolSlot] = []
    taken = np.zeros(mask.shape, dtype=bool)
    for s in found:
        x0, y0, x1, y1 = s.slot_region(margin=2)
        taken[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True

    for (bx0, by0, bx1, by1), sub, nmemb in _group_components(mask):
        w, h = bx1 - bx0, by1 - by0
        if min(w, h) < SYMBOL_MIN_SIDE or max(w, h) > SYMBOL_MAX_SIDE:
            continue
        area = int(sub.sum())
        if area < 60:
            continue

        # ---- 闸门一：这块墨迹是不是"导线织物"？
        # 与"洞法"那侧的 _split_wire_loop_holes 是同一条判据的两种表现：
        # 闭合回路靠"洞占满了所在墨迹块"，稀疏网线靠"填充率极低"。
        # 两道都**刻意偏向剔除**：漏放一块导线织物的后果是凭空多出一个元件，
        # 而误杀一个真元件的后果只是本地层承认看不清、升级给视觉模型。
        bbox_area = max(1, w * h)
        inner = [hh for hh in find_holes(sub) if hh.area >= 30]
        biggest_inner = max((hh.area for hh in inner), default=0)
        inner_ratio = biggest_inner / bbox_area
        if inner_ratio > WIRE_LOOP_HOLE_RATIO and max(w, h) >= WIRE_LOOP_MIN_SIDE:
            continue
        if area / bbox_area < WIRE_NETWORK_FILL:
            continue

        free = int((sub & ~taken[by0:by1, bx0:bx1]).sum())
        if free / max(1, area) < 0.3:
            continue                            # 大部分已被别的槽位占了
        scores = match_kind(sub)
        if not scores:
            continue
        best, sc = max(scores.items(), key=lambda t: t[1])
        if sc < OPEN_ACCEPT:
            continue
        kind = best if sc >= OPEN_KIND_MIN else "?"
        ev = ["这类符号是开放图形，没有闭合空洞，只能靠模板判。",
              "模板得分：" + "、".join(
                  f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda t: -t[1]))]
        if nmemb > 1:
            ev.append(
                f"这一组由 {nmemb} 块互不相连的墨迹合成 —— 这是正常的："
                "电容的两片极板电气上本来就分开，电池式电压源的长短线也是两块。"
                "逐块看会把一个元件拆成好几个，所以必须先归组再匹配。")
        if kind == "?":
            ev.append(
                f"最高分 {best}({sc:.2f}) 未达到直接判型的门槛 {OPEN_KIND_MIN}，"
                "只认「这里有个元件」。这类图形的画法差异很大（例如电容的极板"
                "间距与长度之比），硬判类型会错，所以留给你确认。")
        else:
            ev.append("★ 模板匹配对透视与线宽敏感，这个结论必须人工核对。")
        out.append(SymbolSlot(
            x=int(bx0), y=int(by0), w=int(w), h=int(h),
            kind=kind, orientation=("h" if w >= h else "v"),
            confidence=min(0.6, sc), evidence=ev,
            template_scores=scores, template_kind=best))
    return out


def _unexplained_clusters(mask, slots: list[SymbolSlot]) -> list[dict[str, Any]]:
    """找出"大块墨迹但没被任何槽位解释掉"的连通块。

    两类**必须排除**，否则每张图都会满屏假报警：
    - **长条**（长宽比过大）→ 是导线。
    - **无洞且填充率极低** → 是导线网络。这一条是补长宽比的漏：
      两根正交导线连成一个连通块时，整体长宽比可能只有 5.8，
      躲得过上面那条筛选，但它的填充率只有 0.025 左右 —— 一片稀疏的细线。
      元件（含电容这种开放图形）的填充率都在 0.1 以上。
    """
    import numpy as np
    from scipy import ndimage

    taken = np.zeros(mask.shape, dtype=bool)
    for s in slots:
        x0, y0, x1, y1 = s.slot_region(margin=4)
        taken[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    out: list[dict[str, Any]] = []
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        comp = lbl[sl] == i
        area = int(comp.sum())
        # 只关心"够大、可能是个元件"的那些。导线和碎笔画不算。
        if min(w, h) < 10 or area < 200:
            continue
        if max(w, h) / float(max(1, min(w, h))) > WIRE_ASPECT:
            continue
        local = ndimage.binary_fill_holes(comp)
        has_hole = bool((local & ~comp).any())
        fill = area / float(w * h)
        if not has_hole and fill < WIRE_NETWORK_FILL:
            continue                            # 细线网 → 导线，不是元件
        # ★ 有洞的也要排除：闭合导线回路**必然有洞**，而上面那条只在"无洞"时生效，
        # 于是整圈导线会以"未解释墨迹"的身份冒出来 —— 而且它刚刚才在
        # _split_wire_loop_holes 里被判为导线回路剔掉过一次，等于同一件事报两遍。
        if has_hole:
            holes_here = [hh for hh in find_holes(comp) if hh.area >= 30]
            if holes_here and _loop_like(max(h.area for h in holes_here), w, h):
                continue
        free = int((comp & ~taken[sl]).sum())
        if free / max(1, area) < 0.5:
            continue
        item: dict[str, Any] = {"x": int(xs.start), "y": int(ys.start),
                                "w": int(w), "h": int(h), "area": area,
                                "fill": round(fill, 3), "has_hole": has_hole,
                                "free_ratio": round(free / float(max(1, area)), 2)}
        blob = solid_blob_reason(comp)
        if blob:
            item["note"] = (blob + " 不像任何已知元件符号 —— "
                             "可能是涂黑的图形、水印、或者图上的一块阴影。")
        out.append(item)
    # 按面积从大到小，界面上先看最大的那块
    out.sort(key=lambda d: -d["area"])
    return out


def probe() -> dict[str, Any]:
    """给 /api/health 用的体检。只报模板库状态，不读图。"""
    from ..ir.model import ALLOWED_KINDS
    out = {"templates": [], "kinds": sorted(ALLOWED_KINDS)}
    for k in sorted(ALLOWED_KINDS):
        for v in (False, True):
            m = T.template_mask(k, v)
            out["templates"].append({"kind": k, "orientation": "v" if v else "h",
                                     "shape": list(m.shape),
                                     "ink_ratio": round(float(m.mean()), 3)})
    return out
