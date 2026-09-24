"""符号模板：从项目**自己的渲染器常量**生成，供位图通道做形状匹配。

为什么不直接拿 ``app/ir/render.py`` 输出的 SVG 当模板：
项目里没有 SVG 光栅化器（cairosvg 之类），装一个只为生成模板不划算。
折中的做法是**在 PIL 里按同一批常量重画一遍**：

- 本体的半长直接取 ``render.BODY`` / ``render.BODY_W`` / ``render.BODY_HALF``，
  不复制数字。这样渲染器将来改尺寸，模板会跟着改，不会悄悄失配。
- 画法逐条对照 ``_glyph_resistor`` / ``_glyph_voltage`` / ``_glyph_current`` /
  ``_glyph_capacitor`` / ``_glyph_inductor``，包括"极性笔画已删除"这类历史决定。

**模板匹配在这一层只是辅助，不是主判据。** 原因很实在：真实照片有透视、
手绘有抖、印刷有各自的字体与线宽，逐像素形状匹配很容易失手；
而"闭合空洞"是**拓扑**性质，对风格变化不敏感。所以主力是
``symbols.find_holes``，模板只用来在几个形状相近的候选之间分胜负
（典型是空圆到底是电压源还是电流源），以及给置信度打分。
"""

from __future__ import annotations

import functools
from typing import Iterable

#: 超采样倍数。先按大尺寸画再缩回来，笔画才有抗锯齿的灰度，
#: 否则二值化后单像素宽的圆会到处断线。
SUPERSAMPLE = 4

#: 模板画布在**本体 bbox** 之外额外留的边（单位：渲染器坐标）。
#: 留一点是为了让"本体引线"（比如电容那两小段）能画进去。
PAD = 6.0


def _import_render():
    from ..ir import render
    return render


def glyph_canvas(kind: str) -> tuple[float, float, float, float]:
    """返回某个符号在渲染器坐标下的本体范围 ``(x0, y0, x1, y1)``（水平朝向）。

    ★ 电压源/电流源**不把 ± 符号算进范围**。渲染器把它们画在圆**外**，
    而位图通道要的是"本体圆柱"（用来剔除体内的导线线段）。
    如果按文字包围盒算范围，圆柱会大一圈，把紧邻的导线也剔掉 ——
    那就把元件两端所在的那段导线删了，拓扑直接断掉。
    """
    r = _import_render()
    half = r.BODY_HALF
    if kind == "R":
        return (-r.BODY, -r.BODY_W, r.BODY, r.BODY_W)
    if kind in ("V", "I"):
        h = half[kind]
        return (-h, -h, h, h)
    if kind == "C":
        # 极板 y=±11，本体引线 x=±12
        return (-12.0, -11.0, 12.0, 11.0)
    if kind == "L":
        h = half["L"]
        # 弧串的半高 = 弧半径 = seg/2，seg = 2*BODY/4
        seg = (2 * r.BODY) / 4
        return (-h, -seg / 2, h, seg / 2)
    raise ValueError(f"没有 {kind!r} 这个符号的画法")


def draw_glyph(kind: str, *, vertical: bool = False) -> "object":
    """把某个符号画成 PIL 的 L 模式图（0=墨迹，255=底）。

    画布正好是本体 bbox（加 PAD），所以缩放到候选框时不会因为留白比例不同
    而整体失配。
    """
    from PIL import Image, ImageDraw

    r = _import_render()
    x0, y0, x1, y1 = glyph_canvas(kind)
    bw = (x1 - x0) + 2 * PAD
    bh = (y1 - y0) + 2 * PAD
    if vertical:
        bw, bh = bh, bw

    # 渲染器里线宽是 2（电容极板 2.4）。超采样后按比例放大。
    lw = 2.4 if kind == "C" else 2.0
    W = int(round(bw * SUPERSAMPLE))
    H = int(round(bh * SUPERSAMPLE))
    im = Image.new("L", (max(W, 4), max(H, 4)), 255)
    d = ImageDraw.Draw(im)

    def T(x: float, y: float) -> tuple[float, float]:
        """渲染器坐标 → 画布像素坐标（含 PAD 偏移、超采样、垂直朝向旋转）。"""
        px = (x - x0 + PAD)
        py = (y - y0 + PAD)
        if vertical:
            px, py = py, px          # 沿对角线翻一次 = 旋转 90°
        return px * SUPERSAMPLE, py * SUPERSAMPLE

    w = max(1, int(round(lw * SUPERSAMPLE)))

    if kind == "R":
        p0, p1 = T(x0, y0), T(x1, y1)
        d.rectangle([p0, p1], outline=0, width=w)
    elif kind == "V":
        h = r.BODY_HALF["V"]
        p0, p1 = T(-h, -h), T(h, h)
        d.ellipse([p0, p1], outline=0, width=w)
        # 极性标注画在圆外，用短横代替文字（模板只比形状，不比字）
        # 不画：见 glyph_canvas 的说明，标不参与形状匹配
    elif kind == "I":
        h = r.BODY_HALF["I"]
        p0, p1 = T(-h, -h), T(h, h)
        d.ellipse([p0, p1], outline=0, width=w)
        # 圈内箭头：M -8 0 L 4 0 加箭头尖（0,-4)→(4,0)→(0,4)
        a, b = T(-8, 0), T(4, 0)
        d.line([a, b], fill=0, width=max(1, w - 1))
        for p, q in ((T(0, -4), T(4, 0)), (T(4, 0), T(0, 4))):
            d.line([p, q], fill=0, width=max(1, w - 1))
    elif kind == "C":
        # 本体引线 ±12 → ±4
        for xa, xb in ((-12, -4), (4, 12)):
            d.line([T(xa, 0), T(xb, 0)], fill=0, width=w)
        # 两片极板
        plate = max(1, int(round(lw * SUPERSAMPLE * 1.2)))
        for x in (-4, 4):
            d.line([T(x, -11), T(x, 11)], fill=0, width=plate)
    elif kind == "L":
        seg = (2 * r.BODY) / 4
        rad = seg / 2
        for xa, xb in ((-r.BODY_HALF["L"], -r.BODY), (r.BODY, r.BODY_HALF["L"])):
            d.line([T(xa, 0), T(xb, 0)], fill=0, width=w)
        for i in range(4):
            sx = -r.BODY + i * seg
            p0, p1 = T(sx, -rad), T(sx + seg, rad)
            # PIL 的 arc 角度：0°=3 点钟方向，顺时针。半圆在上方 → 180→360
            d.arc([p0, p1], start=180, end=360, fill=0, width=w)
    else:                                      # pragma: no cover
        raise ValueError(f"没有 {kind!r} 这个符号的画法")

    return im


@functools.lru_cache(maxsize=32)
def template_mask(kind: str, vertical: bool = False):
    """返回二值化的模板掩码（True = 墨迹），形状 ``(h, w)``。

    带缓存：一张图里同一个符号可能要被比对几十次，重画一遍纯属浪费。
    返回的数组**只读**，调用方不要原地改它（缓存会串味）。
    """
    import numpy as np

    im = draw_glyph(kind, vertical=vertical)
    m = np.array(im) < 160
    m.setflags(write=False)
    return m


def all_templates(kinds: Iterable[str] | None = None,
                  verticals: Iterable[bool] = (False, True)) -> dict[str, "object"]:
    """所有模板。键形如 ``"R/h"`` / ``"V/v"``。"""
    from ..ir.model import ALLOWED_KINDS
    ks = list(kinds or sorted(ALLOWED_KINDS))
    out = {}
    for k in ks:
        for v in verticals:
            out[f"{k}/{'v' if v else 'h'}"] = template_mask(k, v)
    return out


def save_sheet(path: "str") -> "str":
    """把全部模板拼成一张对照图，便于人工核对模板画得对不对。"""
    from PIL import Image

    tiles = []
    from ..ir.model import ALLOWED_KINDS
    for k in sorted(ALLOWED_KINDS):
        for v in (False, True):
            im = draw_glyph(k, vertical=v).convert("L")
            tiles.append((f"{k}/{'v' if v else 'h'}", im))
    W = max(t[1].width for t in tiles) + 16
    H = sum(t[1].height + 22 for t in tiles) + 8
    sheet = Image.new("L", (W, H), 255)
    y = 8
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for name, im in tiles:
        d.text((8, y), name, fill=0)
        sheet.paste(im, (8, y + 16))
        y += im.height + 22
    sheet.save(path)
    return path
