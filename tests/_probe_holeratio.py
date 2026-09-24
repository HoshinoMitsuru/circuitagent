# -*- coding: utf-8 -*-
"""探针：找一条能分开"真符号的洞"与"导线回路的洞"的不变量。

候选不变量： hole.area / (包含该洞的那个连通块的墨迹面积)
物理含义：真符号是"粗笔画围出小洞"，导线回路是"一根细线围出巨大空白"。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from app.vision import symbols as SY

W = 4


def analyze(im, label):
    m = SY.ink_mask(im)
    H, Wd = m.shape
    lbl, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))
    holes = SY.find_holes(m)
    # 每个洞落在哪个连通块里：取洞外扩一圈后的多数标签
    hlbl = np.zeros_like(lbl)
    rows = []
    for h in holes:
        if h.area < 30:
            continue
        x0, y0 = max(0, h.x - 3), max(0, h.y - 3)
        x1, y1 = min(Wd, h.x + h.w + 3), min(H, h.y + h.h + 3)
        ring = lbl[y0:y1, x0:x1]
        vals, cnt = np.unique(ring[ring > 0], return_counts=True)
        comp = int(vals[np.argmax(cnt)]) if len(vals) else 0
        cink = int((lbl == comp).sum()) if comp else 0
        cb = ndimage.find_objects(lbl)[comp - 1] if comp else None
        cbox = ((cb[1].stop - cb[1].start) * (cb[0].stop - cb[0].start)) if cb else 0
        rows.append((h, comp, cink, cbox, h.area / max(1, cink), h.area / max(1, cbox)))
    rows.sort(key=lambda r: -r[0].area)
    print(f"-- {label}  图 {Wd}x{H}  连通块 {n}  洞 {len(rows)} --")
    for h, comp, cink, cbox, r_ink, r_bbox in rows:
        print(f"     洞 {h.w:4d}x{h.h:<4d} area={h.area:7d} fill={h.fill:4.2f} "
              f"aspect={h.aspect:5.2f} | 所在块[{comp}] 墨迹={cink:7d} bbox={cbox:8d} "
              f"| 洞/墨迹={r_ink:7.2f}  洞/块bbox={r_bbox:6.3f}"
              f"  洞长边/图短边={max(h.w,h.h)/min(Wd,H):6.3f}")
    print()


def img(w, h, draw):
    im = Image.new("RGB", (w, h), "white")
    draw(ImageDraw.Draw(im))
    return im


# 1 串联回路（整张图一个块，洞=整条回路的内部）
def d1(d):
    d.rectangle([260, 105, 340, 135], outline="black", width=W)
    d.line([120, 120, 260, 120], fill="black", width=W)
    d.line([340, 120, 480, 120], fill="black", width=W)
    d.line([480, 120, 480, 360], fill="black", width=W)
    d.line([480, 360, 120, 360], fill="black", width=W)
    d.ellipse([95, 215, 145, 265], outline="black", width=W)
    d.line([120, 120, 120, 215], fill="black", width=W)
    d.line([120, 265, 120, 360], fill="black", width=W)


analyze(img(620, 480, d1), "串联回路（整图一块）")


# 2 简单串联回路 V+R
def d2(d):
    d.ellipse([210, 30, 260, 80], outline="black", width=W)
    d.line([40, 55, 210, 55], fill="black", width=W)
    d.line([260, 55, 440, 55], fill="black", width=W)
    d.line([440, 55, 440, 340], fill="black", width=W)
    d.line([440, 340, 40, 340], fill="black", width=W)
    d.line([40, 340, 40, 55], fill="black", width=W)
    d.rectangle([180, 325, 300, 355], outline="black", width=W)


analyze(img(500, 400, d2), "串联回路(V+R)")


# 3 只有 R，带引线（真符号）
def d3(d):
    d.rectangle([80, 45, 160, 75], outline="black", width=W)
    d.line([0, 60, 80, 60], fill="black", width=W)
    d.line([160, 60, 240, 60], fill="black", width=W)


analyze(img(240, 120, d3), "R-框+引线（孤立）")


# 4 只有 V 空圆，带引线（真符号）
def d4(d):
    d.ellipse([95, 45, 145, 95], outline="black", width=W)
    d.line([120, 0, 120, 45], fill="black", width=W)
    d.line([120, 95, 120, 160], fill="black", width=W)


analyze(img(240, 160, d4), "V-空圆+引线（孤立）")


# 5 很大的 R（画得大）—— 检验不变量是否与缩放无关
def d5(d):
    d.rectangle([80, 180, 560, 300], outline="black", width=4)
    d.line([0, 240, 80, 240], fill="black", width=4)
    d.line([560, 240, 640, 240], fill="black", width=4)


analyze(img(640, 480, d5), "R-大框+引线（放大 6 倍）")


# 6 导线粗一点的回路（线宽 8）—— 检验是否与线宽有关
def d6(d):
    d.rectangle([160, 130, 480, 350], outline="black", width=8)


analyze(img(640, 480, d6), "细线闭合回形（线宽8，无元件）")


# 7 纯导线十字
def d7(d):
    d.line([60, 250, 440, 250], fill="black", width=4)
    d.line([250, 60, 250, 440], fill="black", width=4)


analyze(img(500, 500, d7), "纯导线十字")

print("=" * 100)
print("结论区：看「洞/墨迹」与「洞长边/图短边」两列能不能把真符号与导线回路分开")
print("=" * 100)
