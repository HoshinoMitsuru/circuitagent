# -*- coding: utf-8 -*-
"""探针：量"合法元件组"与"导线织物组"的几何特征差在哪 —— 为阈值找数据。

不许凭感觉写常数。这里把候选特征全打出来，再决定用哪一条。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from app.vision import symbols as SY

W = 4


def feat(mask, box, label):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    sub = mask[y0:y1, x0:x1]
    area = int(sub.sum())
    bbox_area = w * h
    fill = area / max(1, bbox_area)

    # 粗估笔画宽度：面积 / 骨架长度
    try:
        from skimage.morphology import skeletonize  # noqa
        sk = skeletonize(sub)
        slen = int(sk.sum())
    except Exception:
        # 没有 skimage 就用周长粗估：stroke ≈ 2*area/perimeter
        slen = 0
    stroke = (area / slen) if slen else float("nan")

    # 空洞
    holes = SY.find_holes(sub)
    big = [hh for hh in holes if hh.area >= 30]
    biggest = max((hh.area for hh in big), default=0)
    holes_fill = biggest / max(1, bbox_area)

    solid = SY.solidity(sub)
    print(f"{label:24s} bbox={w:4d}x{h:<4d} aspect={max(w,h)/max(1,min(w,h)):5.2f} "
          f"area={area:7d} fill={fill:5.3f} n_hole={len(big):2d} "
          f"hole/bbox={holes_fill:5.3f} solid={solid:5.3f} "
          f"skel={slen:5d} stroke={stroke:5.2f} "
          f"maxside/stroke={max(w,h)/stroke if stroke == stroke and stroke else float('nan'):7.1f}")


def ink(im):
    return SY.ink_mask(im)


print("=" * 100)
print("A. 串联回路（整张图一个连通块 = 导线织物）")
print("=" * 100)
im = Image.new("RGB", (620, 480), "white")
d = ImageDraw.Draw(im)
d.rectangle([260, 105, 340, 135], outline="black", width=W)
d.line([120, 120, 260, 120], fill="black", width=W)
d.line([340, 120, 480, 120], fill="black", width=W)
d.line([480, 120, 480, 360], fill="black", width=W)
d.line([480, 360, 120, 360], fill="black", width=W)
d.ellipse([95, 215, 145, 265], outline="black", width=W)
d.line([120, 120, 120, 215], fill="black", width=W)
d.line([120, 265, 120, 360], fill="black", width=W)
m = ink(im)
lbl, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))
print(f"连通块总数 = {n}")
for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
    if sl is None:
        continue
    ys, xs = sl
    print(f"  块 {i}: x[{xs.start},{xs.stop}) y[{ys.start},{ys.stop}) 面积={int((lbl[sl]==i).sum())}")
print()
for (bx0, by0, bx1, by1), sub, nm in SY._group_components(m):
    feat(m, (bx0, by0, bx1, by1), f"回路组(成员{nm})")

print()
print("=" * 100)
print("B. 各元件单独画（带短引线）")
print("=" * 100)
cases = {}

# B1 电阻框 + 引线
b1 = Image.new("RGB", (240, 120), "white")
d1 = ImageDraw.Draw(b1)
d1.rectangle([80, 45, 160, 75], outline="black", width=W)
d1.line([0, 60, 80, 60], fill="black", width=W)
d1.line([160, 60, 240, 60], fill="black", width=W)
cases["R-框+引线"] = b1

# B2 电压源空圆 + 引线
b2 = Image.new("RGB", (240, 160), "white")
d2 = ImageDraw.Draw(b2)
d2.ellipse([95, 45, 145, 95], outline="black", width=W)
d2.line([120, 0, 120, 45], fill="black", width=W)
d2.line([120, 95, 120, 160], fill="black", width=W)
cases["V-空圆+引线"] = b2

# B3 电容（两片极板 + 引线）
b3 = Image.new("RGB", (240, 200), "white")
d3 = ImageDraw.Draw(b3)
d3.line([80, 60, 160, 60], fill="black", width=W)     # 上极板
d3.line([80, 80, 160, 80], fill="black", width=W)     # 下极板
d3.line([120, 0, 120, 60], fill="black", width=W)
d3.line([120, 80, 120, 200], fill="black", width=W)
cases["C-两片极板+引线"] = b3

# B4 电感（四段弧 + 引线）
b4 = Image.new("RGB", (280, 120), "white")
d4 = ImageDraw.Draw(b4)
for k in range(4):
    cx = 110 + k * 20
    d4.arc([cx - 10, 50, cx + 10, 70], start=180, end=360, fill="black", width=W)
d4.line([90, 60, 90, 60], fill="black", width=W)
d4.line([70, 60, 110, 60], fill="black", width=W)
d4.line([190, 60, 230, 60], fill="black", width=W)
cases["L-弧串+引线"] = b4

# B5 电池式电压源（长短线 + 引线）
b5 = Image.new("RGB", (240, 200), "white")
d5 = ImageDraw.Draw(b5)
d5.line([80, 60, 160, 60], fill="black", width=W)     # 长线
d5.line([100, 80, 140, 80], fill="black", width=W)    # 短线
d5.line([120, 0, 120, 60], fill="black", width=W)
d5.line([120, 80, 120, 200], fill="black", width=W)
cases["V-电池长短线"] = b5

# B6 两根正交导线（纯导线织物，无元件）
b6 = Image.new("RGB", (500, 500), "white")
d6 = ImageDraw.Draw(b6)
d6.line([60, 250, 440, 250], fill="black", width=W)
d6.line([250, 60, 250, 440], fill="black", width=W)
cases["纯导线十字"] = b6

# B7 一个简单的串联回路（V + R，无第三条线）
b7 = Image.new("RGB", (500, 400), "white")
d7 = ImageDraw.Draw(b7)
d7.ellipse([210, 30, 260, 80], outline="black", width=W)
d7.line([40, 55, 210, 55], fill="black", width=W)
d7.line([260, 55, 440, 55], fill="black", width=W)
d7.line([440, 55, 440, 340], fill="black", width=W)
d7.line([440, 340, 40, 340], fill="black", width=W)
d7.line([40, 340, 40, 55], fill="black", width=W)
d7.rectangle([180, 325, 300, 355], outline="black", width=W)
cases["串联回路(V+R)"] = b7

for name, img in cases.items():
    mm = ink(img)
    print(f"-- {name} --")
    groups = SY._group_components(mm)
    for (bx0, by0, bx1, by1), sub, nm in groups:
        feat(mm, (bx0, by0, bx1, by1), f"   组(成员{nm})")
    if not groups:
        print("   （无组）")
    print()
