# -*- coding: utf-8 -*-
"""探针：纵向膨胀把"位号行"和"数值行"并成一条 OCR 行，能否救回被丢弃的短 token。

若成立，这就是本地层的核心预处理手段：
- 膨胀**只用于 OCR**，不改变任何几何量（词的 bbox 仍是原图坐标），
  所以拓扑判断完全不受影响。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import Buffer

FONT = "C:/Windows/Fonts/arial.ttf"
F40 = ImageFont.truetype(FONT, 40)


def ocr(im):
    rgba = im.convert("RGBA")
    px = rgba.tobytes()
    b = Buffer(len(px))
    b.length = len(px)
    memoryview(b)[:] = px
    sb = SoftwareBitmap.create_copy_from_buffer(b, BitmapPixelFormat.RGBA8,
                                                im.size[0], im.size[1])
    eng = OcrEngine.try_create_from_language(Language("zh-Hans-CN"))
    res = eng.recognize_async(sb).get()
    out = []
    for line in res.lines:
        ws = [(w.text, (round(w.bounding_rect.x), round(w.bounding_rect.y),
                        round(w.bounding_rect.width), round(w.bounding_rect.height)))
              for w in line.words]
        out.append((line.text, ws))
    return out


def dilate(im, kx=1, ky=1):
    """方向可控的膨胀。kx/ky 是结构元尺寸（1 = 该方向不膨胀）。"""
    a = np.array(im.convert("L")) < 200          # 墨迹 = True
    if ky > 1:                                   # 纵向：合并上下相邻行
        a = np.logical_or.reduce(
            [np.roll(a, s, axis=0) for s in range(-(ky // 2), ky // 2 + 1)])
    if kx > 1:                                   # 横向：连接断裂笔画
        a = np.logical_or.reduce(
            [np.roll(a, s, axis=1) for s in range(-(kx // 2), kx // 2 + 1)])
    return Image.fromarray(np.where(a, 0, 255).astype(np.uint8)).convert("RGB")


# ---- 真实版式：电阻框，位号在上、数值在下
def schematic(gap=150):
    im = Image.new("RGB", (760, 460), "white")
    d = ImageDraw.Draw(im)
    y0 = 120
    d.rectangle([260, y0, 420, y0 + 60], outline="black", width=3)
    d.line([120, y0 + 30, 260, y0 + 30], fill="black", width=3)
    d.line([420, y0 + 30, 600, y0 + 30], fill="black", width=3)
    d.text((300, y0 - 58), "R1", fill="black", font=F40)       # 位号在上
    d.text((285, y0 + gap - 40), "10k", fill="black", font=F40)  # 数值在下
    return im


base = schematic()
print("=" * 74)
print("0. 基线（不膨胀）")
print("=" * 74)
for t, ws in ocr(base):
    print(f"   {t!r}   词={ws}")

print()
print("=" * 74)
print("1. 只做纵向膨胀，扫 ky")
print("=" * 74)
for ky in (3, 5, 9, 15, 25, 41, 61):
    im = dilate(base, ky=ky)
    res = ocr(im)
    got = [t for t, _ in res]
    print(f"   ky={ky:3d}: {got}")
    for t, ws in res:
        print(f"            {t!r} 词={ws}")

print()
print("=" * 74)
print("2. 纵向 + 少量横向膨胀")
print("=" * 74)
for ky, kx in [(25, 3), (41, 3), (41, 5), (61, 3)]:
    im = dilate(base, kx=kx, ky=ky)
    res = ocr(im)
    print(f"   ky={ky} kx={kx}: {[t for t, _ in res]}")

print()
print("=" * 74)
print("3. 把数值放得更近 / 更远，看合并阈值")
print("=" * 74)
for gap in (80, 100, 130, 170, 220):
    im = schematic(gap=gap)
    for ky in (15, 25, 41):
        res = ocr(dilate(im, ky=ky))
        print(f"   gap={gap:3d} ky={ky:2d} -> {[t for t, _ in res]}")

print()
print("=" * 74)
print("4. 关键：膨胀后词的 bbox 是否仍在原图坐标系（必须'是'，否则几何被破坏）")
print("=" * 74)
im = dilate(base, ky=25)
for t, ws in ocr(im):
    for txt, r in ws:
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        print(f"   {txt!r:8s} bbox={r}  中心=({cx:.0f},{cy:.0f})")
print("   原图: R1 画在 y≈62(中心约82)，10k 画在 y≈230(中心约250)，电阻框 y=120..180")
print("   -> 若上面 bbox 的 y 仍分别落在 ~62-100 与 ~230-270，说明坐标系没被破坏")

print()
print("=" * 74)
print("5. 膨胀会不会把不该合并的合了（邻近行误合并风险）")
print("=" * 74)
im = Image.new("RGB", (900, 700), "white")
d = ImageDraw.Draw(im)
for i, t in enumerate(["R1", "10k", "C1", "100uF", "V1", "12V", "L1", "2.2mH"]):
    d.text((60 + (i % 2) * 420, 40 + (i // 2) * 160), t, fill="black", font=F40)
print("   原始布局：两列，每列 4 行，行距 160px")
for ky in (1, 25, 41, 61, 91):
    print(f"   ky={ky:3d} -> {[t for t, _ in ocr(dilate(im, ky=ky))]}")

print()
print("=" * 74)
print("6. 在膨胀后的图上，逐词文本与 bbox（这是 ocr.py 真正要消费的东西）")
print("=" * 74)
im = dilate(base, ky=25)
for t, ws in ocr(im):
    print(f"   line={t!r}")
    for txt, r in ws:
        print(f"      word={txt!r:10s} x={r[0]:4d} y={r[1]:4d} w={r[2]:4d} h={r[3]:4d}")
