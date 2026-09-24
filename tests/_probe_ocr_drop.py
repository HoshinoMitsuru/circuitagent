# -*- coding: utf-8 -*-
"""探针：验证"短 token 被整条丢弃"是不是版面分析把它当表格竖线滤掉了。

如果是，那么"给这行加一点横向笔画/文字"应该能把它救回来 ——
这是一个便宜的通用缓解手段，值得测清楚再决定要不要写进 ocr.py。
"""
from PIL import Image, ImageDraw, ImageFont

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import Buffer

FONT = "C:/Windows/Fonts/arial.ttf"


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
    return [l.text for l in res.lines]


def canvas(w=520, h=180):
    im = Image.new("RGB", (w, h), "white")
    return im, ImageDraw.Draw(im)


f40 = ImageFont.truetype(FONT, 40)

print("=" * 74)
print("A. 基线：孤零零的 R1")
print("=" * 74)
im, d = canvas()
d.text((200, 60), "R1", fill="black", font=f40)
print("  只有 R1            ->", ocr(im))

print()
print("=" * 74)
print("B. 加一根横线（模拟导线）：能否把它救回来")
print("=" * 74)
for y in (100, 120):
    im, d = canvas()
    d.text((200, 60), "R1", fill="black", font=f40)
    d.line([20, y, 500, y], fill="black", width=3)
    print(f"  R1 + 横线(y={y})     ->", ocr(im))
im, d = canvas()
d.text((200, 60), "R1", fill="black", font=f40)
d.rectangle([150, 30, 320, 130], outline="black", width=3)
print("  R1 + 矩形框        ->", ocr(im))

print()
print("=" * 74)
print("C. 加同行的其它文字")
print("=" * 74)
for extra, label in [("R1 10k", "R1 10k（同行）"), ("R1 (10k)", "R1 (10k)"),
                     ("R1\n10k", "R1 换行 10k"), ("R = R1", "R = R1"),
                     ("R1:", "R1: 带冒号"), ("R1.", "R1. 带句点"),
                     ("R1_", "R1_ 带下划线"), ("[R1]", "[R1] 带方括号")]:
    im, d = canvas(600, 260)
    d.text((60, 40), extra, fill="black", font=f40)
    print(f"  {label:16s} ->", ocr(im))

print()
print("=" * 74)
print("D. 加粗/加描边是否能救（让笔画更粗，不再像细表格线）")
print("=" * 74)
for sw, label in [(0, "常规"), (1, "stroke_width=1"), (2, "stroke_width=2")]:
    im, d = canvas()
    d.text((200, 60), "R1", fill="black", font=f40, stroke_width=sw,
           stroke_fill="black")
    print(f"  R1 {label:16s} ->", ocr(im))

print()
print("=" * 74)
print("E. 更大字号 + 更长画布")
print("=" * 74)
for sz in (48, 60, 80, 100):
    ff = ImageFont.truetype(FONT, sz)
    im, d = canvas(600, 260)
    d.text((200, 80), "R1", fill="black", font=ff)
    print(f"  R1 @{sz:3d}px         ->", ocr(im))

print()
print("=" * 74)
print("F. 缩放整图（把画布放大，字相对变小）")
print("=" * 74)
im, d = canvas(520, 180)
d.text((200, 60), "R1", fill="black", font=f40)
for sc in (0.5, 1.0, 1.5, 2.0):
    big = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    print(f"  scale={sc}  ({big.width}x{big.height}) ->", ocr(big))

print()
print("=" * 74)
print("G. 其它短位号：确认哪些真的救不回来")
print("=" * 74)
im, d = canvas(1400, 200)
xs = 40
for t in ["R1", "C1", "V1", "I1", "L1", "R2"]:
    d.text((xs, 70), t, fill="black", font=f40)
    xs += 220
print("  6 个短位号平铺一行 ->", ocr(im))
for t in ["R1", "C1", "V1", "I1", "L1", "R2"]:
    im, d = canvas(600, 260)
    d.text((60, 60), t + " ", fill="black", font=f40)
    r = ocr(im)
    print(f"  {t:4s} 单独（尾部带空格）->", r)

print()
print("=" * 74)
print("H. 关键验证：真实场景（位号紧贴符号）能否正常")
print("=" * 74)
im, d = canvas(760, 300)
# 画一个电阻框 + 旁边的位号，模仿真实原理图
d.rectangle([260, 100, 420, 160], outline="black", width=3)
d.line([120, 130, 260, 130], fill="black", width=3)
d.line([420, 130, 560, 130], fill="black", width=3)
d.text((295, 40), "R1", fill="black", font=f40)
d.text((280, 190), "10k", fill="black", font=f40)
print("  电阻框 + R1 + 10k ->", ocr(im))
