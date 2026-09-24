# -*- coding: utf-8 -*-
"""探针 v2：修正上一版"只取该行第一个词"的测量错误。

上一版把 OCR 拆出的多个词只取了第一个，于是 "4.7k" 被记成 "4"，
看起来像"在小数点处截断"——那是我探针的错，不是 OCR 的规律。
这一版按 OCR 自己的行分组（line.text）来比对。
"""
import sys
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import Buffer


def ocr(im, lang="zh-Hans-CN"):
    rgba = im.convert("RGBA")
    px = rgba.tobytes()
    buf = Buffer(len(px))
    buf.length = len(px)
    memoryview(buf)[:] = px
    sb = SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.RGBA8,
                                                im.size[0], im.size[1])
    eng = OcrEngine.try_create_from_language(Language(lang))
    res = eng.recognize_async(sb).get()
    return [(line.text, [(w.text, (w.bounding_rect.x, w.bounding_rect.y,
                                  w.bounding_rect.width, w.bounding_rect.height))
                         for w in line.words])
            for line in res.lines]


FONT = "C:/Windows/Fonts/arial.ttf"

# 逐条独立渲染：每条一张图。彻底排除"行分组/相邻干扰"这类测量污染。
ITEMS = ["R1", "R2", "R3", "R4", "R5", "R10", "R11", "R12", "R20", "R21",
         "C1", "C2", "C3", "C10", "V1", "V2", "V3", "I1", "I2", "L1", "L2", "L10",
         "10k", "4.7k", "1k", "100uF", "12V", "2.2mH", "220", "47", "0.5A",
         "4k7", "1M", "1M5", "0.1uF", "-5V", "3.3V", "1.5k", "2N3904", "0.01uF"]


def one(text, size):
    f = ImageFont.truetype(FONT, size)
    box = f.getbbox(text)
    W = box[2] - box[0] + 120
    H = box[3] - box[1] + 120
    im = Image.new("RGB", (max(W, 200), max(H, 120)), "white")
    ImageDraw.Draw(im).text((60 - box[0], 60 - box[1]), text, fill="black", font=f)
    return im


print("=" * 78)
print("实验 A：每条单独成图，整行文本比对")
print("=" * 78)
print(f"  {'期望':>8s} " + "".join(f"{('@'+str(s)):>12s}" for s in (24, 32, 40, 48, 60)))
RES = {}
for size in (24, 32, 40, 48, 60):
    RES[size] = {}
    for t in ITEMS:
        lines = ocr(one(t, size))
        joined = " ".join(l for l, _ in lines).strip()
        RES[size][t] = joined

for t in ITEMS:
    row = "".join(f"{RES[s][t] or '(空)':>12s}" for s in (24, 32, 40, 48, 60))
    allok = all(RES[s][t] == t for s in (24, 32, 40, 48, 60))
    print(f"  {t:>8s} {row}   {'一致' if allok else '★'}")

print()
print("=" * 78)
print("实验 B：逐字符差异（只统计非空的）")
print("=" * 78)
char_pairs = Counter()
len_issues = Counter()
for s, m in RES.items():
    for want, got in m.items():
        if got == want or not got:
            continue
        if len(want) == len(got):
            for a, b in zip(want, got):
                if a != b:
                    char_pairs[(a, b)] += 1
        else:
            len_issues[(want, got)] += 1
print("  等长替换:", dict(char_pairs) or "（无）")
print()
print("  长度变化（含拆词、吞字）:")
for (w, g), n in len_issues.most_common():
    print(f"    {w!r} -> {g!r}   x{n}")

print()
print("=" * 78)
print("实验 C：忽略空白后是否一致（拆词问题）")
print("=" * 78)
n_space_only = 0
for s, m in RES.items():
    for want, got in m.items():
        if got and got != want and got.replace(" ", "") == want:
            n_space_only += 1
print(f"  仅因多出空格导致不一致的: {n_space_only} / {len(RES) * len(ITEMS)}")

print()
print("=" * 78)
print("实验 D：哪些条目在**所有**字号下都读对（这才是可依赖的集合）")
print("=" * 78)
always_ok = [t for t in ITEMS if all(RES[s][t] == t for s in RES)]
sometimes = [t for t in ITEMS if any(RES[s][t] == t for s in RES) and t not in always_ok]
never = [t for t in ITEMS if not any(RES[s][t] == t for s in RES)]
print("  全字号正确:", always_ok)
print()
print("  部分字号正确:", sometimes)
print()
print("  全字号错误:", never)
if never:
    print()
    for t in never:
        print(f"    {t:>8s} ->", {s: RES[s][t] for s in RES})

print()
print("=" * 78)
print("实验 E：放大 / 锐化能否救回全错的条目")
print("=" * 78)
for t in (never or ITEMS[:3]):
    base = one(t, 40)
    print(f"  === {t} ===")
    print(f"    @40 原图        -> {[l for l, _ in ocr(base)]}")
    for sc in (2, 3):
        big = base.resize((base.width * sc, base.height * sc), Image.LANCZOS)
        print(f"    x{sc} LANCZOS     -> {[l for l, _ in ocr(big)]}")
    big = base.resize((base.width * 2, base.height * 2), Image.NEAREST)
    print(f"    x2 NEAREST     -> {[l for l, _ in ocr(big)]}")
    # 反白
    inv = Image.new("RGB", base.size, "black")
    inv.paste(base.point(lambda v: 255 - v))
    print(f"    反白(黑底白字)  -> {[l for l, _ in ocr(inv)]}")

print()
print("=" * 78)
print("实验 F：换成 user_profile 引擎（不指定语言）是否不同")
print("=" * 78)
rgba = one("R1 10k", 40).convert("RGBA")
px = rgba.tobytes()
b = Buffer(len(px))
b.length = len(px)
memoryview(b)[:] = px
sb = SoftwareBitmap.create_copy_from_buffer(b, BitmapPixelFormat.RGBA8,
                                            rgba.size[0], rgba.size[1])
for label, eng in (("zh-Hans-CN", OcrEngine.try_create_from_language(Language("zh-Hans-CN"))),
                   ("user_profile", OcrEngine.try_create_from_user_profile_languages())):
    r = eng.recognize_async(sb).get()
    print(f"  {label:>13s} -> {[l.text for l in r.lines]}  (engine lang={eng.recognizer_language.language_tag})")
