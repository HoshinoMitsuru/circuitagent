# -*- coding: utf-8 -*-
"""探针：Windows 内置 OCR 读"元件标注"时到底受什么支配。

写这个探针是因为三个反直觉的实测结果，它们直接决定了 ``app/vision`` 里
几处阈值与规则：

1. **短文本行会被整条丢掉**，而且**无解**。``"R1"`` 在 18~56px 每一档
   都读不出来（"0 个词"），而 ``"R1 10k"`` 多数档能读出来。
   → 位号不要单独占一行（合成图 / 回绘 SVG / 操作指引都要遵守这条）。
2. **字号的影响不是单调的**。``"R1 10k"`` 在 26px 一个字都没有，
   在 22/30/34px 会被切成三段，在 18/20/24/40px 才是干净的两段。
   → 任何"调大字号就更稳"的直觉都是错的；测试图必须挑**实测过**的字号。
3. **字距一大，一个 token 会被切成好几个词**：``R12`` → ``['R','1','2']``、
   ``10k`` → ``['1','0','k']``。
   而**缝宽不能当判据** —— token 内字符缝 0.29~1.24 字高、
   正常词间空格 0.38~0.80 字高，两者**重叠**。
   → 判"是不是被拆散"只能靠**形状**（字母段 + 单字符数字段），
     见 ``pipeline._claim_refdes`` 与常量区的注释。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\_probe_ocr_lanelen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont                # noqa: E402

from app.vision import ocr as OCR                          # noqa: E402

SIZES = (18, 20, 22, 24, 26, 30, 34, 40, 56)


def _font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:                          # noqa: BLE001
            continue
    return ImageFont.load_default()


def _plain(text: str, size: int):
    im = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(im).text((40, 50), text, fill="black", font=_font(size))
    return im


def _spaced(text: str, size: int, gap: int):
    """逐字排版，人为拉开**字符间距**（模拟字距大的排版/放大后的图）。"""
    im = Image.new("RGB", (900, 160), "white")
    d = ImageDraw.Draw(im)
    f = _font(size)
    x = 40
    for ch in text:
        d.text((x, 50), ch, fill="black", font=f)
        x += int(d.textlength(ch, font=f)) + gap
    return im


def _gaps(words) -> list[int]:
    ws = sorted(words, key=lambda w: w.x)
    return [int(b.x - (a.x + a.w)) for a, b in zip(ws, ws[1:])]


def main() -> int:
    print("=" * 78)
    print("① 行长的影响：同一条标注、同一个字号，词数差多少")
    print("=" * 78)
    for text in ("R1", "R1 10k", "R1 10k R2 20k", "R1=10k"):
        row = []
        for size in SIZES:
            words = [w.text for w in OCR.recognize(_plain(text, size)).words]
            row.append(f"{size}:{'/'.join(words) if words else '（全丢）'}")
        print(f"  {text!r:16s} " + "  ".join(row))
    print("  ※ 'R1' 整行全丢 —— 短文本行会被 Windows OCR 跳过，无解；")
    print("    'R1 10k R2 20k' 每一档都读得干干净净 —— **行越长越稳**。")

    print()
    print("=" * 78)
    print("② 字距的影响：一个 token 被拆成几个词")
    print("=" * 78)
    for text in ("R12", "10k"):
        for size in (24, 36):
            for gap in (4, 12, 20):
                r = OCR.recognize(_spaced(text, size, gap))
                ws = sorted(r.words, key=lambda w: w.x)
                if len(ws) < 2:
                    print(f"  {text!r:6s} {size}px 字距{gap:2d} → "
                          f"{[w.text for w in ws]}（没被拆）")
                    continue
                g = _gaps(ws)
                h = ws[0].h
                print(f"  {text!r:6s} {size}px 字距{gap:2d} → "
                      f"{[w.text for w in ws]}  缝={g} 缝/字高="
                      + str([round(x / h, 2) for x in g]))

    print()
    print("=" * 78)
    print("③ 缝宽能不能当判据：token 内缝 vs 正常词距")
    print("=" * 78)
    inner: list[float] = []
    for text in ("R12", "10k"):
        for size in (24, 28, 36):
            for gap in (4, 6, 12, 20):
                ws = sorted(OCR.recognize(_spaced(text, size, gap)).words,
                            key=lambda w: w.x)
                if len(ws) > 1:
                    h = ws[0].h
                    inner += [x / h for x in _gaps(ws)]
    between: list[float] = []
    for text in ("R1 10k", "R1 10k R2 20k", "R1 10k C1 0.1uF"):
        for size in (20, 22, 26, 34):
            ws = sorted(OCR.recognize(_plain(text, size)).words,
                        key=lambda w: w.x)
            if len(ws) > 1:
                h = ws[0].h
                between += [x / h for x in _gaps(ws)]
    print(f"  token 内字符缝（一个位号/数值被拆开）："
          f"{min(inner):.2f} ~ {max(inner):.2f} 字高，n={len(inner)}")
    print(f"  正常词间空格：                        "
          f"{min(between):.2f} ~ {max(between):.2f} 字高，n={len(between)}")
    print("  ★ 两段区间**重叠** —— 缝宽单独作判据必然误伤。"
          "判「被拆散」只能靠形状（字母段 + 单字符数字段）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
