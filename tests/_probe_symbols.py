# -*- coding: utf-8 -*-
"""探针：验证符号定位（洞法 + 模板）在合成图上的行为。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from app.vision import symbols as SY

FAILS = []


def ck(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   -> " + str(extra)) if extra else ""))
    if not cond:
        FAILS.append(name)


print("=" * 76)
print("1. 洞法：矩形框 / 圆 / 圆内带直径线 / 圆内带箭头 / 电容 / 电感")
print("=" * 76)
im = Image.new("L", (1400, 320), 255)
d = ImageDraw.Draw(im)
# 矩形框电阻（国标，30x20 比例的放大版）
d.rectangle([60, 80, 220, 180], outline=0, width=4)
# 空圆（本项目的电压源画法）
d.ellipse([320, 80, 440, 200], outline=0, width=4)
# 圆内带竖直直径线（教材式电压源）
d.ellipse([520, 80, 640, 200], outline=0, width=4)
d.line([580, 82, 580, 198], fill=0, width=3)
# 圆内带水平横线（同上的横切版本，验证朝向无关）
d.ellipse([720, 80, 840, 200], outline=0, width=4)
d.line([522 + 200, 140, 638 + 200, 140], fill=0, width=3)
# 圆内带箭头（本项目电流源）
d.ellipse([920, 80, 1040, 200], outline=0, width=4)
d.line([948, 140, 1004, 140], fill=0, width=3)
d.line([984, 120, 1004, 140], fill=0, width=3)
d.line([1004, 140, 984, 160], fill=0, width=3)
# 电容（两极板）
d.line([1120, 140, 1160, 140], fill=0, width=4)
d.line([1160, 90, 1160, 190], fill=0, width=5)
d.line([1180, 90, 1180, 190], fill=0, width=5)
d.line([1180, 140, 1220, 140], fill=0, width=4)
# 电感（四段弧）
lx, ly = 1280, 140
d.line([1260, ly, 1280, ly], fill=0, width=4)
for i in range(4):
    d.arc([lx + i * 20, ly - 10, lx + i * 20 + 20, ly + 10], 180, 360, fill=0, width=3)
d.line([1360, ly, 1380, ly], fill=0, width=4)

mask = SY.ink_mask(im)
holes = SY.find_holes(mask)
print(f"  检出洞 {len(holes)} 个:")
for h in holes:
    print(f"    ({h.x:4d},{h.y:3d}) {h.w:3d}x{h.h:3d} area={h.area:5d} "
          f"fill={h.fill:.2f} aspect={h.aspect:.2f}")
groups = SY.group_holes(holes)
print(f"  归成 {len(groups)} 组:", [[f"{h.w}x{h.h}" for h in g] for g in groups])

print()
print("=" * 76)
print("2. 逐块模板匹配（看每个图形被判成什么）")
print("=" * 76)
regions = {
    "矩形框": (56, 76, 226, 186),
    "空圆(V的画法)": (316, 76, 446, 206),
    "圆+竖直径线": (516, 76, 646, 206),
    "圆+横直径线": (716, 76, 846, 206),
    "圆+箭头(I)": (916, 76, 1046, 206),
    "电容C": (1116, 86, 1226, 196),
    "电感L": (1256, 126, 1390, 156),
}
for name, (x0, y0, x1, y1) in regions.items():
    patch = mask[y0:y1, x0:x1]
    sc = SY.match_kind(patch)
    best = max(sc.items(), key=lambda t: t[1])
    print(f"  {name:16s} -> 判为 {best[0]} ({best[1]:.2f})   " +
          " ".join(f"{k}={v:.2f}" for k, v in sorted(sc.items(), key=lambda t: -t[1])))

print()
print("=" * 76)
print("3. 端到端 detect_symbols")
print("=" * 76)
rep = SY.detect_symbols(im, correct=False)
print(f"  位 {len(rep.slots)} 个，洞 {len(rep.holes)} 个，未解释墨迹 {len(rep.unexplained)} 块")
for s in rep.slots:
    print(f"    {s.kind:2s} ({s.x:4d},{s.y:3d}) {s.w:3d}x{s.h:3d} "
          f"conf={s.confidence:.2f} {'★需人工' if s.needs_human else '  '} "
          f"洞法={s.hole_kind} 模板={s.template_kind} 一致={s.agreed}")
for w in rep.warnings:
    print("    WARN:", w)

print()
print("=" * 76)
print("4. 判据：矩形 -> R；两洞 -> V")
print("=" * 76)
rects = [s for s in rep.slots if 56 <= s.x <= 230]
ck("矩形框被判为 R", rects and rects[0].kind == "R", rects[0].kind if rects else None)
ck("R 置信度达门槛", rects and rects[0].confidence >= SY.CONFIDENCE_GATE,
   rects[0].confidence if rects else None)
two = [s for s in rep.slots if 510 <= s.x <= 650]
ck("圆+竖直径线 -> V", two and two[0].kind == "V", two[0].kind if two else None)
ck("V 置信度达门槛", two and two[0].confidence >= SY.CONFIDENCE_GATE,
   two[0].confidence if two else None)

print()
print("=" * 76)
print("5. 否定结果：空圆与圆+箭头洞法分不开，必须落到需人工或模板低分")
print("=" * 76)
for lo, hi, label in ((316, 450, "空圆(V画法)"), (916, 1050, "圆+箭头(I)")):
    got = [s for s in rep.slots if lo <= s.x <= hi]
    if got:
        s = got[0]
        print(f"  {label}: kind={s.kind} conf={s.confidence:.2f} "
              f"needs_human={s.needs_human} agreed={s.agreed} 模板={s.template_kind}")
    else:
        print(f"  {label}: 未定位到")

print()
print("=" * 76)
print("6. 导线不应被当成符号")
print("=" * 76)
im2 = Image.new("L", (900, 200), 255)
d2 = ImageDraw.Draw(im2)
d2.line([40, 100, 860, 100], fill=0, width=3)
d2.line([450, 30, 450, 170], fill=0, width=3)
rep2 = SY.detect_symbols(im2, correct=False)
ck("纯导线图：0 个符号位", len(rep2.slots) == 0, len(rep2.slots))
ck("纯导线图：0 块未解释墨迹", len(rep2.unexplained) == 0, rep2.unexplained)
print("    warnings:", rep2.warnings)

print()
print("=" * 76)
print("7. 未解释墨迹：塞一个大黑方块，必须被报出来")
print("=" * 76)
im3 = Image.new("L", (600, 300), 255)
d3 = ImageDraw.Draw(im3)
d3.rectangle([100, 100, 220, 200], outline=0, width=4)   # 正常电阻
d3.ellipse([380, 90, 500, 210], fill=0)                  # 实心黑圆（异物）
rep3 = SY.detect_symbols(im3, correct=False)
print(f"  位 {len(rep3.slots)} 个，未解释 {len(rep3.unexplained)} 块: {rep3.unexplained}")
ck("实心异物被报成未解释墨迹", len(rep3.unexplained) >= 1, rep3.unexplained)

print()
print("=" * 76)
print("8. slot_region 必须只比本体大一点点（放大了会把导线也剔掉）")
print("=" * 76)
if rep.slots:
    s = rep.slots[0]
    x0, y0, x1, y1 = s.slot_region(margin=3)
    print(f"  本体 {s.x},{s.y},{s.w}x{s.h}  ->  slot_region {x0:.0f},{y0:.0f}..{x1:.0f},{y1:.0f}")
    ck("margin 3 时区域只外扩 3px",
       (x0, y0, x1, y1) == (s.x - 3, s.y - 3, s.x + s.w + 3, s.y + s.h + 3))

print()
print("=" * 76)
print("9. 方向判定")
print("=" * 76)
im4 = Image.new("L", (700, 300), 255)
d4 = ImageDraw.Draw(im4)
d4.rectangle([60, 120, 240, 180], outline=0, width=4)    # 横放 (180x60)
d4.rectangle([400, 60, 460, 240], outline=0, width=4)    # 竖放 (60x180)
rep4 = SY.detect_symbols(im4, correct=False)
for s in rep4.slots:
    print(f"    {s.kind} {s.w}x{s.h} orientation={s.orientation}")
ck("横放判 h", any(s.orientation == "h" for s in rep4.slots))
ck("竖放判 v", any(s.orientation == "v" for s in rep4.slots))

print()
print("=" * 76)
print("FAILS =", len(FAILS))
for f in FAILS:
    print("  -", f)
