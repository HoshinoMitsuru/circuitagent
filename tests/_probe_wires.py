# -*- coding: utf-8 -*-
"""探针：导线追踪。用**已知拓扑**的合成图验证，重点是两条铁律。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from app.vision import symbols as SY
from app.vision import wires as WR

FAILS = []


def ck(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   -> " + str(extra)) if extra else ""))
    if not cond:
        FAILS.append(name)


def series_loop():
    """串联回路：左边 V1（圆）、上边 R1（矩形框）、右边与下边是纯导线。

    正确答案：**两个**电气结点。
    - 上左角 A：R1 左端 + V1 上端
    - 上右→右下→下左 这一整条纯导线串：R1 右端 + V1 下端
    纯导线不产生元件，只是把结点并成同一个，所以 4 个几何角只构成 2 个结点。
    """
    im = Image.new("RGB", (620, 480), "white")
    d = ImageDraw.Draw(im)
    W = 4
    # 回路四角
    A, B, C, D = (120, 120), (480, 120), (480, 360), (120, 360)
    # R1：上边中段的矩形框
    d.rectangle([260, 105, 340, 135], outline="black", width=W)
    d.line([A[0], 120, 260, 120], fill="black", width=W)     # 上左
    d.line([340, 120, B[0], 120], fill="black", width=W)     # 上右
    d.line([480, 120, 480, 360], fill="black", width=W)      # 右
    d.line([480, 360, 120, 360], fill="black", width=W)      # 下
    # V1：左边中段的空圆
    d.ellipse([95, 215, 145, 265], outline="black", width=W)
    d.line([120, 120, 120, 215], fill="black", width=W)      # 左上
    d.line([120, 265, 120, 360], fill="black", width=W)      # 左下
    return im


print("=" * 78)
print("1. 串联回路：应为 2 个结点，两端子接法正确")
print("=" * 78)
im = series_loop()
sym = SY.detect_symbols(im, correct=False)
print(f"  符号 {len(sym.slots)} 个:")
for i, s in enumerate(sym.slots):
    print(f"    #{i} {s.kind} ({s.x},{s.y}) {s.w}x{s.h} conf={s.confidence:.2f} "
          f"朝向={s.orientation}")

g = WR.build_wire_graph(im, sym)
print()
print(f"  线段 {len(g.segments)} 条，结点 {len(g.junctions)} 个，圆点 {len(g.dots)} 个")
for j in g.junctions:
    print(f"    结点 {j.name}: ({j.x:.0f},{j.y:.0f}) kind={j.kind} 度={j.degree}")
    print(f"        依据: {j.evidence}")
print()
for si, terms in g.terminals.items():
    s = sym.slots[si]
    print(f"  元件 #{si} {s.kind} 端子: " +
          ", ".join(f"{t.side}->结点{t.node}({t.x:.0f},{t.y:.0f})" for t in terms))

ck("结点数为 2", len(g.junctions) == 2, len(g.junctions))
if len(g.junctions) == 2:
    r_idx = next((i for i, s in enumerate(sym.slots) if s.kind == "R"), None)
    v_idx = next((i for i, s in enumerate(sym.slots) if s.kind == "V"), None)
    ck("定位到 R 与 V", r_idx is not None and v_idx is not None)
    if r_idx is not None and v_idx is not None:
        rt = g.terminals[r_idx]
        vt = g.terminals[v_idx]
        ck("R 两端子都接上了结点", all(t.node is not None for t in rt),
           [t.node for t in rt])
        ck("V 两端子都接上了结点", all(t.node is not None for t in vt),
           [t.node for t in vt])
        if all(t.node is not None for t in rt + vt):
            ck("★ R 两端在不同结点（否则等于短路）", rt[0].node != rt[1].node,
               (rt[0].node, rt[1].node))
            ck("★ V 两端在不同结点", vt[0].node != vt[1].node,
               (vt[0].node, vt[1].node))
            # 同一个角上的两个端子应在同一结点
            corner = vt[0].node if abs(vt[0].y - 120) < abs(vt[1].y - 120) else vt[1].node
            r_corner = rt[0].node if abs(rt[0].x - 120) < abs(rt[1].x - 120) else rt[1].node
            ck("★ R 左端与 V 上端在同一结点（上左角）", corner == r_corner,
               (corner, r_corner))
        ck("V 的轴线按导线判为竖直（不是按外接矩形抛硬币）",
           abs(vt[0].x - vt[1].x) < 30, [(t.x, t.y) for t in vt])

print()
print("=" * 78)
print("2. 铁律：十字交叉且无圆点 → 不相连")
print("=" * 78)
im2 = Image.new("RGB", (500, 500), "white")
d2 = ImageDraw.Draw(im2)
d2.line([60, 250, 440, 250], fill="black", width=4)      # 水平线
d2.line([250, 60, 250, 440], fill="black", width=4)      # 垂直线，十字交叉，无圆点
sym2 = SY.detect_symbols(im2, correct=False)
g2 = WR.build_wire_graph(im2, sym2)
print(f"  线段 {len(g2.segments)} 条，结点 {len(g2.junctions)} 个，圆点 {len(g2.dots)} 个")
for j in g2.junctions:
    print(f"    结点 {j.name}: ({j.x:.0f},{j.y:.0f}) 度={j.degree}")
for w in g2.warnings:
    print("    WARN:", w[:120])
ck("十字交叉且无圆点 → 仍是 2 个独立结点（不相连）",
   len(g2.junctions) == 2, len(g2.junctions))
ck("没有检出圆点", len(g2.dots) == 0, g2.dots)

print()
print("=" * 78)
print("3. 铁律：同一交叉处**加上圆点** → 变成相连")
print("=" * 78)
d2.ellipse([244, 244, 256, 256], fill="black")           # 圆心加一个实心圆点
sym3 = SY.detect_symbols(im2, correct=False)
g3 = WR.build_wire_graph(im2, sym3)
print(f"  线段 {len(g3.segments)} 条，结点 {len(g3.junctions)} 个，圆点 {len(g3.dots)} 个")
for dd in g3.dots:
    print(f"    圆点 ({dd['x']:.0f},{dd['y']:.0f}) r={dd['radius']}")
for j in g3.junctions:
    print(f"    结点 {j.name}: ({j.x:.0f},{j.y:.0f}) 度={j.degree}")
ck("检出圆点", len(g3.dots) >= 1, g3.dots)
ck("有圆点时四段并成 1 个结点（相连）", len(g3.junctions) == 1, len(g3.junctions))

print()
print("=" * 78)
print("4. 铁律：T 形接入（无圆点）→ 相连")
print("=" * 78)
im4 = Image.new("RGB", (500, 400), "white")
d4 = ImageDraw.Draw(im4)
d4.line([60, 150, 440, 150], fill="black", width=4)      # 横线
d4.line([250, 150, 250, 330], fill="black", width=4)     # 竖线，端点落在横线中段
sym4 = SY.detect_symbols(im4, correct=False)
g4 = WR.build_wire_graph(im4, sym4)
print(f"  结点 {len(g4.junctions)} 个")
for j in g4.junctions:
    print(f"    结点 {j.name}: ({j.x:.0f},{j.y:.0f}) kind={j.kind} 度={j.degree}")
    print(f"        依据: {j.evidence}")
ck("T 形接入 → 并成 1 个结点（相连）", len(g4.junctions) == 1, len(g4.junctions))

print()
print("=" * 78)
print("5. 符号本体必须不参与连通（否则元件被自身短路）")
print("=" * 78)
im5 = Image.new("RGB", (400, 300), "white")
d5 = ImageDraw.Draw(im5)
d5.rectangle([140, 130, 260, 170], outline="black", width=4)
d5.line([40, 150, 140, 150], fill="black", width=4)
d5.line([260, 150, 360, 150], fill="black", width=4)
sym5 = SY.detect_symbols(im5, correct=False)
g5 = WR.build_wire_graph(im5, sym5)
r5 = next((i for i, s in enumerate(sym5.slots) if s.kind == "R"), None)
print(f"  结点 {len(g5.junctions)} 个")
if r5 is not None:
    t = g5.terminals[r5]
    print("  R 端子:", [(x.node, round(x.x), round(x.y)) for x in t])
    ck("电阻两端子在**不同**结点（本体没把它自己短接）",
       t[0].node != t[1].node or t[0].node is None, (t[0].node, t[1].node))
    ck("电阻两端子都接上了结点", all(x.node is not None for x in t),
       [x.node for x in t])

print()
print("=" * 78)
print("6. 圆点检测不该把粗线误判成圆点")
print("=" * 78)
im6 = Image.new("RGB", (500, 300), "white")
d6 = ImageDraw.Draw(im6)
d6.line([40, 150, 460, 150], fill="black", width=12)     # 很粗的线
sym6 = SY.detect_symbols(im6, correct=False)
g6 = WR.build_wire_graph(im6, sym6)
print(f"  粗线上的圆点检出数: {len(g6.dots)} -> {g6.dots}")
ck("粗直线不被误判成圆点", len(g6.dots) == 0, g6.dots)

print()
print("=" * 78)
print("7. probe()")
print("=" * 78)
print(" ", WR.probe())

print()
print("=" * 78)
print("FAILS =", len(FAILS))
for f in FAILS:
    print("  -", f)
