"""SVG 通道实跑自检 —— 重点是验收技能文档里那张判定表。

四组用例：
  A. 纯交叉无圆点  -> **不相连**（跨线）
  B. 交叉处有圆点  -> **相连**
  C. T 接无圆点    -> **相连**（教科书惯例）
  D. 回绘往返      -> 本工具画出的 SVG 重新导入，IR 应当逐项还原

★ 断言的写法很重要：**不要比节点总数**。一根导线本身就是导体，
它的两个端点天然属于同一个节点，所以"节点数"这个指标很容易数错 ——
我第一次就把它期望成 4 而实际是 2（其实是我不对，代码是对的）。
正确姿势是问："坐标 P 上的点和坐标 Q 上的点，是不是同一个节点？"

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_ingest_svg.py
"""

from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingest.svg_in import parse_svg_text, svg_to_ir   # noqa: E402
from app.ingest.topology import group_nodes, node_at      # noqa: E402
from app.ir.model import Circuit, Component, Evidence     # noqa: E402
from app.ir.render import render_svg                      # noqa: E402
from app.solver.reconcile import run_all                  # noqa: E402


def wrap(inner: str, vb: str = "0 0 200 200") -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}">{inner}</svg>'


H_LINE = '<line x1="20" y1="60" x2="180" y2="60"/>'
V_LINE = '<line x1="100" y1="10" x2="100" y2="110"/>'
T_LINE = '<line x1="100" y1="60" x2="100" y2="110"/>'
DOT = '<circle class="junction" cx="100" cy="60" r="5" fill="#000000"/>'

CASES = [
    ("A 纯交叉无圆点", wrap(H_LINE + V_LINE),
     [((20, 60), (180, 60), True, "同一根横导线，两端必同节点"),
      ((20, 60), (100, 10), False, "跨界交叉，应**不相连**"),
      ((100, 10), (100, 110), True, "同一根竖导线，两端必同节点")]),
    ("B 交叉处有圆点", wrap(H_LINE + V_LINE + DOT),
     [((20, 60), (100, 10), True, "有圆点 -> 判为**相连**"),
      ((180, 60), (100, 110), True, "有圆点，四端全并")]),
    ("C T接无圆点", wrap(H_LINE + T_LINE),
     [((20, 60), (180, 60), True, "同一根横导线"),
      ((20, 60), (100, 110), True, "T 接无圆点 -> 教科书惯例判**相连**")]),
]


def test_rules() -> int:
    print("=" * 74)
    print("第一组：跨线 / 圆点 / T 接 三条判定规则")
    print("=" * 74)
    fails = 0
    for name, svg, checks in CASES:
        geo = parse_svg_text(svg)
        g = group_nodes(geo)
        print(f"\n  {name}")
        print(f"    线段 {len(g['segments'])} 条，圆点 {g['n_dots']} 个，"
              f"电学节点 {len(g['roots'])} 个，跨线记录 {len(g['crossings'])} 处")
        for (p, q, want_same, why) in checks:
            na, nb = node_at(g, *p), node_at(g, *q)
            same = (na == nb)
            ok = (same == want_same)
            if not ok:
                fails += 1
            print(f"    [{'通过' if ok else '失败'}] {p} -> {na}  vs  {q} -> {nb}"
                  f"  同节点={same}（期望 {want_same}）")
            print(f"          {why}")
    return fails


# ---------------------------------------------------------------- 回绘往返


def sample_circuit() -> Circuit:
    """与求解器测试同款电桥，便于交叉印证。

    ★ 元件带 ``geom`` 视觉坐标，取值就是下面 ``BRIDGE_POS`` 那套 ——
    相当于"这张图是从照片/SVG 解析出来的"。往返测试必须走这条路径：
    真实电路永远有图上坐标，``layout_geom`` 直接复用它，
    而自动网格布局只是给报告插图用的示意排版。
    """
    def C(ref, kind, a, b, v):
        return Component(ref=ref, kind=kind, nodes=(a, b), value=v,
                         evidence=Evidence(source="manual", confidence=1.0),
                         geom={"p1": list(BRIDGE_POS[a]), "p2": list(BRIDGE_POS[b])})
    return Circuit(name="往返测试电桥", components=[
        C("V1", "V", "1", "0", 12.0),
        C("R1", "R", "1", "2", 100.0),
        C("R2", "R", "2", "0", 200.0),
        C("R3", "R", "1", "3", 150.0),
        C("R4", "R", "3", "0", 300.0),
        C("R5", "R", "2", "3", 50.0),
    ])


#: 电桥的图上坐标。**每一对结点之间都没有第三个结点落在连线上** ——
#: 这是往返测试成立的前提（有的话结点会被几何误并，那是布局的问题，不是解析的问题）。
BRIDGE_POS = {"0": (100.0, 400.0), "1": (100.0, 100.0),
              "2": (400.0, 250.0), "3": (700.0, 250.0)}


def test_roundtrip() -> int:
    print("\n" + "=" * 74)
    print("第二组：回绘往返（IR -> SVG -> IR）")
    print("=" * 74)
    fails = 0
    circ = sample_circuit()

    # ---- 2a. geom 模式 + 语义标记：应当无损还原
    svg = render_svg(circ, mode="geom")
    back, report = svg_to_ir(svg, name="回导")
    print(f"\n  [2a] geom 模式 + 语义标记（data-ca-*）")
    print(f"       元件识别来源：{report['component_source']}")
    print(f"       几何量：{report['geometry']}")
    by_ref = {c.ref: c for c in circ.components}
    for c in back.components:
        orig = by_ref.get(c.ref)
        if orig is None:
            print(f"       [失败] 原图里没有 {c.ref}")
            fails += 1
            continue
        # 节点名往返后可能被重排（参考节点恒为 0），所以比"连接关系"而不是比字面
        same_kind = (c.kind == orig.kind)
        same_val = (c.value == orig.value)
        # 两端节点：用"等价类"比 —— 两端在两次解析中是否落在同一对节点上
        ok = same_kind and same_val
        if not ok:
            fails += 1
        print(f"       [{'通过' if ok else '失败'}] {c.ref}: kind {c.kind}"
              f"（原 {orig.kind}） value {c.value}（原 {orig.value}）"
              f" nodes {c.nodes}（原 {orig.nodes}）")
        if not (same_kind and same_val):
            continue
        if tuple(c.nodes) != tuple(orig.nodes):
            print(f"             注意：节点名从 {orig.nodes} 变为 {c.nodes}"
                  "（参考节点重排或名字冲突时的正常改名，需人工核对）")
            fails += 1
    n_same = len(back.components) == len(circ.components)
    print(f"       [{'通过' if n_same else '失败'}] 元件数量 {len(back.components)}"
          f"（原 {len(circ.components)}）")
    if not n_same:
        fails += 1

    # 关键：往返后的电路必须一样能解，且解一样
    p1, p2 = run_all(circ), run_all(back)
    v1 = p1["solutions"]["mna"]["node_voltages"]
    v2 = p2["solutions"]["mna"]["node_voltages"]
    print(f"\n       原图解 V = {v1}")
    print(f"       回导解 V = {v2}")
    ok_sol = (v1 == v2)
    print(f"       [{'通过' if ok_sol else '失败'}] 往返后节点电压逐一相等")
    if not ok_sol:
        fails += 1

    # ---- 2b. 抹掉语义标记：走纯几何推断
    stripped = re.sub(r'\s*data-ca-[a-z0-9]+="[^"]*"', "", svg)
    n_hints = len(re.findall(r"data-ca-", stripped))
    print(f"\n  [2b] 抹掉语义标记后走纯几何推断（残留 data-ca- 计数 {n_hints}，应为 0）")
    try:
        back2, report2 = svg_to_ir(stripped, name="几何回导")
        print(f"       元件识别来源：{report2['component_source']}")
        print(f"       识别出 {len(back2.components)} 个元件（原图 6 个）")
        for c in back2.components:
            flag = "需人工确认" if c.evidence.needs_human else "可采信"
            print(f"         {c.ref}: {c.kind} {c.nodes} value={c.value} "
                  f"conf={c.evidence.confidence:.2f} [{flag}]")
        n_ok = len(back2.components) == len(circ.components)
        print(f"       [{'通过' if n_ok else '失败'}] 元件数量与原图一致")
        if not n_ok:
            fails += 1
        # 几何推断的数值来自 SVG 里的真文字（矢量图有文字，位图没有）
        got_values = sorted(c.value for c in back2.components if c.value is not None)
        want_values = sorted(c.value for c in circ.components)
        ok_v = got_values == want_values
        print(f"       [{'通过' if ok_v else '失败'}] 数值读出：{got_values}"
              f"（原 {want_values}）")
        if not ok_v:
            fails += 1
        # 几何推断的"圆形符号"分不清电压源还是电流源，必须落进人工确认闸门
        v_like = [c for c in back2.components if c.kind in ("V", "I")]
        ok_gate = all(c.evidence.needs_human for c in v_like) if v_like else True
        print(f"       [{'通过' if ok_gate else '失败'}] 电压源/电流源类符号全部落入"
              f"人工确认闸门（{len(v_like)} 个，置信度 "
              f"{[round(c.evidence.confidence,2) for c in v_like]}）")
        if not ok_gate:
            fails += 1
    except Exception as e:
        print(f"       [失败] 几何推断失败：{type(e).__name__}: {e}")
        fails += 1

    # ---- 2c. 自动网格布局：**只作示意**，其重叠必须被自检抓出来
    #    这一条不是"顺手加的功能"，而是踩过的坑的防复现：
    #    grid 把电桥的 1/2/3 放在同一列，R3 的长导线会穿过 R1 的符号，
    #    回导时两个结点被几何**误并成一个**，而且不报错。
    print("\n  [2c] 自动网格布局的自检（应当报出重叠，并明示不可回导）")
    dg: dict = {}
    svg_grid = render_svg(circ, mode="grid", diagnostics=dg)
    n_groups = len(re.findall(r'class="glyph ca-component"', svg_grid))
    print(f"       布局={dg['layout']}  画出元件组 {n_groups} 个（应为 6）")
    print(f"       回导安全性 roundtrip_safe={dg['roundtrip_safe']}"
          f"  重叠 {len(dg['overlaps'])} 处")
    for o in dg["overlaps"][:4]:
        print(f"         · {o['why']}")
    ok_g1 = (n_groups == len(circ.components))
    ok_g2 = (dg["roundtrip_safe"] is False and len(dg["overlaps"]) > 0)
    print(f"       [{'通过' if ok_g1 else '失败'}] 网格布局画出了全部元件符号")
    print(f"       [{'通过' if ok_g2 else '失败'}] 布局重叠被自检抓出并标记为不可回导")
    if not ok_g1:
        fails += 1
    if not ok_g2:
        fails += 1

    # ---- 2d. 自动选版式：电桥这类"同层三结点两两相连"的图，分层会重叠，
    #    生成-检验应当回退到**环形**，而且环形版面的自检要干净 ——
    #    版面无重叠是"报告插图能看"的底线，`roundtrip_safe=True` 则意味着
    #    插图和原电路是同一份几何，可以互相印证。
    print("\n  [2d] 自动选版式 + 环形布局自检（密集图应当无重叠、可回导）")
    bare = Circuit(name=circ.name, components=[
        Component(ref=c.ref, kind=c.kind, nodes=c.nodes, value=c.value,
                  evidence=c.evidence) for c in circ.components])
    dg2: dict = {}
    svg_auto = render_svg(bare, mode="auto", diagnostics=dg2)
    print(f"       选中的布局：{dg2['layout']}（分层试过有重叠，回退到环形）")
    print(f"       重叠 {len(dg2['overlaps'])} 处，roundtrip_safe={dg2['roundtrip_safe']}")
    for o in dg2["overlaps"][:4]:
        print(f"         · {o['why']}")
    ok_d1 = (dg2["layout"] == "radial" and dg2["roundtrip_safe"] is True)
    print(f"       [{'通过' if ok_d1 else '失败'}] 密集图选了环形且版面无重叠")
    if not ok_d1:
        fails += 1
    # 环形图也必须能原样导回来（几何干净 = 可回导，这正是它比分层强的地方）
    try:
        back3, _ = svg_to_ir(svg_auto, name="环形回导")
        ok_d2 = (len(back3.components) == len(circ.components))
        print(f"       [{'通过' if ok_d2 else '失败'}] 环形布局回导元件数 "
              f"{len(back3.components)}（原 {len(circ.components)}）")
        if not ok_d2:
            fails += 1
    except Exception as e:
        print(f"       [失败] 环形布局回导失败：{type(e).__name__}: {e}")
        fails += 1

    # ---- 2e. 简单回路应当仍走分层。★ 这里刻意用**单回路**而不是"串联链"：
    #    真实电路的串联必然要回流，所以最小串联电路就是一个单回路（3 结点 3 支路）。
    #    而"单回路"的边数正好等于结点数 —— 我早先想用"边数≥结点数→环形"当判据，
    #    被它打脸了一次：单回路分层画是漂亮的直线，环形反而摊成一个圆。
    print("\n  [2e] 简单单回路应当仍走分层（疏密不是判据，自检结果才是）")
    chain = Circuit(name="单回路", components=[
        Component(ref="V1", kind="V", nodes=("1", "0"), value=10.0,
                  evidence=Evidence(source="manual", confidence=1.0)),
        Component(ref="R1", kind="R", nodes=("1", "2"), value=10.0,
                  evidence=Evidence(source="manual", confidence=1.0)),
        Component(ref="R2", kind="R", nodes=("2", "0"), value=10.0,
                  evidence=Evidence(source="manual", confidence=1.0)),
    ])
    dg3: dict = {}
    render_svg(chain, mode="auto", diagnostics=dg3)
    ok_e = (dg3["layout"] == "grid" and dg3["roundtrip_safe"] is True)
    print(f"       布局={dg3['layout']}  重叠 {len(dg3['overlaps'])} 处"
          f"  roundtrip_safe={dg3['roundtrip_safe']}")
    print(f"       [{'通过' if ok_e else '失败'}] 单回路保持分层且版面干净")
    if not ok_e:
        fails += 1
    return fails


def main() -> int:
    f = test_rules()
    f += test_roundtrip()
    print("\n" + "=" * 74)
    print(f"总计失败项：{f}")
    print("=" * 74)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
