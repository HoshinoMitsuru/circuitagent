"""跨工具对账：把同一份网表同时喂给两套**独立实现**，逐项比数。

为什么需要它
------------
本项目自己那条"不许只算一遍"的规矩，约束的是**同一份 IR 上的三条代码路径**
（节点电压法 / 支路电流法 / ngspice）。但它有个共同的盲区：**三条路径共用同一份 IR**。
所以「网表抄错了」「参考方向定反了」「参考节点选错了」这类错误，
三法会**一致地**给出同一个错答案，互校完全拦不住 —— 技能文档里已经写明这一点。

真正能补上这个盲区的是：换一套**独立写的实现**，喂**同一份网表**，比数。
本脚本用的就是 SharedUmbrella 项目里那份独立写的三法求解器
（``D:\\Psyche\\SharedUmbrella\\.workbuddy\\tools\\circuit_dc_solve.py``）。

★ 两套实现的符号约定**故意不同**，对账时必须显式映射，否则会把约定差异误报成算错：

| 量 | circuit_agent | SharedUmbrella | 映射 |
|---|---|---|---|
| 电压源 +端 | ``nodes[0]`` | 网表第一个节点 | 同 |
| **电压源电流参考方向** | 内部 ``−→+``（``nodes[1]→nodes[0]``） | ``+→−``（``n1→n2``） | **取相反数** |
| 电阻/电流源参考方向 | ``nodes[0]→nodes[1]`` | ``n1→n2`` | 同 |
| 参考节点 | ``Circuit.ref_node`` | ``ref`` 行 | 同 |

映射后若两套实现**逐项严格相等**（都是精确有理数，不是"足够小"），
才说明这张网表的读图与建模真的可以采信。

用法：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe \\
        tools\\crosscheck_sharedumbrella.py [--spice] [网表.txt ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHARED_TOOLS = Path(r"D:\Psyche\SharedUmbrella\.workbuddy\tools")
SHARED_SOLVER = SHARED_TOOLS / "circuit_dc_solve.py"

from app.ir.model import Circuit, Component, Evidence          # noqa: E402
from app.solver.base import declared_direction                  # noqa: E402
from app.solver.branch import branch_current_method             # noqa: E402
from app.solver.mna import node_voltage_method                  # noqa: E402


def load_shared_solver():
    """把外部那份独立求解器当模块载进来（不复制、不改动，直接用原件）。"""
    if not SHARED_SOLVER.exists():
        raise SystemExit(f"找不到独立求解器：{SHARED_SOLVER}")
    spec = importlib.util.spec_from_file_location("shared_dc_solve", SHARED_SOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_agent_ir(branches, ref: str, name: str) -> Circuit:
    """SharedUmbrella 网表 -> circuit_agent IR。

    节点次序直接照搬：它们的 ``V V1 A B 125`` 约定 V_A − V_B = 125，
    而 IR 里电压源 ``nodes[0]`` 就是 +端 —— 两边的 **+端定义一致**，不用翻转。
    （会翻转的只有"电压源电流参考方向"，那在对账时单独处理。）
    """
    comps = []
    for b in branches:
        comps.append(Component(
            ref=b.name, kind=b.kind, nodes=(b.n1, b.n2), value=float(b.val),
            evidence=Evidence(source="shared_netlist", confidence=1.0,
                              detail=f"来自 SharedUmbrella 网表（{b.name}）"),
        ))
    return Circuit(name=name, components=comps, ref_node=ref)


def same_dir(c: Component, b) -> bool:
    """IR 的参考方向与 SharedUmbrella 的参考方向是否同向。"""
    f, t = declared_direction(c)
    return (f, t) == (b.n1, b.n2)


def frac_of(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x) if isinstance(x, int) else Fraction(repr(float(x)))


def check_one(sd, net_path: Path, want_spice: bool) -> int:
    branches, nodes, ref, probes = sd.parse_netlist(net_path.read_text("utf-8"))
    br_by_name = {b.name: b for b in branches}
    circ = to_agent_ir(branches, ref, name=net_path.stem)

    print("=" * 84)
    print(f"网表：{net_path.name}   {len(nodes)} 节点 / {len(branches)} 支路   参考节点 = {ref}")
    print("=" * 84)
    fails = 0

    # ---------------------------------------------------- 各自独立求解
    ag_mna = node_voltage_method(circ)
    ag_br = branch_current_method(circ)
    sh_V, sh_isrc = sd.nodal(branches, nodes, ref)
    sh_i, sh_vI, _loops, _icol = sd.branch_current(branches, nodes, ref)

    # ---------------------------------------------------- 1. 节点电压
    print("\n[1] 节点电压 —— 精确有理数逐项比对（这里是约定无关的，必须严格相等）")
    print(f"  {'节点':<6}{'circuit_agent':>20}{'SharedUmbrella':>22}{'差':>10}")
    for n in nodes:
        a = frac_of(ag_mna.get_node(n))
        s = frac_of(sh_V.get(n, Fraction(0)))
        ok = a == s
        if not ok:
            fails += 1
        print(f"  {n:<6}{str(a):>20}{str(s):>22}{'0' if ok else '≠ ' + str(a - s):>10}"
              + ("" if ok else "   ✘"))

    # ---------------------------------------------------- 2. 支路电流（含符号映射）
    print("\n[2] 支路电流 —— 按 IR 的参考方向比对（电压源两边约定相反，已显式翻转）")
    print(f"  {'支路':<7}{'类型':<5}{'circuit_agent':>18}{'SharedUmbrella':>20}")
    print(f"  {'':<7}{'':<5}{'（IR 方向）':>18}{'（换算到 IR 方向）':>20}")
    for c in circ.components:
        b = br_by_name[c.ref]
        a = frac_of(ag_br.get_current(c.ref))
        raw = frac_of(sh_i.get(c.ref))
        conv = raw if same_dir(c, b) else -raw
        ok = a == conv
        if not ok:
            fails += 1
        flip = "" if same_dir(c, b) else "（已翻转）"
        print(f"  {c.ref:<7}{c.kind:<5}{str(a):>18}{str(conv):>20}"
              + (f"   ✘ 原始 {raw} {flip}" if not ok else f"   {flip}"))

    # ---------------------------------------------------- 3. 每元件功率（物理量）
    print("\n[3] 每元件吸收功率 —— 物理量，与参考方向约定无关，两套必须给同一个数")
    print(f"  {'元件':<7}{'circuit_agent/W':>18}{'SharedUmbrella/W':>20}{'差':>10}")
    for c in circ.components:
        b = br_by_name[c.ref]
        f, t = declared_direction(c)
        pa = (frac_of(ag_mna.get_node(f)) - frac_of(ag_mna.get_node(t))) * frac_of(ag_br.get_current(c.ref))
        ps = (frac_of(sh_V.get(b.n1, Fraction(0))) - frac_of(sh_V.get(b.n2, Fraction(0)))) * frac_of(sh_i.get(b.name))
        ok = pa == ps
        if not ok:
            fails += 1
        print(f"  {c.ref:<7}{str(pa):>18}{str(ps):>20}{'0' if ok else '≠':>10}" + ("" if ok else "   ✘"))

    # ---------------------------------------------------- 4. 功率守恒（各自）
    print("\n[4] 功率守恒 ΣP = 0 —— 双方各自算，都必须成立")
    tot_a = sum(
        (frac_of(ag_mna.get_node(declared_direction(c)[0])) - frac_of(ag_mna.get_node(declared_direction(c)[1])))
        * frac_of(ag_br.get_current(c.ref)) for c in circ.components
    )
    tot_s = sum(
        (frac_of(sh_V.get(b.n1, Fraction(0))) - frac_of(sh_V.get(b.n2, Fraction(0)))) * frac_of(sh_i.get(b.name))
        for b in branches
    )
    for label, tot in (("circuit_agent", tot_a), ("SharedUmbrella", tot_s)):
        ok = tot == 0
        if not ok:
            fails += 1
        print(f"  [{'通过' if ok else '失败'}] {label:<16} ΣP = {tot}")
    ok_eq = tot_a == tot_s
    if not ok_eq:
        fails += 1
    print(f"  [{'通过' if ok_eq else '失败'}] 双方 ΣP 相等（{tot_a} == {tot_s}）")

    # ---------------------------------------------------- 5. 第三方法一致性（双方各自的 ngspice）
    if want_spice:
        print("\n[5] ngspice 复算 —— 两套实现各自的第三方路径")
        print("     ★ 这一节必须用**容差**比，不能用 == ：ngspice 走 double，")
        print("       两次独立运行之间本身就有 ~1e-10 的浮点噪声，这是物理限制不是算错。")
        from app.solver.ngspice import ngspice_method, probe_availability
        info = probe_availability()
        if not info.get("available"):
            print(f"  [跳过] circuit_agent 侧 ngspice 不可用：{info.get('reason')}")
        else:
            ag_ng = ngspice_method(circ)
            sh_Vs, _ = sd.spice(branches, nodes, ref)
            #  1e-6 绝对 + 1e-9 相对：与两套工具自己的判定阈值同量级
            def close(x: float, y: float, scale: float = 1.0) -> bool:
                return abs(x - y) <= max(1e-6, 1e-9 * max(abs(scale), 1.0))

            print(f"  {'节点':<6}{'agent/ngspice':>18}{'shared/ngspice':>18}"
                  f"{'精确解':>16}{'两者差':>13}")
            for n in nodes:
                x = float(ag_ng.get_node(n))
                y = float(sh_Vs.get(n, Fraction(0)))
                e = frac_of(ag_mna.get_node(n))
                ef = e.numerator / e.denominator
                d = abs(x - y)
                ok = close(x, y, scale=ef)
                if not ok:
                    fails += 1
                print(f"  {n:<6}{x:>18.10f}{y:>18.10f}{ef:>16.6f}{d:>13.2e}"
                      + ("" if ok else "   ✘ 超出容差"))
            #  更该问的是：两条 ngspice 各自离**精确解**有多远（这才是真正的验收）
            worst = 0.0
            for n in nodes:
                e = frac_of(ag_mna.get_node(n))
                ef = e.numerator / e.denominator
                worst = max(worst, abs(float(ag_ng.get_node(n)) - ef),
                            abs(float(sh_Vs.get(n, Fraction(0))) - ef))
            ok_w = worst <= 1e-6 * max(1.0, max(
                abs(float(ag_mna.get_node(n))) for n in nodes))
            if not ok_w:
                fails += 1
            print(f"  [{'通过' if ok_w else '失败'}] 两条 ngspice 对精确解的最大偏离 "
                  f"{worst:.2e} V（应只来自 double 精度）")

    print(f"\n  → 本项目差异项：{fails}")
    return fails

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("netlists", nargs="*")
    ap.add_argument("--spice", action="store_true", help="额外比 ngspice 路径（慢）")
    a = ap.parse_args()

    sd = load_shared_solver()

    paths = [Path(p) for p in a.netlists] or sorted((SHARED_TOOLS / "examples").glob("*.txt"))
    print(f"独立实现：{SHARED_SOLVER}")
    print(f"本次对账 {len(paths)} 份网表\n")

    total = 0
    for p in paths:
        total += check_one(sd, p, a.spice)
        print()

    print("=" * 84)
    print(f"跨工具对账总计差异项：{total}")
    print("=" * 84)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
