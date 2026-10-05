"""受控源（E/G/H/F）与自定义表达式：四条**各自独立**的验收。

★ 为什么四条要各自成立，缺一条都不算过：

1. **三路径逐项对账** —— 节点电压法（精确有理 MNA）、支路电流法（生成树基本回路）、
   ngspice（浮点）是三条**独立代码路径**。节点电压 + 支路电流 + 支路压降
   三项逐项相等，是"方程写对了"的必要条件。
2. **手算期望** —— 三法共用同一份 IR，所以「方程本身写错」这类错误，
   三条路径会**一致地**给出同一个错答案，互校根本拦不住。
   所以每个用例的期望值都是**独立手算**写在这里的（推演留在注释里），
   不是从程序输出抄的 —— 否则这个文件就变成自己验自己。
3. **网表往返** —— ``from_spice(to_spice(c))`` 逐项相等（含自动插入的探针源），
   且 H/F 的增益负号必须**反出去再反回来还是原值**。
   ★ 只反一边是最容易犯的错，而且三条路径会一致地差这一个负号，
   三法互校照样发现不了 —— 只有往返能拦住它。往返还必须**幂等**：
   跑两遍元件数不许变（``sense_ref`` 丢了的话每往返一次就多插一个探针）。
4. **文本网表实跑** —— 把 ``to_spice`` 生成的**文本**写成 .cir 交给 ngspice，
   读回的节点电压要与三法一致。前三步走的都是 PySpice 建模 API，
   只有这一步真正证明"网表文本本身写对了"。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_controlled.py
"""

from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import (                                          # noqa: E402
    CONTROLLED_KINDS, Circuit, CircuitError, Component, Control,
)
from app.ir.params import CONTROL_MODE, GAIN_SYMBOL, VALUE_UNIT, params_view  # noqa: E402
from app.ir.probes import SENSE_PREFIX, ensure_sense_sources         # noqa: E402
from app.ir.spice import (                                           # noqa: E402
    CTRL_MARK, canonical_netlist_view, from_spice, to_spice,
)
from app.ir.spice_expr import SPICE_SOURCE_CURRENT_SIGN              # noqa: E402
from app.solver import branch as BR                                  # noqa: E402
from app.solver import mna as MN                                     # noqa: E402
from app.solver import ngspice as NG                                 # noqa: E402
from app.solver.reconcile import run_all                             # noqa: E402

FAILS = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global FAILS
    if not ok:
        FAILS += 1
    line = f"  [{'通过' if ok else '失败'}] {name}"
    if detail:
        line += f"\n           {detail}"
    print(line)
    return ok


def banner(text: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {text}")
    print("#" * 72)


def close(a, b) -> bool:
    """跨路径比较：ngspice 是浮点，两次运行本身有 ~1e-10 噪声，必须给容差。"""
    if a is None or b is None:
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))


def C(ref, kind, a, b, value=None, ctrl=None, note=""):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value,
                     ctrl=ctrl, note=note)


def vctrl(x, y):
    return Control(mode="V", nodes=(x, y))


def ictrl(ref):
    return Control(mode="I", ref=ref)


def ectrl(ref, expr):
    """电流控制 + 自定义表达式。**不需要探针**：表达式里的电流由参数名解析。"""
    return Control(mode="I", ref=ref, expr=expr)


# ---------------------------------------------------------------- 七道题
#
# 每道题的期望值都是手算的，推演写在函数体里。**不要**改成从输出里抄。


def case_E() -> tuple[Circuit, dict]:
    """E（VCVS，压控压源）。增益 μ = 3，控制端 V(2) − V(0)。

        V1 = 10V(1-0), R1 = 1kΩ(1-2), R2 = 2kΩ(2-0)
        E1 = 3·(V(2) − V(0))(3-0), R3 = 1kΩ(3-0)

        节点 2：（V2 − 10)/1000 + V2/2000 = 0   乘 2000
                2(V2 − 10) + V2 = 0  →  3V2 = 20  →  V2 = 20/3 V
        控制量：V(2) − V(0) = 20/3
        E1 输出：V3 = 3 × 20/3 = 20 V
        i(E1)：节点 3 上只挂 E1 与 R3，R3 从节点 3 取走 20/1000 = 1/50 A，
               所以 E1 必须供出同样多 → i(E1) = +1/50（IR 约定 i > 0 = 供电）
        u(E1)：支路压降沿**参考方向**。declared_direction(电压输出元件) = (nodes[1], nodes[0])
               = (0, 3)，故 drop = V0 − V3 = −20 V
        功率核对：P(E1) = −20 × 1/50 = −0.4 W（提供）；R3 消耗 20²/1000 = 0.4 W ✓
    """
    c = Circuit(name="E-压控压源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 2000.0),
        C("E1", "E", "3", "0", 3.0, ctrl=vctrl("2", "0")),
        C("R3", "R", "3", "0", 1000.0),
    ])
    return c, {"V(2)": Fraction(20, 3), "V(3)": Fraction(20),
               "i(E1)": Fraction(1, 50), "u(E1)": Fraction(-20)}


def case_G() -> tuple[Circuit, dict]:
    """G（VCCS，压控流源）。转移电导 gm = 2 mS，控制端 V(1) − V(0)。

        V1 = 10V(1-0), R2 = 1kΩ(1-2), G1 = 2mS·(V(1) − V(0))(2-0)

        G1 的参考方向是 nodes[0] → nodes[1] = 2 → 0（电流输出元件），
        所以它把 gm·10 = 0.02 A 从节点 2 抽向节点 0。
        节点 2：（V2 − 10)/1000 + 0.02 = 0  →  V2 = 10 − 20 = −10 V
        i(G1) = 0.02 = 1/50 A
        u(G1) = V2 − V0 = −10 V
        功率核对：R2 消耗 20²/1000 = 0.4 W；G1 消耗 −10 × 0.02 = −0.2 W（提供 0.2 W）；
                 V1 提供 10 × 0.02 = 0.2 W ✓
    """
    c = Circuit(name="G-压控流源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R2", "R", "1", "2", 1000.0),
        C("G1", "G", "2", "0", 0.002, ctrl=vctrl("1", "0")),
    ])
    return c, {"V(2)": Fraction(-10),
               "i(G1)": Fraction(1, 50), "u(G1)": Fraction(-10)}


def case_H() -> tuple[Circuit, dict]:
    """H（CCVS，流控压源）。转移电阻 rm = 2000 Ω，被采样支路 R1。**需要 0V 探针源。**

        V1 = 10V(1-0), R1 = 1kΩ(1-2), R2 = 1kΩ(2-0)
        H1 = 2000·i(R1)(3-0), R3 = 1kΩ(3-0)

        节点 2：R1 与 R2 分压 → V2 = 5 V，i(R1) 沿 1→2 = (10 − 5)/1000 = 5 mA
        H1 输出：V3 = 2000 × 0.005 = 10 V
        i(H1) = V3/R3 = 10/1000 = 1/100 A（供出给 R3）
        u(H1) = V0 − V3 = −10 V
        ★ 探针方向核对（最容易差号的地方）：
          探针 Vsense_R1 的 nodes = (b, mid) = (2, ns_R1)，
          declared_direction(电压源) = (nodes[1], nodes[0]) = (ns_R1, 2)，即"内部 ns→2"。
          而实际电流沿 1 → R1 → ns_R1 → 探针 → 2，正是 ns_R1 → 2。
          所以探针电流 = +5 mA = i(R1)，**同号**，控制量取值不会整体差号。
          若探针 nodes 写成 (mid, b)（方向反过来），取到的会是 −5 mA，
          而三条路径会一致地差这一个负号 —— 互校发现不了。
    """
    c = Circuit(name="H-流控压源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 1000.0),
        C("H1", "H", "3", "0", 2000.0, ctrl=ictrl("R1")),
        C("R3", "R", "3", "0", 1000.0),
    ])
    return c, {"V(2)": Fraction(5), "V(3)": Fraction(10),
               "i(H1)": Fraction(1, 100), "u(H1)": Fraction(-10)}


def case_F() -> tuple[Circuit, dict]:
    """F（CCCS，流控流源）。α = 3，被采样支路 R1。**需要 0V 探针源。**

        V1 = 10V(1-0), R1 = 1kΩ(1-0), R2 = 1kΩ(1-2), F1 = 3·i(R1)(2-0)

        i(R1) 沿 1→0 = 10/1000 = 10 mA
        F1 参考方向 2 → 0，所以它把 3 × 0.01 = 0.03 A 从节点 2 抽走
        节点 2：（V2 − 10)/1000 + 0.03 = 0  →  V2 = 10 − 30 = −20 V
        i(F1) = 0.03 = 3/100 A
        u(F1) = V2 − V0 = −20 V
        功率核对：F1 消耗 −20 × 0.03 = −0.6 W（提供）；R2 消耗 30²/1000 = 0.9 W；
                 R1 消耗 10²/1000 = 0.1 W；V1 提供 10 × 0.04 = 0.4 W
                 （V1 支路 = R1 的 0.01 + R2 的 0.03）
                 提供 0.6 + 0.4 = 0.9 + 0.1 = 1.0，消耗 1.0 ✓
    """
    c = Circuit(name="F-流控流源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("R2", "R", "1", "2", 1000.0),
        C("F1", "F", "2", "0", 3.0, ctrl=ictrl("R1")),
    ])
    return c, {"V(2)": Fraction(-20),
               "i(F1)": Fraction(3, 100), "u(F1)": Fraction(-20)}


def case_expr_v() -> tuple[Circuit, dict]:
    """自定义表达式：E1 的输出 = 3·u_2 + 5（**按参数名书写**，不是增益的标准形）。

        电路与 case_E 相同，只是把增益换成表达式。
        u_2 就是 V(2) = 20/3，所以 V3 = 3 × 20/3 + 5 = 25 V
        i(E1) = 25/1000 = 1/40 A
        u(E1) = V0 − V3 = −25 V
        ★ 表达式路径与标准形路径是**两条不同的代码路径**（一个走参数表解析、
          一个直接乘增益），必须各测一遍：实测过一次"标准形差号、表达式不差号"。
    """
    c = Circuit(name="表达式-线性组合", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 2000.0),
        C("E1", "E", "3", "0", None, ctrl=Control(mode="V", nodes=("2", "0"),
                                                  expr="3*u_2 + 5")),
        C("R3", "R", "3", "0", 1000.0),
    ])
    return c, {"V(2)": Fraction(20, 3), "V(3)": Fraction(25),
               "i(E1)": Fraction(1, 40), "u(E1)": Fraction(-25)}


def case_expr_i_v() -> tuple[Circuit, dict]:
    """表达式里**引用支路电流**，输出是电压（E 型）。

        V1 = 10V(1-0), R1 = 1kΩ(1-2), R2 = 1kΩ(2-0)
        E1 = 2000·i_R1 (3-0), R3 = 1kΩ(3-0)

        V2 = 5 V，i_R1 = 5 mA → V3 = 2000 × 0.005 = 10 V
        i(E1) = 10/1000 = 1/100 A
        ★ 这条走的是 ``spice_expression`` 的 ``I(...)`` 片段，与上面 H 卡
          （标准形）是不同代码路径；IR 的电流符号与 SPICE 的 ``I(V)`` **反号**，
          由 ``SPICE_SOURCE_CURRENT_SIGN`` 统一修正 —— 这里就是它的回归点。
    """
    c = Circuit(name="表达式-引用支路电流(压控)", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 1000.0),
        C("E1", "E", "3", "0", None,
          ctrl=Control(mode="I", ref="R1", expr="2000*i_R1")),
        C("R3", "R", "3", "0", 1000.0),
    ])
    return c, {"V(2)": Fraction(5), "V(3)": Fraction(10),
               "i(E1)": Fraction(1, 100), "u(E1)": Fraction(-10)}


def case_expr_i_i() -> tuple[Circuit, dict]:
    """同上，但输出是电流（F 型），且**不插探针**（表达式自带电流解析）。

        F1 = 3·i_R1 (2-0)。i_R1 = 10 mA → i(F1) = 0.03 A，V2 = −20 V。
    """
    c = Circuit(name="表达式-引用支路电流(流控)", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("R2", "R", "1", "2", 1000.0),
        C("F1", "F", "2", "0", None,
          ctrl=Control(mode="I", ref="R1", expr="3*i_R1")),
    ])
    return c, {"V(2)": Fraction(-20),
               "i(F1)": Fraction(3, 100), "u(F1)": Fraction(-20)}


CASES = [
    ("E 压控压源（VCVS）", case_E),
    ("G 压控流源（VCCS）", case_G),
    ("H 流控压源（CCVS，自动插 0V 探针）", case_H),
    ("F 流控流源（CCCS，自动插 0V 探针）", case_F),
    ("自定义表达式（节点电压，线性组合）", case_expr_v),
    ("自定义表达式（引用支路电流 → 输出压，不插探针）", case_expr_i_v),
    ("自定义表达式（引用支路电流 → 输出流，不插探针）", case_expr_i_i),
]


# ---------------------------------------------------------------- 1. 三路径对账


def solve_three(circ: Circuit) -> dict:
    """三条独立路径各算一遍。任一路失败都**如实记下来**，不许静默少一条。"""
    out: dict = {}
    for label, fn in (("节点法", MN.node_voltage_method),
                      ("支路法", BR.branch_current_method),
                      ("ngspice", NG.ngspice_method)):
        try:
            out[label] = fn(circ)
        except Exception as e:                                       # noqa: BLE001
            print(f"           [路径失败] {label}: {type(e).__name__}: {e}")
    return out


def test_crosscheck() -> None:
    banner("1. 三路径逐项对账 + 手算期望（节点电压 / 支路电流 / 支路压降）")
    for name, fn in CASES:
        circ, expect = fn()
        print(f"\n--- {name} ---")
        try:
            circ.validate()
        except CircuitError as e:
            check(f"{name}: 结构校验", False, str(e))
            continue

        sols = solve_three(circ)
        check(f"{name}: 三条路径都算出来了", len(sols) == 3,
              "缺失：" + "、".join(k for k in ("节点法", "支路法", "ngspice")
                                 if k not in sols))
        if len(sols) < 3:
            continue

        # ---- 逐项对账（三项全要：节点电压、支路电流、支路压降）
        ref = sols["节点法"]
        bad: list[str] = []
        for label in ("支路法", "ngspice"):
            other = sols[label]
            for n in ref.node_voltages:
                if not close(ref.node_voltages.get(n), other.node_voltages.get(n)):
                    bad.append(f"{label} u({n}): {ref.node_voltages.get(n)} vs "
                               f"{other.node_voltages.get(n)}")
            for r in ref.branch_currents:
                if not close(ref.branch_currents.get(r), other.branch_currents.get(r)):
                    bad.append(f"{label} i({r}): {ref.branch_currents.get(r)} vs "
                               f"{other.branch_currents.get(r)}")
            for r in ref.branch_drops:
                if not close(ref.branch_drops.get(r), other.branch_drops.get(r)):
                    bad.append(f"{label} u({r}): {ref.branch_drops.get(r)} vs "
                               f"{other.branch_drops.get(r)}")
        n_items = (len(ref.node_voltages) + len(ref.branch_currents)
                   + len(ref.branch_drops))
        check(f"{name}: 三法 {n_items} 项逐项相等", not bad,
              "；".join(bad[:4]))

        # ---- 手算期望（精确有理数，与节点法逐项严格相等）
        wrong: list[str] = []
        for key, want in expect.items():
            tag, key_name = key[0], key[2:-1]
            if tag == "V":
                got = ref.node_voltages.get(key_name)
            elif tag == "i":
                got = ref.branch_currents.get(key_name)
            else:
                got = ref.branch_drops.get(key_name)
            if got is None:
                wrong.append(f"{key} 在手算期望里有、解里却没有")
            elif Fraction(got) != want:
                wrong.append(f"{key}: 手算 {want}，实得 {got}")
        check(f"{name}: {len(expect)} 项手算期望全部命中（精确有理数）",
              not wrong, "；".join(wrong))

        # ---- 单条最强判据：功率守恒。放在三法对账之后单独报，
        #      因为"三法一致"和"物理上说得通"是两件事。
        pack = run_all(circ)
        pw = pack["verifications"]["mna"]["checks"][0]
        check(f"{name}: ΣP = 0（功率守恒）", bool(pw["ok"]), pw["conclusion"])
        check(f"{name}: run_all 总判定通过", bool(pack["overall_pass"]),
              "；".join(f"{k}={v}" for k, v in pack["failures"].items()))
        if ref.detail.get("controlled"):
            for row in ref.detail["controlled"]:
                print(f"           控制关系：{row['ref']} → {row['control']}"
                      f"（{row['mode']}）")


# ---------------------------------------------------------------- 2. 探针源


def test_sense_probes() -> None:
    banner("2. 0V 测量探针源：只在需要时插、留痕、幂等、不写回原电路")

    # ---- H / F：需要探针
    for name, fn, target in (("H", case_H, "R1"), ("F", case_F, "R1")):
        circ, _ = fn()
        circ.validate()
        # ★ 展开前的端子要**先抄下来**再比 —— 写死 ("1","2") 是错的：
        #   F 那道题里 R1 是 (1,0)。断言要观察的是"有没有被动过"，
        #   不是"值是多少"，所以基准必须来自同一份原电路。
        before = {c.ref: c.nodes for c in circ.components}
        expanded, rec = ensure_sense_sources(circ)
        check(f"{name}: 确实插入了 1 个探针源", rec["applied"]
              and len(rec["inserted"]) == 1,
              f"inserted={rec['inserted']}")
        if rec["inserted"]:
            ins = rec["inserted"][0]
            check(f"{name}: 探针位号带 {SENSE_PREFIX} 前缀（一眼可辨，不是题目元件）",
                  ins["ref"].startswith(SENSE_PREFIX), ins["ref"])
            check(f"{name}: 探针位号首字母是 V（H/F 卡只认电压源）",
                  ins["ref"][0] == "V", ins["ref"])
            check(f"{name}: 留痕写明为谁而插、插在哪条支路上",
                  ins["for"] == name + "1" and ins["target"] == target
                  and "不是题目元件" in rec["notes"][-1],
                  f"for={ins['for']} target={ins['target']}")
            probe = expanded.by_ref(ins["ref"])
            check(f"{name}: 探针确实是 0V 电压源", probe.kind == "V"
                  and float(probe.value) == 0.0, f"kind={probe.kind} value={probe.value}")
            check(f"{name}: 探针 note 声明它不是题目元件",
                  "不是题目元件" in probe.note, probe.note[:50])

        # ★ 原电路一个字段都不许动：插进来的探针会移动元件端子（R1 的 b 端
        #   变成内部节点），一旦写回会话，界面上的接线与叠图坐标就全错了。
        check(f"{name}: 原电路的元件数与端子**原样未动**",
              len(circ.components) == rec["original_components"]
              and {c.ref: c.nodes for c in circ.components} == before,
              f"components={len(circ.components)} "
              f"端子={ {c.ref: c.nodes for c in circ.components} }")

        # ---- 幂等：对**已展开**的电路再跑一次，不许再插一个
        again, rec2 = ensure_sense_sources(expanded)
        check(f"{name}: 幂等（对已展开的电路再插一次，元件数不变）",
              len(again.components) == len(expanded.components)
              and not rec2["applied"],
              f"{len(expanded.components)} → {len(again.components)}")

    # ---- 表达式形式：**不需要**探针（表达式里的电流由参数名解析）
    for name, fn in (("E+表达式引用电流", case_expr_i_v),
                     ("F+表达式引用电流", case_expr_i_i)):
        circ, _ = fn()
        circ.validate()
        expanded, rec = ensure_sense_sources(circ)
        check(f"{name}: 不插探针（表达式自带电流解析，插了就是往题目里塞多余元件）",
              not rec["applied"] and len(expanded.components) == len(circ.components),
              f"applied={rec['applied']}")

    # ---- 探针数量必须与"需要探针的受控源个数"一致，不是与受控源总数一致
    circ = Circuit(name="混合", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 1000.0),
        C("E1", "E", "3", "0", 2.0, ctrl=vctrl("2", "0")),        # 压控 → 不要探针
        C("H1", "H", "3", "0", 100.0, ctrl=ictrl("R1")),          # 流控标准形 → 要
        C("F1", "F", "3", "0", 2.0, ctrl=ictrl("R2")),            # 流控标准形 → 要
        C("G1", "G", "3", "0", 0.001, ctrl=vctrl("2", "0")),      # 压控 → 不要
        C("R3", "R", "3", "0", 1000.0),
    ])
    expanded, rec = ensure_sense_sources(circ)
    check("混合题：只给流控的插探针（2 个），压控的一个都不插",
          len(rec["inserted"]) == 2, f"inserted={[i['ref'] for i in rec['inserted']]}")

    # ---- 被采样支路本身就是电压源：直接引用它，不插探针
    c2 = Circuit(name="采样电压源", components=[
        C("Vs", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("H1", "H", "2", "0", 2000.0, ctrl=ictrl("Vs")),
        C("R2", "R", "2", "0", 1000.0),
    ])
    ex2, rec3 = ensure_sense_sources(c2)
    check("被采样支路本身就是电压源：直接引用，不插探针",
          not rec3["applied"] and ex2.by_ref("H1").ctrl.sense_ref == "Vs",
          f"sense_ref={ex2.by_ref('H1').ctrl.sense_ref!r} notes={rec3['notes']}")

    # ---- 采样不存在的支路：明确报错，点名是哪个受控源
    c3 = Circuit(name="采样不存在", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("H1", "H", "2", "0", 2000.0, ctrl=ictrl("RX")),
        C("R2", "R", "2", "0", 1000.0),
    ])
    try:
        ensure_sense_sources(c3)
        check("采样不存在的支路：明确报错", False, "没有报错")
    except CircuitError as e:
        check("采样不存在的支路：明确报错并点名受控源", "H1" in str(e), str(e))


# ---------------------------------------------------------------- 3. 网表往返


def test_netlist_roundtrip() -> None:
    banner("3. 网表往返：from_spice(to_spice(c)) 逐项相等 + 负号来回 + 幂等")

    for name, fn in CASES:
        circ, _ = fn()
        circ.validate()
        text = to_spice(circ, title=name)

        if fn in (case_expr_v, case_expr_i_v, case_expr_i_i):
            # ★ 自定义表达式落盘成**行为源 B**，而 B 卡读不回来（本版本不支持）。
            #   这是刻意的：能写出去（给 ngspice 算）但读不回来，必须**明确报错**，
            #   而不是读回一个"看着像、其实丢了表达式"的电路。
            try:
                from_spice(text)
                check(f"{name}: 行为源 B 本该明确拒绝，却读进来了", False)
            except CircuitError as e:
                check(f"{name}: 行为源 B 明确拒绝（" + str(e).splitlines()[0][:38] + "…）",
                      "B" in text)
            continue

        try:
            back = from_spice(text)
        except Exception as e:                                       # noqa: BLE001
            check(f"{name}: 往返读取", False, f"{type(e).__name__}: {e}")
            continue

        # 往返的目标是**展开后**的电路：to_spice 默认给 H/F 插 0V 探针源，
        # 那正是被求解的那一张，网表文本里当然也有它。
        want_circuit, _ = ensure_sense_sources(circ)
        left = sorted((x.ref, x.kind, x.nodes, x.value) for x in want_circuit.components)
        rght = sorted((x.ref, x.kind, x.nodes, x.value) for x in back.components)
        extra = "" if len(left) == len(circ.components) else \
            f"（含自动插入的探针，{len(circ.components)} → {len(left)} 条支路）"
        check(f"{name}: 往返 {len(left)} 项元件逐项相等{extra}", left == rght,
              f"期望={left}\n           实得={rght}")

        # 控制关系也必须活着回来（位号 / 控制端 / 采样支路）
        bad: list[str] = []
        for a in want_circuit.components:
            if not a.is_controlled:
                continue
            b = back.by_ref(a.ref)
            if b.ctrl is None:
                bad.append(f"{a.ref}: 受控源标记丢了")
                continue
            if b.ctrl.mode != a.ctrl.mode:
                bad.append(f"{a.ref}: 控制方式 {a.ctrl.mode} → {b.ctrl.mode}")
            if a.ctrl.mode == "V" and tuple(b.ctrl.nodes or ()) != tuple(a.ctrl.nodes or ()):
                bad.append(f"{a.ref}: 控制端 {a.ctrl.nodes} → {b.ctrl.nodes}")
            if a.ctrl.mode == "I" and b.ctrl.ref != a.ctrl.ref:
                bad.append(f"{a.ref}: 被采样支路 {a.ctrl.ref} → {b.ctrl.ref}")
        check(f"{name}: 控制关系（方式 / 控制端 / 被采样支路）往返无损", not bad,
              "；".join(bad))

        # ★ 幂等：再走一遍往返，元件数一个都不许多 —— sense_ref 若丢失，
        #   每往返一次就会多插一个探针源，而那种"慢慢长胖"最难被发现。
        text2 = to_spice(back, title=name)
        back2 = from_spice(text2)
        check(f"{name}: 往返幂等（第二遍元件数不变）",
              len(back2.components) == len(back.components),
              f"{len(back.components)} → {len(back2.components)}")

    # ---- H/F 的增益负号：必须"反出去，再反回来"
    for name, fn, ref in (("H 流控压源", case_H, "H1"), ("F 流控流源", case_F, "F1")):
        circ, _ = fn()
        circ.validate()
        text = to_spice(circ, title=name)
        card = [ln for ln in text.splitlines() if ln.startswith(ref)][0]
        back = from_spice(text)
        orig = float(circ.by_ref(ref).value)
        card_gain = float(card.split()[-1])
        got = float(back.by_ref(ref).value)
        check(f"{name}: 卡上增益 = −IR 值（SPICE 的 I(V) 与 IR 反号），且读回来还原",
              close(card_gain, SPICE_SOURCE_CURRENT_SIGN * orig) and close(got, orig),
              f"IR={orig} 卡上={card_gain} 读回={got}（卡：{card}）")

    # ---- 探针源在网表里有独立注释，不是静默塞进去的
    circ, _ = case_F()
    circ.validate()
    text = to_spice(circ)
    check("F1 的网表里出现自动插入的探针源 Vsense_R1",
          f"{SENSE_PREFIX}R1" in text, "")
    check("网表对自动插入的探针有独立注释（不是静默塞进去的）",
          "测量探针" in text, "")

    view = canonical_netlist_view(case_H()[0])
    probe_rows = [r for r in view["rows"] if r.get("probe")]
    check("H1 的网表视图恰好 1 行探针（供界面/报告单列出来）",
          len(probe_rows) == 1, f"实际 {len(probe_rows)} 行")

    test_ctrl_mark()


def test_ctrl_mark() -> None:
    """★ 网表语义标记 ``* ca-ctrl …``：往返后「被采样支路」必须还是题目那条。

    没有它就会这样：``H``/``F`` 卡引用的是系统插入的 0V 探针，读回来
    ``ctrl.ref`` 就从 ``R1`` 变成 ``Vsense_R1`` —— 电学等价，但
    **参数指向错了**：界面上「被采样支路」那一格会显示成探针，
    用户看到的控制关系从 ``i(R1)`` 变成 ``i(Vsense_R1)``。
    """
    print("\n--- 网表语义标记 * ca-ctrl ---")
    for name, fn, ref, sampled in (("H", case_H, "H1", "R1"),
                                   ("F", case_F, "F1", "R1")):
        circ, _ = fn()
        circ.validate()
        text = to_spice(circ)
        want_line = f"{CTRL_MARK} {ref} mode=I ref={sampled} sense={SENSE_PREFIX}{sampled}"
        check(f"{name}: 网表里写了语义标记行（{want_line}）", want_line in text,
              "\n           ".join(ln for ln in text.splitlines()
                                   if CTRL_MARK.split()[-1] in ln)[:200])

        back = from_spice(text)
        b = back.by_ref(ref)
        check(f"{name}: 往返后被采样支路仍是题目里的 {sampled}（不是探针）",
              b.ctrl.ref == sampled,
              f"ref={b.ctrl.ref!r} sense_ref={b.ctrl.sense_ref!r}")
        check(f"{name}: 探针仍被记为实际采样支路（求解走它）",
              b.ctrl.sampling == f"{SENSE_PREFIX}{sampled}",
              f"sampling={b.ctrl.sampling!r}")
        diag = [d["text"] for d in back.diagnostics]
        check(f"{name}: 还原这件事**写进了 diagnostics**（不静默）",
              any("按网表语义标记把被采样支路还原" in d for d in diag),
              "\n           ".join(diag)[:220])
        # 还原之后控制关系的人读写法必须指向用户认得的东西
        check(f"{name}: 人读控制关系里出现的是题目支路名，不只有探针名",
              f"i({sampled})" in b.ctrl.describe(b.kind, b.value)
              and b.ctrl.sampling in b.ctrl.describe(b.kind, b.value),
              b.ctrl.describe(b.kind, b.value))

        # ---- 还原后必须**照样能解**，且解与原来完全一致
        try:
            pack = run_all(back)
            ok = bool(pack["overall_pass"])
        except Exception as e:                                       # noqa: BLE001
            ok, pack = False, e
        check(f"{name}: 还原后的电路照常求解并通过总判定", ok,
              "" if ok else str(pack)[:160])

    # ---- 标记与卡片不符时：以卡片为准，并说明原因，**绝不动电路连接**
    circ, _ = case_H()
    circ.validate()
    text = to_spice(circ)
    tampered = text.replace(
        f"{CTRL_MARK} H1 mode=I ref=R1 sense=Vsense_R1",
        # 谎称探针是 R2（卡片上写的仍是 Vsense_R1）—— 必须被拒
        f"{CTRL_MARK} H1 mode=I ref=R2 sense=R2")
    back = from_spice(tampered)
    b = back.by_ref("H1")
    diag = [d["text"] for d in back.diagnostics]
    check("标记与卡片不符时以卡片为准（ref 仍是卡上引用的探针）",
          b.ctrl.ref == f"{SENSE_PREFIX}R1",
          f"ref={b.ctrl.ref!r}")
    check("标记被忽略时给出了原因（不是悄悄丢掉）",
          any("标记已忽略" in d for d in diag), "\n           ".join(diag)[:220])

    # ---- 标记指向不存在的支路 / 根本不是受控源
    for bad_mark, why in (
        (f"{CTRL_MARK} H1 mode=I ref=RX sense=Vsense_R1", "被采样支路不在网表里"),
        (f"{CTRL_MARK} R2  mode=I ref=R1 sense=Vsense_R1", "那个位号不是受控源"),
        (f"{CTRL_MARK} H1 mode=V ref=R1 sense=Vsense_R1", "标记自称是电压控制"),
        (f"{CTRL_MARK} H1 mode=I ref=R1", "标记缺 sense= 字段"),
    ):
        base = to_spice(case_H()[0]).replace(
            f"{CTRL_MARK} H1 mode=I ref=R1 sense=Vsense_R1", bad_mark)
        b2 = from_spice(base)
        diag2 = [d["text"] for d in b2.diagnostics]
        check(f"坏标记（{why}）被忽略且给出原因",
              any("标记已忽略" in d for d in diag2),
              "\n           ".join(d for d in diag2 if "标记" in d)[:220])

    # ---- 外来网表（没有标记）不许被动：探针名就是它的被采样支路
    foreign = ("V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n"
               "VsenseX 2 3 DC 0\nH1 4 0 VsenseX 2000\nR3 4 0 1k\n.end\n")
    fb = from_spice(foreign)
    check("外来网表（无标记）保持原样：ref 就是卡上引用的那个器件",
          fb.by_ref("H1").ctrl.ref == "VsenseX",
          f"ref={fb.by_ref('H1').ctrl.ref!r}")
    check("外来网表不会凭空多出 diagnostics 里的'还原'记录",
          not any("还原" in d["text"] for d in fb.diagnostics), "")


# ---------------------------------------------------------------- 4. 文本实跑


def test_netlist_runs() -> None:
    banner("4. 生成的网表**文本**交给 ngspice 实跑，与三法逐项对账")

    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.LIBRARY_PATH = str(NG.find_ngspice_dll())
        ng = NgSpiceShared.new_instance()
    except Exception as e:                                           # noqa: BLE001
        check("ngspice 共享库可用", False, f"{type(e).__name__}: {e}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="ca_controlled_"))

    def run_netlist(text: str) -> dict[str, float]:
        """**唯一**一条不经过 PySpice 建模 API、只看我们生成的文本的验收路径。"""
        f = tmp / f"case_{abs(hash(text)) % 10 ** 8}.cir"
        f.write_text(text, encoding="utf-8")
        ng.source(str(f))
        ng.run()
        plot = ng.plot(None, ng.plot_names[0])
        volts = {w.name.lower(): float(w[0]) for w in plot.nodes()}
        ng.remove_circuit()
        return volts

    for name, fn in CASES:
        circ, _ = fn()
        circ.validate()
        text = to_spice(circ, title=name)
        try:
            volts = run_netlist(text)
        except Exception as e:                                       # noqa: BLE001
            check(f"{name}: 文本网表跑得动", False, f"{type(e).__name__}: {e}")
            continue

        sol = MN.node_voltage_method(circ)
        bad = []
        for n, want in sol.node_voltages.items():
            if n == circ.ref_node:
                continue
            got = volts.get(n.lower())
            if got is None:
                bad.append(f"节点 {n} 读数缺失")
            elif not close(got, want):
                bad.append(f"节点 {n}: 三法={float(want):.6g} 文本网表={got:.6g}")
        shown = {n: round(float(v), 6) for n, v in sol.node_voltages.items()
                 if n != circ.ref_node}
        check(f"{name}: 文本网表与三法节点电压逐项一致 {shown}", not bad,
              "；".join(bad))


# ---------------------------------------------------------------- 5. 拒绝


def test_rejects() -> None:
    banner("5. 不该支持的，必须明确报错（不许静默降级）")

    # ---- 元件层：受控源没有控制支路 / 普通元件带了控制支路
    try:
        Component("E1", "E", ("3", "0"), 3.0)
        check("受控源缺控制支路：构造时即报错", False, "没有报错")
    except CircuitError as e:
        check("受控源缺控制支路：构造时即报错", "E" in str(e), str(e)[:70])

    try:
        Component("R1", "R", ("3", "0"), 3.0, ctrl=vctrl("2", "0"))
        check("普通电阻带控制支路：构造时即报错", False, "没有报错")
    except CircuitError as e:
        check("普通电阻带控制支路：构造时即报错", "不是受控源" in str(e), str(e)[:70])

    try:
        Control(mode="V", nodes=("2", "2"))
        check("控制端两端同一节点：构造时即报错（控制量恒 0）", False, "没有报错")
    except CircuitError as e:
        check("控制端两端同一节点：构造时即报错（控制量恒 0）", True, str(e)[:70])

    try:
        Control(mode="I")
        check("电流控制缺被采样支路：构造时即报错", False, "没有报错")
    except CircuitError as e:
        check("电流控制缺被采样支路：构造时即报错", True, str(e)[:70])

    # ---- 网表层：读不回来的必须明说
    for bad_text, why in (
        ("B9 3 0 V = V(2)*2\n.end\n", "行为源 B（自定义表达式的落盘形式）"),
        ("V1 1 0 DC 10\nR1 1 2 1k\nH1 3 0 R1 2000\nR3 3 0 1k\n.end\n",
         "H 卡控制端写了电阻位号"),
        ("E1 3 0 2 2000\n.end\n", "E 卡缺一列"),
        ("V1 1 0 DC 10\nR1 1 0 1k\nG1 2 0 2\n.end\n", "G 卡缺两列"),
    ):
        try:
            from_spice(bad_text)
            check(f"{why}：明确报错", False, "本该报错却读进来了")
        except CircuitError as e:
            check(f"{why}：明确报错", True, str(e).splitlines()[0][:78])

    # ---- 常量表必须自洽：四种受控源的控制方式与增益单位按定义推出
    expect_mode = {"E": "V", "G": "V", "H": "I", "F": "I"}
    expect_unit = {"E": "V/V", "G": "S", "H": "Ω", "F": "A/A"}
    check("CONTROL_MODE 与四类受控源的定义一致", dict(CONTROL_MODE) == expect_mode,
          f"{dict(CONTROL_MODE)}")
    check("VALUE_UNIT 里受控源的增益单位与定义一致",
          all(VALUE_UNIT[k] == v for k, v in expect_unit.items()),
          f"{ {k: VALUE_UNIT[k] for k in expect_mode} }")
    check("GAIN_SYMBOL 四类齐全（μ / gm / rm / α）",
          set(GAIN_SYMBOL) == set(CONTROLLED_KINDS), f"{GAIN_SYMBOL}")

    # ---- 参数表：受控源增益要在参数表里有一行，单位是它自己的单位
    circ = case_G()[0]
    circ.validate()
    pv = params_view(circ)
    row = [e for e in pv["items"] if e["binder"] == ["value", "G1"]]
    check("受控源的增益在参数表里占一行，且单位是 S（不是 Ω）",
          len(row) == 1 and row[0]["unit"] == "S",
          f"{row}")
    ctl = [x for x in pv["controlled"] if x["ref"] == "G1"]
    check("受控源在 controlled 里带控制方式、控制量与增益符号",
          len(ctl) == 1 and ctl[0]["mode"] == "V" and ctl[0]["gain_symbol"] == "gm"
          and ctl[0]["control_nodes"] == ["1", "0"],
          f"{ctl}")


# ---------------------------------------------------------------- 跑


def main() -> int:
    print("受控源与自定义表达式：四条独立验收")
    test_crosscheck()
    test_sense_probes()
    test_netlist_roundtrip()
    test_netlist_runs()
    test_rejects()
    print("\n" + "=" * 72)
    print(f"总计失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
