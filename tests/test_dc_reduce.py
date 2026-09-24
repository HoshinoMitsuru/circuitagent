"""直流稳态化简（C 开路 / L 短路）的断言测试。

★ **为什么单独一个文件**：在这之前 ``dc_reduce.py`` 一个断言测试都没有 ——
四套自检里没有任何一个 import 它，也没有任何用例构造过 C/L 元件。
它只在 KiCad demo 冒烟里被**顺带**执行，而那些断言只看解析与可解性，
不看化简结果。后果是：**化简层算错时四套自检可以全绿。**

实测踩到的正是这类错误。旧版把"两端落进同一个节点"的元件一律当作
"被短接、等价于消失"删掉，于是 ``V1(10V)`` 与 ``L1`` 并联时电压源被悄悄删掉，
剩下的电路照常可解 → 三条路径一致地给出全零解 → 功率守恒平凡成立（ΣP = 0）→
最后报告 ``overall_pass = True`` 并告诉学生"这张电路的读图与建模**可以采信**"。
而 ngspice 独立复核给出 ``singular matrix: check node l1#branch``，
Dynamic/True gmin stepping 与 source stepping 全部失败 —— 该电路**确实无解**。

所以这里的每一个期望值都是**独立手算**出来的（推导写在各自 docstring 里），
不是从程序输出里抄的 —— 否则测试就变成"自己验自己"。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_dc_reduce.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import (                                            # noqa: E402
    Circuit,
    CircuitError,
    CircuitUnsatisfiable,
    Component,
)
from app.solver.dc_reduce import reduce_to_dc                          # noqa: E402
from app.solver.ngspice import ngspice_method, probe_availability      # noqa: E402
from app.solver.reconcile import format_text_report, run_all           # noqa: E402

FAILS = 0
SKIPS = 0


def C(ref, kind, a, b, value=None):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value)


def eq(label, got, want) -> None:
    global FAILS
    ok = got == want
    if not ok:
        FAILS += 1
    print(f"  [{'通过' if ok else '失败'}] {label}: 期望 {want}  实得 {got}")


def truth(label, cond: bool, detail: str = "") -> None:
    global FAILS
    if not cond:
        FAILS += 1
    print(f"  [{'通过' if cond else '失败'}] {label}"
          + (f"  —— {detail}" if detail else ""))


def banner(title: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72)


def _short(refs_and_cur):
    return [(e["ref"], e.get("current")) for e in refs_and_cur]


# ---------------------------------------------------------------- 用例


def case_c_open():
    """C 开路：V1=12V, R1=4Ω(1-2), R2=8Ω(2-0), C1=10µF(2-0)。

    手算（C 在直流稳态下开路，i_C = 0，这条支路断开）：
        节点 2： (V2 − 12)/4 + V2/8 = 0  →  ×8： 2(V2 − 12) + V2 = 0
                 →  3V2 = 24  →  V2 = 8 V
        i(R1) 参考方向 1→2 = (12 − 8)/4 = 1 A
        i(R2) 参考方向 2→0 = 8/8 = 1 A
        i(V1) 参考方向 0→1 = 1 A（V1 供电）
        功率核对：R1 = 4·1² = 4 W，R2 = 8·1²·... 注意 R2 = 8 Ω → 8·1² = 8 W，
                  合计消耗 12 W；V1 提供 12·1 = 12 W ✓ 守恒
    """
    banner("C 开路：电容支路应被断开，且必须在报告里留痕")
    c = Circuit(name="C开路", ref_node="0", components=[
        C("V1", "V", "1", "0", 12.0),
        C("R1", "R", "1", "2", 4.0),
        C("R2", "R", "2", "0", 8.0),
        C("C1", "C", "2", "0", 1e-5),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]

    eq("V(2)", m["node_voltages"].get("2"), "8")
    eq("i(R1)", m["branch_currents"].get("R1"), "1")
    eq("i(R2)", m["branch_currents"].get("R2"), "1")
    eq("i(V1)", m["branch_currents"].get("V1"), "1")
    eq("C1 被列入 opened", [e["ref"] for e in red["opened"]], ["C1"])
    eq("shorted 应为空", _short(red["shorted"]), [])
    eq("merged_groups 应为空", red["merged_groups"], [])
    eq("isolated_nodes 应为空", red["isolated_nodes"], [])
    truth("摘要写明是电容开路", "电容视为开路" in red["summary"]
          and "C1" in red["summary"], red["summary"])
    truth("化简后电路里不再有 C/L",
          all(x["kind"] in ("R", "V", "I")
              for x in pack["circuit_after_reduction"]["components"]))
    # ★ 回归：含 C 的电路曾让 format_text_report 抛 StopIteration → /api/solve HTTP 500
    txt = format_text_report(c, pack)
    truth("文本报告可生成（回归：曾抛 StopIteration → HTTP 500）", len(txt.splitlines()) > 20)
    truth("文本报告里 C1 被标为『化简后移除』", "直流化简后移除" in txt)
    truth("主解一节不含 i(C1)", "  i(C1) = " not in txt)


def case_l_series():
    """L 串联：V1=10V, R1=5Ω(1-2), L1(2-3), R2=5Ω(3-0)。

    手算（L 在直流稳态下短路，节点 2 与 3 等电位）：
        总电阻 = 5 + 5 = 10 Ω  →  回路电流 10/10 = 1 A
        V(2) = V(3) = 10 − 5·1 = 5 V
        i(R1) 参考方向 1→2 = (10 − 5)/5 = 1 A
        i(R2) 参考方向 3→0 = (5 − 0)/5 = 1 A
        i(L1) 参考方向 2→3 = 1 A（L1 与 R1、R2 串联，电流相同）
        i(V1) 参考方向 0→1 = 1 A
    """
    banner("L 串联：节点合并 + 被移除支路的电流必须反算出来（i(L1) 正是学生要的答案）")
    c = Circuit(name="L串联", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("L1", "L", "2", "3", 1e-3),
        C("R2", "R", "3", "0", 5.0),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]

    eq("V(2)", m["node_voltages"].get("2"), "5")
    eq("i(R1)", m["branch_currents"].get("R1"), "1")
    eq("i(R2)", m["branch_currents"].get("R2"), "1")
    eq("i(V1)", m["branch_currents"].get("V1"), "1")
    eq("merged_groups（代表元在前）", red["merged_groups"], [["2", "3"]])
    # ★ 回归缺陷③：被合并掉的节点名曾被误报成"孤立节点"
    eq("merged_away", red["merged_away"], ["3"])
    eq("isolated_nodes 必须为空（3 不是孤立，它只是与 2 等电位）",
       red["isolated_nodes"], [])
    # ★ 反算出的 i(L1)
    eq("i(L1)（KCL 反算）", _short(red["shorted"]), [("L1", "1")])
    # ★ 回归缺陷②：没有电容的电路曾输出"电容视为开路，移除 2 条支路（L1, R1）"
    truth("摘要不得出现『电容视为开路』（本电路没有电容）",
          "电容视为开路" not in red["summary"], red["summary"])
    truth("摘要必须交代『被合并掉的节点名』与『它仍然存在』",
          "被合并掉的节点名" in red["summary"] and "仍然存在" in red["summary"])
    txt = format_text_report(c, pack)
    truth("文本报告可生成（回归：曾抛 StopIteration → HTTP 500）", len(txt.splitlines()) > 20)
    truth("文本报告写明 i(L1) = 1", "i(L1) = 1" in txt)
    return c, Fraction(1)


def case_v_shorted():
    """★ 核心缺陷：电感把理想电压源短路。

    电路：V1=10V(1-0)、L1(1-0)、R1=5Ω(1-2)、R2=5Ω(2-0)。
    直流下 L1 是理想短路 → 同时要求 V(1) − V(0) = 10 V 与 V(1) = V(0)，
    约束集自相矛盾 → **该电路无解**。

    第三方独立复核（ngspice）：
        Warning: singular matrix:  check node l1#branch
        Note: Dynamic gmin stepping failed / True gmin stepping failed /
              source stepping failed  →  NgSpiceCommandError: Command 'run' failed

    旧版行为：把 V1 当"被短接即消失"删掉 → 三条路径一致给出全零解 →
    ΣP = 0 平凡成立 → overall_pass = True，并输出"读图与建模可以采信"。
    """
    banner("★ V1 被电感短路：必须报『无解』，绝不许删掉电压源后报『通过』")
    c = Circuit(name="V被电感短路", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("L1", "L", "1", "0", 1e-3),
        C("R1", "R", "1", "2", 5.0),
        C("R2", "R", "2", "0", 5.0),
    ])
    truth("CircuitUnsatisfiable 是 CircuitError 的子类（API 的 except 分支要能接住）",
          issubclass(CircuitUnsatisfiable, CircuitError))

    # ---- 1) 直接调化简层
    try:
        reduce_to_dc(c)
        truth("reduce_to_dc 抛 CircuitUnsatisfiable", False, "它竟然正常返回了")
    except CircuitUnsatisfiable as e:
        eq("矛盾条数", len(e.contradictions), 1)
        eq("矛盾指向 V1", e.contradictions[0]["ref"], "V1")
        eq("矛盾类型", e.contradictions[0]["kind"], "ideal_voltage_source_shorted")
        truth("错误信息说明是『无解』而不是『算不出来』", "无解" in str(e))
        truth("错误信息点名 V1", "V1" in str(e))
        truth("错误信息给出可操作建议（线圈电阻 / 暂态题）",
              "电阻" in str(e) and "暂态" in str(e))

    # ---- 2) 端到端：run_all 不许产出一个"通过"的对账包
    try:
        pack = run_all(c)
        truth("run_all 抛 CircuitUnsatisfiable", False,
              f"它竟然返回了 overall_pass={pack.get('overall_pass')}，"
              f"conclusion={pack.get('conclusion')!r}")
    except CircuitUnsatisfiable as e:
        truth("run_all 抛 CircuitUnsatisfiable（未产出误导性的『通过』）", True,
              str(e).splitlines()[0])
    except CircuitError as e:
        truth("run_all 抛 CircuitUnsatisfiable", False, f"抛的是别的 CircuitError：{e}")


def case_v_zero_ok():
    """0 V 电压源（导线）被电感短路 —— 这是**一致**的，绝不该报无解。

    V0=0V(1-0)、L1(1-0)、R1=5Ω(1-2)、R2=5Ω(2-0)。
    L1 与 V0 都使 1 ≡ 0；V0 的支路方程是 0 = 0，与短路**不矛盾**
    （对比 10 V 的电压源：0 = 10 才矛盾）。

    手算：
        V(1) = V(0) = 0 V（被 0 V 源与短路共同钉住）
        节点 2 只经 R1、R2 接到地，且没有任何电流注入
            → V(2) = 0 V，i(R1) = i(R2) = 0
        i(V0) 与 i(L1)：两条支路都并联在节点 1-0 之间，都是理想 0 Ω
            → 节点 1 的 KCL 只给出 i(V0) − i(L1) = 0（与其余部分无关）
            → 一条方程、两个未知量 → **电流不唯一**，必须标"待定"而不是猜 0

    这一例同时压住两个分支：**0 V 源不算矛盾**、**真欠定时不许猜**。
    """
    banner("0 V 电压源被短路：一致 → 不许误报无解；但两支理想短线并联 → 电流确实不唯一")
    c = Circuit(name="0V源被短路", ref_node="0", components=[
        C("V0", "V", "1", "0", 0.0),
        C("L1", "L", "1", "0", 1e-3),
        C("R1", "R", "1", "2", 5.0),
        C("R2", "R", "2", "0", 5.0),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]
    eq("merged_groups（0 当地，不改名）", red["merged_groups"], [["0", "1"]])
    eq("merged_away", red["merged_away"], ["1"])
    eq("V0 不得被列为矛盾", red["contradictions"], [])
    truth("V0 被列为『短接但一致』", any(e["ref"] == "V0" for e in red["shorted"]))
    eq("V0 与 L1 的电流都待定",
       sorted(_short(red["shorted"])), [("L1", None), ("V0", None)])
    eq("V(1)", m["node_voltages"].get("0"), "0")
    eq("i(R1)", m["branch_currents"].get("R1"), "0")
    eq("i(R2)", m["branch_currents"].get("R2"), "0")
    truth("报告写明 i 待定", "待定" in red["summary"], red["summary"])
    truth("摘要点名『纯理想 0 Ω 回路』或『不唯一』",
          "0 Ω" in red["summary"] or "不唯一" in red["summary"], red["summary"])
    truth("每条待定的支路都写了『为什么不定』",
          all(e.get("current_note") for e in red["indeterminate"]))
    truth("明确说明这不是算错", "不是算错" in " ".join(red["notes"]))


def case_isource_l():
    """电流源 + 电感：I1=2A(1-0)、L1(1-2)、R1=10Ω(2-0)。

    手算（L1 短路使 1 ≡ 2）：
        节点 1（含节点 2）： I1 的参考方向是 1→0，i(I1) = +2，
            即 2 A 从节点 1 经电流源流向节点 0（离开节点 1）
            KCL：i(R1)（离开节点 1，参考方向 1→0）+ 2 = 0  →  i(R1) = −2 A
        于是 V(1) = 0 + R1·i(R1) = 10·(−2) = −20 V
        i(I1) = 2 A（理想电流源的定义）
        i(L1) 参考方向 1→2（即 1→1 合并后仍在原节点 1 与 2 之间）：
            节点 1 KCL： 2 + i(L1) = 0  →  i(L1) = −2 A
    """
    banner("电流源 + 电感：反算要能处理负号与『电流源电流由定义给定』")
    c = Circuit(name="I+L", ref_node="0", components=[
        C("I1", "I", "1", "0", 2.0),
        C("L1", "L", "1", "2", 1e-3),
        C("R1", "R", "2", "0", 10.0),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]
    eq("V(1)", m["node_voltages"].get("1"), "-20")
    eq("i(R1)", m["branch_currents"].get("R1"), "-2")
    eq("i(I1)", m["branch_currents"].get("I1"), "2")
    eq("i(L1)（KCL 反算）", _short(red["shorted"]), [("L1", "-2")])


def case_r_parallel_l():
    """R 与 L 并联：V1=10V(1-0)、R1=5Ω(1-2)、L1(1-2)、R2=5Ω(2-0)。

    手算（L1 短路使 1 ≡ 2）：
        V(1) = 10 V（V1 直接定住）
        i(R2) 参考方向 2→0 = 10/5 = 2 A
        i(V1) 参考方向 0→1 = 2 A
        i(R1)：两端等电位 → u = 0 → i = 0（**元件方程**给出的确定值，
            不是"因为被删掉了所以没有"）
        i(L1)：节点 2 的 KCL： 由 R2 流入 2 A，全部经 L1 流走
            → x + (−2) = 0 形式 → i(L1) = 2 A
        功率核对：V1 提供 10·2 = 20 W；R2 消耗 5·2² = 20 W；R1 = 0；L1 压降 0 → 0
                  ΣP = 0 ✓ 守恒
    """
    banner("R ∥ L：R 的 i=0 来自元件方程，L 的 i 来自 KCL 反算，两者来源不同")
    c = Circuit(name="R并L", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("L1", "L", "1", "2", 1e-3),
        C("R2", "R", "2", "0", 5.0),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]
    eq("V(1)", m["node_voltages"].get("1"), "10")
    eq("i(R2)", m["branch_currents"].get("R2"), "2")
    eq("i(V1)", m["branch_currents"].get("V1"), "2")
    eq("shorted 明细（R1 由元件方程、L1 由 KCL）",
       sorted(_short(red["shorted"])), [("L1", "2"), ("R1", "0")])
    by = {e["ref"]: e for e in red["shorted"]}
    truth("i(R1)=0 的依据写明是元件方程 u = R·i",
          "元件方程" in (by["R1"].get("current_method") or ""),
          by["R1"].get("current_method", ""))
    truth("i(L1)=2 的依据写明是 KCL 反算",
          "KCL" in (by["L1"].get("current_method") or ""),
          by["L1"].get("current_method", ""))
    pw = pack["verifications"]["mna"]["checks"][0]
    truth("化简后电路功率守恒", pw["ok"], pw["conclusion"])
    truth("总判定通过", pack["overall_pass"])


def case_isolated_by_c():
    """真正的孤立节点：V1=10V(1-0)、R1=5Ω(1-2)、C1=1µF(2-3)。

    C1 开路后节点 3 不再属于任何支路 → 它是**真孤立**。
    节点 2 只剩 R1，KCL 要求 (V2 − V1)/5 = 0 → V2 = V1 = 10 V，
    i(R1) = 0，i(V1) = 0。

    这正是"隔直电容"的物理结论，报告要把它**说成结论**而不是"图不连通"的错。
    """
    banner("真·隔直电容造成的孤立节点：要作为正确结论报出来")
    c = Circuit(name="隔直", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("C1", "C", "2", "3", 1e-6),
    ])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    red = pack["reduction"]
    eq("isolated_nodes", red["isolated_nodes"], ["3"])
    eq("V(2)", m["node_voltages"].get("2"), "10")
    eq("i(R1)", m["branch_currents"].get("R1"), "0")
    reduced_nodes = [
        x for c2 in pack["circuit_after_reduction"]["components"] for x in c2["nodes"]
    ]
    truth("节点 3 不在化简后电路里（它已不接任何支路）",
          "3" not in reduced_nodes, str(reduced_nodes))
    truth("摘要把它解释成『只剩电容相连、直流无电流』的正确结论",
          "孤立节点" in red["summary"] and "正确结论" in red["summary"],
          red["summary"])
    txt = format_text_report(c, pack)
    truth("文本报告可生成", len(txt.splitlines()) > 20)


def case_underdetermined():
    """两个电感并联 → 纯理想 0 Ω 回路，电流不唯一。

    V1=10V(1-0)、R1=5Ω(1-2)、L1(2-3)、L2(2-3)。
    L1、L2 都使 2 ≡ 3，两条支路构成一个纯理想 0 Ω 回路。
    化简后只剩 V1 与 R1（节点 2 悬空）→ V(2) = 10 V，i(R1) = 0。
    KCL 只能给出 x1 + x2 = 0 一条独立方程（两个未知量）→ **欠定**。
    物理上就是：L1–L2 回路里的环流可以是任意值。
    """
    banner("纯理想 0 Ω 回路：欠定要明说『不唯一』，不许猜一个数")
    c = Circuit(name="双L", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("L1", "L", "2", "3", 1e-3),
        C("L2", "L", "2", "3", 1e-3),
    ])
    pack = run_all(c)
    red = pack["reduction"]
    eq("两条电感都被移除", sorted(e["ref"] for e in red["shorted"]), ["L1", "L2"])
    eq("两条都标为电流待定", _short(red["shorted"]), [("L1", None), ("L2", None)])
    eq("indeterminate 列出两条", sorted(e["ref"] for e in red["indeterminate"]),
       ["L1", "L2"])
    truth("摘要说明不唯一的原因是纯理想 0 Ω 回路",
          "不唯一" in red["summary"], red["summary"])
    truth("每条待定的支路都写了『为什么不定』",
          all(e.get("current_note") for e in red["indeterminate"]))
    txt = format_text_report(c, pack)
    truth("文本报告可生成且写明 i 待定", "待定" in txt)


def case_ref_node_kept():
    """参考节点在节点合并后必须**仍然是 0**，不能被改名。

    V1=10V(1-0)、R1=5Ω(1-2)、L1(2-0)。
    L1 使 2 ≡ 0。若并查集让代表元取到 "2"，地就被改名成节点 2，
    报告里"节点 2"其实是地 —— 极易误读。

    手算：化简后 V1(1-0) 与 R1(1-0 重写自 1-2) 并联，
        V(1) = 10 V，i(R1) 参考方向 1→0 = 10/5 = 2 A，i(V1) = 2 A
        i(L1) 参考方向 2→0：节点 2 的 KCL，R1 经原节点 2 流入 2 A
            → i(L1) = 2 A
    """
    banner("节点合并后参考节点不得改名")
    c = Circuit(name="地不改名", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("L1", "L", "2", "0", 1e-3),
    ])
    red, rep = reduce_to_dc(c)
    eq("化简后 ref_node 仍是 '0'", red.ref_node, "0")
    truth("化简后电路中确实还有节点 '0'", "0" in red.nodes, str(red.nodes))
    eq("合并组以 0 为代表元", rep.merged_groups, [["0", "2"]])
    eq("merged_away", rep.merged_away, ["2"])
    pack = run_all(c)
    m = pack["solutions"]["mna"]
    eq("V(1)", m["node_voltages"].get("1"), "10")
    eq("i(R1)", m["branch_currents"].get("R1"), "2")
    eq("i(L1)（KCL 反算）", _short(pack["reduction"]["shorted"]), [("L1", "2")])


def case_no_mutation():
    """化简**不得改动传入的原电路**（它是只读输入）。"""
    banner("化简是纯函数：不得改动传入的原电路")
    c = Circuit(name="只读", ref_node="0", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("L1", "L", "2", "3", 1e-3),
        C("R2", "R", "3", "0", 5.0),
    ])
    before = c.to_dict()
    reduce_to_dc(c)
    truth("原电路逐字段未被改动", c.to_dict() == before)
    truth("原电路仍含电感", any(x.kind == "L" for x in c.components))
    truth("原电路节点名仍是 0/1/2/3", c.nodes == ["0", "1", "2", "3"], str(c.nodes))


# ---------------------------------------------------------------- 第三方独立复核


def ngspice_inductor_current(circuit: Circuit, l_ref: str) -> float | None:
    """请 ngspice 给出电感支路电流 —— **第三方独立实现**的复核，不是自己再算一遍。

    做法：把该电感替换成 **0 V 理想电压源**（"电流探针"）。这个替换是**等价**的，
    不是近似：直流稳态下电感的支路方程就是 u_L = 0，与 0 V 电压源完全一样。
    于是 ngspice 报出的 ``i(Vsense)`` 就是 ``i_L``。

    ★ 符号：电压源的 IR 参考方向是"内部 − → +"，即 ``declared_direction`` 给出
    ``(nodes[1], nodes[0])``，与电感（``(nodes[0], nodes[1])``）相反 → 取负。
    """
    probe = circuit.copy()
    probe.components = [
        Component(ref="Vsense", kind="V", nodes=cc.nodes, value=0.0,
                  note="把电感换成 0V 源以取该支路电流（直流下两者等价）")
        if cc.ref == l_ref else cc
        for cc in probe.components
    ]
    sol = ngspice_method(probe)
    v = sol.get_current("Vsense")
    return None if v is None else -float(v)


def main() -> int:
    global FAILS, SKIPS

    case_c_open()
    c_l, want_i_l = case_l_series()
    case_v_shorted()
    case_v_zero_ok()
    case_isource_l()
    case_r_parallel_l()
    case_isolated_by_c()
    case_underdetermined()
    case_ref_node_kept()
    case_no_mutation()

    # ---- 第三方独立复核：i(L1) 由 ngspice 给出，不依赖本仓库任何一行求解代码
    banner("第三方独立复核：把 L1 换成 0V 电压源，请 ngspice 给出 i(L1)")
    avail = probe_availability()
    if not avail.get("available"):
        SKIPS += 1
        print(f"  [未执行] ngspice 不可用：{avail.get('reason') or avail} —— "
              "本项**不算通过**，也不许当成通过")
    else:
        float_l = ngspice_inductor_current(c_l, "L1")
        if float_l is None:
            SKIPS += 1
            print("  [未执行] ngspice 未返回 Vsense 的支路电流")
        else:
            d = abs(float_l - float(want_i_l))
            ok = d <= 1e-9
            if not ok:
                FAILS += 1
            print(f"  [{'通过' if ok else '失败'}] ngspice 给的 i(L1) = {float_l:.10g}，"
                  f"本项目 KCL 反算 = {want_i_l} = {float(want_i_l):.10g}，差 {d:.3g}")
            print("       （两条独立实现给出同一个数 —— 反算不是自证）")

    print("\n" + "=" * 72)
    print(f"总计失败项：{FAILS}"
          + (f"    未执行项：{SKIPS}（环境依赖，未算作通过）" if SKIPS else ""))
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
