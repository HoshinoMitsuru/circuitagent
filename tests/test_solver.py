"""三法对账的实跑自检。

这几张电路都是能手算验算的，用来确认符号约定、矩阵构造、ngspice 取值全对。
**每张电路的期望值都是独立手算出来的**，不是从程序输出里抄的 ——
否则这个测试就变成"自己验自己"。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_solver.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import Circuit, Component, Evidence, CircuitError  # noqa: E402
from app.solver.reconcile import run_all, format_text_report          # noqa: E402
from app.solver.equivalence import thevenin_norton                    # noqa: E402


def C(ref, kind, a, b, value, detail=""):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value,
                     evidence=Evidence(source="manual", confidence=1.0, detail=detail))


# ---------------------------------------------------------------- 电路集


def case_bridge() -> tuple[Circuit, dict]:
    """电桥。手算验算（节点电压法，已知 V1 为唯一源）：

        V1 = 12V 加在节点 1/0。节点 2 是 R1-R2-R5 交汇，节点 3 是 R3-R4-R5 交汇。
        R1=100(1-2), R2=200(2-0), R3=150(1-3), R4=300(3-0), R5=50(2-3)

        节点 2: (V2-12)/100 + V2/200 + (V2-V3)/50 = 0
        节点 3: (V3-12)/150 + V3/300 + (V3-V2)/50 = 0

        第一式乘 200:  2(V2-12) + V2 + 4(V2-V3) = 0  ->  7V2 - 4V3 = 24
        第二式乘 300:  2(V3-12) + V3 + 6(V3-V2) = 0   ->  -6V2 + 9V3 = 24

        解:  7V2 - 4V3 = 24
            -6V2 + 9V3 = 24
        第一式 *9:  63V2 - 36V3 = 216
        第二式 *4: -24V2 + 36V3 = 96
        相加:        39V2 = 312  ->  V2 = 8
        代入:        7*8 - 4V3 = 24  ->  56 - 4V3 = 24  ->  V3 = 8

        所以 V2 = V3 = 8V，R5 两端电压差为 0，电桥平衡，i(R5) = 0。
        检验平衡条件: R1/R3 = 100/150 = 2/3, R2/R4 = 200/300 = 2/3  ✓ 平衡
    """
    c = Circuit(name="电桥（平衡）", components=[
        C("V1", "V", "1", "0", 12.0),
        C("R1", "R", "1", "2", 100.0),
        C("R2", "R", "2", "0", 200.0),
        C("R3", "R", "1", "3", 150.0),
        C("R4", "R", "3", "0", 300.0),
        C("R5", "R", "2", "3", 50.0),
    ])
    return c, {"V(2)": Fraction(8), "V(3)": Fraction(8), "i(R5)": Fraction(0)}


def case_current_source() -> tuple[Circuit, dict]:
    """含电流源，检验"电流源电流已知、电压为未知量"这条路。

        V1=10V(1-0), R1=5Ω(1-2), I1=2A(2-0)
        节点2: (V2-10)/5 + 2 = 0  ->  V2 = 0
        i(R1) 参考方向 1->2 = (10-0)/5 = 2A
        i(R1) 供给节点2的电流 2A，被 I1 全部吸走 ✓
        功率: R1 吸收 = 5*2^2 = 20W；V1 提供 10V*2A = 20W；I1 两端压差 0 -> 0W
    """
    c = Circuit(name="含电流源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 5.0),
        C("I1", "I", "2", "0", 2.0),
    ])
    return c, {"V(2)": Fraction(0), "i(R1)": Fraction(2)}


def case_two_sources() -> tuple[Circuit, dict]:
    """两个电压源 + 三个电阻，串并联手算。

        V1=12V(1-0), V2=6V(2-0)（2 为 + 端）, R1=2Ω(1-3), R2=4Ω(2-3), R3=3Ω(3-0)
        节点3: (V3-12)/2 + (V3-6)/4 + V3/3 = 0
        乘12:  6(V3-12) + 3(V3-6) + 4V3 = 0
               6V3 - 72 + 3V3 - 18 + 4V3 = 0
               13V3 = 90  ->  V3 = 90/13

        V3 = 90/13 ≈ 6.9231
    """
    c = Circuit(name="双电源", components=[
        C("V1", "V", "1", "0", 12.0),
        C("V2", "V", "2", "0", 6.0),
        C("R1", "R", "1", "3", 2.0),
        C("R2", "R", "2", "3", 4.0),
        C("R3", "R", "3", "0", 3.0),
    ])
    return c, {"V(3)": Fraction(90, 13)}


def case_series_vsources() -> tuple[Circuit, dict]:
    """两个电压源串联夹一个节点 —— 技能文档里点名"未知量可减少"的那种结构。

        V1=10V(1-0), V2=4V(2-1)（2 为 +）, R1=6Ω(2-0)
        V2 夹在 1、2 之间: V2 - V1 = 4  ->  V2 = 14V
        i(R1) 参考方向 2->0 = 14/6 = 7/3 A
        检验: V1 电流 = ? 节点1 KCL: 流出节点1 经 V1 = -i(V1)；无其他支路
              只有 V1 与 V2 挂在节点1上, 所以 i(V1) = -i(V2串)
        手算电流: 回路 2->0(V1 支路... ) 直接用 KVL: 14V 加在 6Ω 上 -> 7/3 A
    """
    c = Circuit(name="电压源串联", components=[
        C("V1", "V", "1", "0", 10.0),
        C("V2", "V", "2", "1", 4.0),      # + 端在 2, - 端在 1
        C("R1", "R", "2", "0", 6.0),
    ])
    return c, {"V(1)": Fraction(10), "V(2)": Fraction(14),
               "i(R1)": Fraction(7, 3)}


def case_textbook_4_17() -> tuple[Circuit, dict]:
    """教材题 4-17：两个 125V 电压源串联供电，6 节点 / 8 支路。

        拓扑（节点名照抄独立解析出的网表，便于跨工具对账）：
          A ─[R1  1Ω]─ T          B ─[R2  2Ω]─ M          C ─[R3  1Ω]─ D   （C 为参考点）
          T ─[R6  6Ω]─ M          M ─[R12 12Ω]─ D        T ─[R24 24Ω]─ D
          V1 = 125V 加在 A、B 之间（A 为 +）；V2 = 125V 加在 B、C 之间（B 为 +）

        手算（节点电压法；三式各自乘最小公倍数把分母消掉）：
          V1 与 V2 串联且 B 是其中点  →  V_A = 250V，V_B = 125V，V_C = 0

          节点 T：(V_T−250)/1 + (V_T−V_M)/6 + (V_T−V_D)/24 = 0
          节点 M：(V_M−V_T)/6 + (V_M−125)/2 + (V_M−V_D)/12 = 0
          节点 D：(V_D−V_T)/24 + (V_D−V_M)/12 + V_D/1 = 0

          三式分别乘 24 / 12 / 24：
            (1)  29V_T −  4V_M −  V_D = 6000
            (2)  −2V_T +  9V_M −  V_D =  750
            (3)   −V_T −  2V_M + 27V_D = 0

          由 (3) 得 V_T = 27V_D − 2V_M，代入 (1)(2)：
            29(27V_D − 2V_M) − 4V_M − V_D = 6000  →   782V_D − 62V_M = 6000
           −2(27V_D − 2V_M) + 9V_M − V_D =  750  →   −55V_D + 13V_M =  750

          由第二式 V_M = (750 + 55V_D)/13，代入第一式并两边乘 13：
            10166V_D − 62(750 + 55V_D) = 78000
            10166V_D − 46500 − 3410V_D = 78000
            6756V_D = 124500
            V_D = 124500/6756 = 10375/563          （分子分母同除 12）

          回代：
            V_M = (750 + 55·10375/563)/13 = (422250 + 570625)/7319
                = 992875/7319 = 76375/563          （同除 13）
            V_T = 27·10375/563 − 2·76375/563
                = (280125 − 152750)/563 = 127375/563

          支路电流（全部沿网表给定的参考方向）：
            i(R1)  = A→T  = (250 − 127375/563)/1      = 13375/563
            i(R2)  = M→B  = (76375/563 − 125)/2       =  3000/563
            i(R3)  = D→C  = (10375/563)/1             = 10375/563
            i(R6)  = T→M  = (127375 − 76375)/563/6    =  8500/563
            i(R12) = M→D  = (76375 − 10375)/563/12    =  5500/563
            i(R24) = T→D  = (127375 − 10375)/563/24   =  4875/563

          电压源电流（按 IR 约定 i > 0 表示该电源供电）：
            i(V1) = i(R1) = 13375/563        （节点 A 上只挂 V1 与 R1）
            节点 B 的 KCL 反查 V2：流入 B 的有 R2 的 3000/563 与 V2，
            流出 B 的只有 V1 的 13375/563，故 i(V2) = 13375/563 − 3000/563
                                                   = 10375/563

        ★ 这道题的价值：6 节点 / 8 支路，是本套用例里规模最大的精确有理数算例，
          含两个串联电压源（技能文档点名的"未知量可减少"结构）。
          它同时被 SharedUmbrella 项目那份**独立写的实现**复算过，两边逐项严格相等 ——
          见 tools/crosscheck_sharedumbrella.py（跨工具对账，与三法互校互补：
          三法共用同一份 IR，拦不住"网表抄错"；换实现喂同一份网表才拦得住）。
    """
    c = Circuit(name="教材题 4-17（双电源串联）", components=[
        C("V1", "V", "A", "B", 125.0),
        C("V2", "V", "B", "C", 125.0),
        C("R1", "R", "A", "T", 1.0),
        C("R2", "R", "M", "B", 2.0),
        C("R3", "R", "D", "C", 1.0),
        C("R6", "R", "T", "M", 6.0),
        C("R12", "R", "M", "D", 12.0),
        C("R24", "R", "T", "D", 24.0),
    ], ref_node="C")
    return c, {
        "V(A)": Fraction(250), "V(B)": Fraction(125), "V(C)": Fraction(0),
        "V(D)": Fraction(10375, 563),
        "V(M)": Fraction(76375, 563),
        "V(T)": Fraction(127375, 563),
        "i(R1)": Fraction(13375, 563),
        "i(R2)": Fraction(3000, 563),
        "i(R3)": Fraction(10375, 563),
        "i(R6)": Fraction(8500, 563),
        "i(R12)": Fraction(5500, 563),
        "i(R24)": Fraction(4875, 563),
        "i(V1)": Fraction(13375, 563),
        "i(V2)": Fraction(10375, 563),
    }


CASES = [
    ("电桥（平衡）", case_bridge),
    ("含电流源", case_current_source),
    ("双电源", case_two_sources),
    ("电压源串联", case_series_vsources),
    ("教材题 4-17（双电源串联 / 6 节点 8 支路）", case_textbook_4_17),
]


# ---------------------------------------------------------------- 跑


def main() -> int:
    fails = 0
    for name, fn in CASES:
        circ, expect = fn()
        print("\n" + "#" * 72)
        print(f"# 用例：{name}")
        print("#" * 72)
        try:
            pack = run_all(circ)
        except CircuitError as e:
            print(f"[失败] 求解抛异常：{e}")
            fails += 1
            continue

        got = pack["solutions"]["mna"]["node_voltages"]
        branches = pack["solutions"]["mna"]["branch_currents"]

        for key, want in expect.items():
            if key.startswith("V("):
                node = key[2:-1]
                raw = got.get(node)
            else:
                ref = key[2:-1]
                raw = branches.get(ref)
            if raw is None:
                print(f"  [失败] 期望 {key} = {want}，但解里没有这一项")
                fails += 1
                continue
            actual = Fraction(raw) if not isinstance(raw, Fraction) else raw
            ok = actual == want
            if not ok:
                fails += 1
            print(f"  [{'通过' if ok else '失败'}] {key}: 期望 {want}  实得 {actual}")

        print(f"  总判定：{'通过' if pack['overall_pass'] else '未通过'}")
        for k, why in pack["failures"].items():
            print(f"    · {k} 路缺失：{why}")
        # 精炼输出
        pw = pack["verifications"]["mna"]["checks"][0]
        print(f"  功率：{pw['conclusion']}")
        print(f"  精确互校：{pack['exact_crosscheck']['detail']}")
        sp = pack["superposition"]
        if sp.get("applicable") and sp.get("rows"):
            print(f"  叠加复核：{sp['conclusion']}")
        # 偏差必须**分方法**报，不能取全局最大值 —— 否则一法出错会显得像另一法出错
        # （这个测试脚本自己就犯过一次：把支路电流法的 19.2V 偏差标成了 ngspice 的）
        for k, label in (("branch", "支路电流法"), ("ngspice", "ngspice")):
            rows_k = [r for r in pack["node_table"] if r.get(k) is not None]
            if not rows_k:
                print(f"  {label} 未参与对账")
                continue
            worst = max(abs(r[k + "_float"] - r["exact_float"]) for r in rows_k)
            print(f"  与{label}最大节点电压偏差：{worst:.3g} V")

    # ---- 戴维南等效单测
    print("\n" + "#" * 72)
    print("# 用例：戴维南等效（端口 2-0）")
    print("#" * 72)
    circ, _ = case_bridge()
    try:
        th = thevenin_norton(circ, "2", "0")
        # 手算：把 V1 置零（短路），节点 1 于是直接落到地。
        #   ★ 这里我第一次算错过，写下来当反例：
        #     我漏掉了 R1（节点1→节点2）在节点1接地后，本身就是节点2 的一条 100Ω 对地支路，
        #     误算成 600/7 ≈ 85.71。正确的等效是三条对地路径并联：
        #       · R2 = 200Ω 直通地
        #       · R1 = 100Ω 直通地（因为节点 1 已被短路到地）
        #       · R5 + (R3 ∥ R4) = 50 + (150 ∥ 300) = 50 + 100 = 150Ω
        #     Req = 100 ∥ 200 ∥ 150
        #         100∥200 = 200/3
        #         200/3 ∥ 150 = (200/3 · 150) / (200/3 + 150) = 10000 / (650/3) = 600/13
        want_req = Fraction(600, 13)
        got_req = Fraction(th["req"].split("/")[0]) / Fraction(th["req"].split("/")[1]) \
            if "/" in str(th["req"]) else Fraction(th["req"])
        ok = got_req == want_req
        print(f"  [{'通过' if ok else '失败'}] Req: 期望 {want_req} = {float(want_req):.6f}  "
              f"实得 {got_req} = {float(got_req):.6f}")
        if not ok:
            fails += 1
        print(f"  Voc = {th['voc']} V   Isc = {th['isc']} A")

        # ---- 独立复核 Isc：真接一根短路线（0V 电压源）看流过的电流，
        #      而不是用 Isc = Voc/Req 这个公式自证（那是自己验自己）
        circ_sc = case_bridge()[0]
        circ_sc.components.append(C("Vsc", "V", "2", "0", 0.0,
                                    "短路支路，用于独立复核 Isc"))
        from app.solver.mna import node_voltage_method
        sc = node_voltage_method(circ_sc)
        # ★ 符号方向要当心（我第一次也写反了）：
        #   Vsc 的 IR 电流参考方向是 `declared_direction` 给出的 "0 -> 2"
        #   （电压源规定为"源内部由 − 流向 +"），也就是**从节点 0 流向节点 2**。
        #   而 Isc 是"经短路线从节点 2 流向节点 0"，方向正好相反，所以要取负。
        #   物理直觉核对：Voc = V2 − V0 = +8V，戴维南源必然把电流从 + 端（节点2）
        #   推出去，所以 Isc 应为正。
        isc_meas = -sc.get_current("Vsc")
        # 顺带用手算再核一遍（注意短路会把节点 3 的电压也改掉，V3 = 8/3 而不是 8）：
        #   节点3: 2(V3−12) + V3 + 6V3 = 0 -> V3 = 8/3
        #   流入节点2: R1 给 12/100 = 9/75；R5 给 (8/3)/50 = 4/75  -> 合计 13/75
        want_isc = Fraction(13, 75)
        ok2 = isc_meas == want_isc
        print(f"  [{'通过' if ok2 else '失败'}] Isc 独立复核（实接短路线）: "
              f"期望 {want_isc} = {float(want_isc):.6f}  实得 {isc_meas} = {float(isc_meas):.6f}")
        if not ok2:
            fails += 1
    except CircuitError as e:
        print(f"  [失败] {e}")
        fails += 1

    print("\n" + "=" * 72)
    print(f"总计失败项：{fails}")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
