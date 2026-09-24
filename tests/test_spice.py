"""SPICE 网表读写的实跑自检。

**为什么单独有这一个文件**：这一层的 bug 属于「写得出、读不回」型 ——
``to_spice()`` 给电压源写 ``DC`` 前缀（有意为之，为了约束 ngspice 的 .op 行为），
而 ``from_spice()`` 只做裸 ``float()``，于是**自家产出的网表自家读不回**，
任何含电压源的电路往返必断。而往返断掉不会输出任何错数字，
它只是让「手工贴网表」这条最省事的入口一直不可用。

所以这里钉死两条契约：
  1. ``from_spice(to_spice(c))`` 必须**逐项无损**（位号/类型/节点/数值全等）；
  2. 读不出来的东西**要么给对、要么给精准的错** —— 不接受笼统的「不是字面数字」，
     因为那会让人以为是自己格式写错，去反复折腾格式，而真问题是别的东西。

数值期望全部**独立手算**，不从程序输出里抄。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_spice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import Circuit, Component, Evidence, CircuitError  # noqa: E402
from app.ir.spice import to_spice, from_spice                         # noqa: E402
from app.solver.reconcile import run_all                              # noqa: E402


def C(ref, kind, a, b, value):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value,
                     evidence=Evidence(source="manual", confidence=1.0, detail="test"))


def sig(circ: Circuit) -> set[tuple]:
    """电路的逐项指纹：位号 / 类型 / 节点 / 数值。"""
    return {(c.ref, c.kind, c.nodes, c.value) for c in circ.components}


def warns_of(circ: Circuit) -> list[str]:
    return [d["text"] for d in circ.diagnostics
            if d.get("kind") == "spice_parse_warning"]


def main() -> int:
    fails = 0

    # ================================================================ 1. 往返无损
    print("=" * 72)
    print("[1] 往返：from_spice(to_spice(c)) 必须逐项无损")
    print("=" * 72)
    circuits = {
        "含电压源（旧代码必断的那一类）": Circuit(name="rt1", ref_node="0", components=[
            C("V1", "V", "1", "0", 12.0),
            C("R1", "R", "1", "2", 100.0),
            C("R2", "R", "2", "0", 200.0),
        ]),
        "含电流源 + 电感电容": Circuit(name="rt2", ref_node="0", components=[
            C("I1", "I", "1", "0", 2.0),
            C("R1", "R", "1", "2", 470.0),
            C("L1", "L", "2", "0", 1e-3),
            C("C1", "C", "2", "0", 1e-6),
        ]),
        "多电压源串联": Circuit(name="rt3", ref_node="0", components=[
            C("V1", "V", "1", "0", 10.0),
            C("V2", "V", "2", "1", 4.0),
            C("R1", "R", "2", "0", 7.0),
        ]),
    }
    for label, circ in circuits.items():
        net = to_spice(circ, title=label)
        try:
            back = from_spice(net)
            ok = sig(back) == sig(circ)
            print(f"  [{'通过' if ok else '失败'}] {label}")
            if not ok:
                fails += 1
                print(f"         原件 {sorted(sig(circ))}")
                print(f"         读回 {sorted(sig(back))}")
        except CircuitError as e:
            fails += 1
            print(f"  [失败] {label} —— 读回抛错：{e}")
            print("         网表内容：")
            for ln in net.splitlines():
                print(f"           {ln}")

    # ================================================================ 2. DC 前缀三态
    print("\n" + "=" * 72)
    print("[2] 直流前缀三态：DC 12 / DC=12 / DC = 12 都应读成 12")
    print("=" * 72)
    for card in ("V1 1 0 DC 12", "V1 1 0 DC=12", "V1 1 0 DC = 12", "V1 1 0 12"):
        try:
            c = from_spice(f"* t\n{card}\n.end\n")
            got = c.components[0].value
            ok = got == 12.0
            print(f"  [{'通过' if ok else '失败'}] {card:<20} -> {got}")
            if not ok:
                fails += 1
        except CircuitError as e:
            fails += 1
            print(f"  [失败] {card:<20} -> 抛错：{e}")

    # ================================================================ 3. 工程记法
    print("\n" + "=" * 72)
    print("[3] 工程记法（期望值独立手算）")
    print("=" * 72)
    #  1k   = 1e3        4k7  = 4.7e3     1meg = 1e6
    #  10u  = 10e-6      100n = 100e-9    2M2  = 2.2e6
    eng = [
        ("R1 1 2 1k",   1000.0),
        ("R1 1 2 4k7",  4700.0),
        ("R1 1 2 1meg", 1000000.0),
        ("C1 1 2 10u",  10e-6),
        ("C1 1 2 100n", 100e-9),
        ("R1 1 2 2M2",  2.2e6),
    ]
    for card, want in eng:
        try:
            c = from_spice(f"* t\n{card}\n.end\n")
            got = c.components[0].value
            ok = got is not None and abs(got - want) <= abs(want) * 1e-12
            print(f"  [{'通过' if ok else '失败'}] {card:<16} 期望 {want:g}  实得 {got:g}")
            if not ok:
                fails += 1
        except CircuitError as e:
            fails += 1
            print(f"  [失败] {card:<16} 期望 {want:g}  但抛错：{e}")

    # ================================================================ 4. 非直流激励
    print("\n" + "=" * 72)
    print("[4] 非直流激励：必须拒绝，且理由要**精准**（不能是「不是字面数字」）")
    print("=" * 72)
    exc = [
        ("V1 1 0 AC 1",                  "AC 幅值"),
        ("V1 1 0 SIN(0 12 50)",          "正弦瞬态"),
        ("V1 1 0 PULSE(0 5 0 1n 1n 1m 2m)", "脉冲瞬态"),
    ]
    for card, label in exc:
        try:
            c = from_spice(f"* t\n{card}\n.end\n")
            fails += 1
            print(f"  [失败] {label:<8} {card:<32} 竟然读通了 -> {c.components[0].value}")
        except CircuitError as e:
            msg = str(e)
            # 三条要求：说清是非直流激励 / 说清本工具只解直流工作点 / 不能是旧那句误导话术
            ok = ("非直流激励" in msg and "直流工作点" in msg
                  and "不是字面数字" not in msg)
            print(f"  [{'通过' if ok else '失败'}] {label:<8} 报错理由精准")
            print(f"           {msg[:104]}…")
            if not ok:
                fails += 1

    # ================================================================ 5. 留痕不静默
    print("\n" + "=" * 72)
    print("[5] 取到直流值但放弃了别的成分时，必须留痕（不静默丢弃）")
    print("=" * 72)
    #  DC 12 AC 1 -> 取 12，但要警告 AC 部分被忽略
    try:
        c = from_spice("* t\nV1 1 0 DC 12 AC 1\n.end\n")
        ws = warns_of(c)
        ok = c.components[0].value == 12.0 and len(ws) == 1 and "非直流激励" in ws[0]
        print(f"  [{'通过' if ok else '失败'}] DC 12 AC 1 -> 12.0 且恰好 1 条警告（不刷屏）")
        for w in ws:
            print(f"           ⚠ {w}")
        if not ok:
            fails += 1
    except CircuitError as e:
        fails += 1
        print(f"  [失败] DC 12 AC 1 抛错：{e}")

    #  1M / 1m 的约定冲突必须一路传到 IR 的 diagnostics ——
    #  三法互校**拦不住**它（三法共用同一份 IR，会一致地给出同一个错答案）
    for card, label in (("R1 1 2 1M", "M"), ("R1 1 2 1m", "m")):
        c = from_spice(f"* t\n{card}\n.end\n")
        ws = warns_of(c)
        ok = any(("SPICE" in w and "KiCad" in w) for w in ws)
        print(f"  [{'通过' if ok else '失败'}] {card:<14} 单位约定冲突警告已转发到 IR")
        if not ok:
            fails += 1

    # ================================================================ 6. 错误信息不误导
    print("\n" + "=" * 72)
    print("[6] 错误话术自检：不能出现「1k 之外的杂式」这种把人往错方向带的说法")
    print("=" * 72)
    #  旧提示语暗示 1k 不能解析（实际能），把人引去改格式而不是查真问题。
    #  这里钉一条：任何报错里都不许再出现这种自我否认的表述。
    probe_cards = ["V1 1 0 AC 1", "V1 1 0 DC", "R1 1 2 {R2}", "R1 1 2 5zz"]
    bad_phrases = ("1k 之外", "不是字面数字")
    for card in probe_cards:
        try:
            from_spice(f"* t\n{card}\n.end\n")
            print(f"  [跳过] {card:<16} 没抛错（{card} 有警告路径）")
            continue
        except CircuitError as e:
            msg = str(e)
            hit = [p for p in bad_phrases if p in msg]
            ok = not hit
            print(f"  [{'通过' if ok else '失败'}] {card:<16} 报错话术无误导")
            if not ok:
                fails += 1
                print(f"           命中误导话术：{hit}")

    # ================================================================ 7. 工程记法进求解器
    print("\n" + "=" * 72)
    print("[7] 工程记法读数要能一路走到求解器（手算：10V/(4.7k+5.3k) = 1/1000 A）")
    print("=" * 72)
    try:
        c = from_spice("* 分压\nV1 1 0 DC 10\nR1 1 2 4k7\nR2 2 0 5k3\n.end\n")
        vals = {x.ref: x.value for x in c.components}
        ok_v = vals == {"V1": 10.0, "R1": 4700.0, "R2": 5300.0}
        print(f"  [{'通过' if ok_v else '失败'}] 读出的数值 {vals}")
        if not ok_v:
            fails += 1

        pack = run_all(c)
        i1 = next(r for r in pack["branch_table"] if r["ref"] == "R1")["exact"]
        #  手算：V1/(R1+R2) = 10/(4700+5300) = 10/10000 = 1/1000
        ok_i = str(i1) in ("1/1000", "0.001")
        print(f"  [{'通过' if ok_i else '失败'}] i(R1) 期望 1/1000  实得 {i1}")
        if not ok_i:
            fails += 1

        ok_all = pack["overall_pass"] and not pack["failures"]
        print(f"  [{'通过' if ok_all else '失败'}] 三法对账总判定 {pack['overall_pass']}  "
              f"failures={pack['failures']}")
        if not ok_all:
            fails += 1
    except CircuitError as e:
        fails += 1
        print(f"  [失败] {e}")

    print("\n" + "=" * 72)
    print(f"总计失败项：{fails}")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
