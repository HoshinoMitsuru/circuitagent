"""KiCad 通道实跑自检。

两级验证：
1. **确定性断言**：手写一份 kicadxml 网表（内容对应已知的电桥题），
   断言解析出的拓扑与数值逐项正确。手写网表的好处是真值由我给定，
   不依赖 kicad-cli 的输出格式是否会变。
2. **冒烟测试**：对 KiCad 自带 demos 批量导出网表并解析，
   统计有多少张能变成"可求解的 IR"。这一级不追求全过（demo 里大量运放、
   晶体管等多端器件本版本不支持），但**必须把排除原因逐条列出来**，
   绝不能静默丢弃。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_ingest_kicad.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingest.kicad_in import (            # noqa: E402
    kicad_netlist_text_to_ir, kicad_sch_to_ir, _find_kicad_cli,
)
from app.solver.reconcile import run_all       # noqa: E402

# ---------------------------------------------------------------- 手写夹具

BRIDGE_NETLIST = """<?xml version="1.0" encoding="UTF-8"?>
<export version="E">
  <components>
    <comp ref="V1"><value>12</value>
      <libsource lib="Simulation_SPICE" part="VDC"/></comp>
    <comp ref="R1"><value>100</value>
      <libsource lib="Device" part="R"/></comp>
    <comp ref="R2"><value>200</value>
      <libsource lib="Device" part="R"/></comp>
    <comp ref="R3"><value>150</value>
      <libsource lib="Device" part="R"/></comp>
    <comp ref="R4"><value>300</value>
      <libsource lib="Device" part="R"/></comp>
    <comp ref="R5"><value>50</value>
      <libsource lib="Device" part="R"/></comp>
  </components>
  <nets>
    <net code="1" name="GND">
      <node ref="V1" pin="2"/><node ref="R2" pin="2"/>
      <node ref="R4" pin="2"/>
    </net>
    <net code="2" name="N1">
      <node ref="V1" pin="1"/><node ref="R1" pin="1"/><node ref="R3" pin="1"/>
    </net>
    <net code="3" name="N2">
      <node ref="R1" pin="2"/><node ref="R2" pin="1"/><node ref="R5" pin="1"/>
    </net>
    <net code="4" name="N3">
      <node ref="R3" pin="2"/><node ref="R4" pin="1"/><node ref="R5" pin="2"/>
    </net>
  </nets>
</export>
"""


def test_handwritten() -> int:
    print("=" * 72)
    print("第一级：手写 kicadxml 网表 -> IR（确定性断言）")
    print("=" * 72)
    fails = 0
    circ = kicad_netlist_text_to_ir(BRIDGE_NETLIST, name="手写电桥网表")

    expect_nodes = {"V1": ("N1", "GND"), "R1": ("N1", "N2"), "R2": ("N2", "GND"),
                    "R3": ("N1", "N3"), "R4": ("N3", "GND"), "R5": ("N2", "N3")}
    expect_values = {"V1": 12.0, "R1": 100.0, "R2": 200.0,
                     "R3": 150.0, "R4": 300.0, "R5": 50.0}

    print(f"  参考节点判定：{circ.ref_node}  （应为 GND）")
    if circ.ref_node != "GND":
        print("  [失败] 参考节点判定错误")
        fails += 1

    for ref, (a, b) in expect_nodes.items():
        c = circ.by_ref(ref)
        ok = c.nodes == (a, b)
        if not ok:
            fails += 1
        print(f"  [{'通过' if ok else '失败'}] {ref}: {c.nodes}  (期望 {(a,b)})")
        okv = c.value == expect_values[ref]
        if not okv:
            fails += 1
        print(f"  [{'通过' if okv else '失败'}] {ref} 数值 {c.value}  (期望 {expect_values[ref]})")

    # 电压源极性必须被标成"需人工确认"
    v1 = circ.by_ref("V1")
    need = v1.evidence.needs_human
    print(f"  [{'通过' if need else '失败'}] V1 极性被标记为需人工确认 "
          f"(source={v1.evidence.source}, conf={v1.evidence.confidence})")
    if not need:
        fails += 1

    # 电阻的拓扑来自网表，应当 confidence=1.0 且不需要人工
    r1 = circ.by_ref("R1")
    okr = (r1.evidence.source == "exact" and not r1.evidence.needs_human)
    print(f"  [{'通过' if okr else '失败'}] R1 拓扑标为 exact 且无需人工")
    if not okr:
        fails += 1

    print(f"  未满足项（需人工确认）：{[u['ref'] for u in circ.unmet_needs()]}")

    # 极性未知，所以解不唯一 —— 这里只验证"三法是否自洽"，不验证具体电压值
    try:
        pack = run_all(circ)
        print(f"  三法对账总判定：{'通过' if pack['overall_pass'] else '未通过'}")
        if not pack["overall_pass"]:
            fails += 1
    except Exception as e:
        print(f"  [失败] 三法对账异常：{e}")
        fails += 1
    return fails


# ---------------------------------------------------------------- 冒烟


def test_kicad_demos(limit: int = 14) -> int:
    print("\n" + "=" * 72)
    print("第二级：KiCad 自带 demos 冒烟（KiCad 通道 + 测试夹具工厂）")
    print("=" * 72)
    cli = _find_kicad_cli()
    if cli is None:
        print("  跳过：未找到 kicad-cli")
        return 0
    import os
    root = Path(os.environ.get("LOCALAPPDATA")) / "Programs" / "KiCad" / "10.0" \
        / "share" / "kicad" / "demos"
    if not root.is_dir():
        print(f"  跳过：找不到 demos 目录 {root}")
        return 0

    schs = sorted(root.rglob("*.kicad_sch"))[:limit]
    ok_cnt = solved_cnt = 0
    reasons: list[str] = []
    for sch in schs:
        try:
            circ = kicad_sch_to_ir(sch, cli)
            ok_cnt += 1
            n_excl = len(circ.diagnostics)
            try:
                circ.validate()
                run_all(circ)
                solved_cnt += 1
                status = f"可求解  元件数={len(circ.components)}  排除项={n_excl}"
            except Exception as e:
                status = f"解析成功但不可求解：{str(e)[:90]}"
        except Exception as e:
            status = f"解析失败：{str(e)[:90]}"
            reasons.append(f"{sch.name}: {status}")
        print(f"  {sch.name[:44]:<46} {status}")

    if reasons:
        print("\n  解析失败的逐条原因：")
        for r in reasons:
            print(f"    · {r}")
    print(f"\n  汇总：{len(schs)} 张原理图中，KiCad 通道解析成功 {ok_cnt} 张，"
          f"其中可直接求解 {solved_cnt} 张")
    print("  说明：demo 里有大量运放/晶体管/变压器等多端器件，本版本只支持二端元件，")
    print("        被排除的器件已在 diagnostics 里逐条列明 —— 不静默丢弃。")
    return 0


def main() -> int:
    f = test_handwritten()
    f += test_kicad_demos()
    print("\n" + "=" * 72)
    print(f"总计失败项：{f}")
    print("=" * 72)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
