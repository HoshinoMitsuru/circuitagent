"""参数体系：命名、映射（binder）、单位、表达式与控制关系编辑。

这一层是"**参数指向**"的唯一来源。它出错的表现不是算错，而是**指向了别的东西**：

* 界面上写 ``i(Vsense_R1)`` 而题目里只有 ``R1``；
* 改了参数名而表达式还指着旧名（求解阶段才发现，而且三条路径会一致地
  把不认识的符号当 0 —— 三法互校**拦不住**）；
* 删了元件，参数表里那一行悄悄消失，而用户的表达式还在用这个名字。

所以这里测的不是"能不能算"，而是**每一个名字都指得准、并且改了之后整条链一起动**：

1. 体系完整性 —— 每个结点有 u、每条支路有 u/i/自身值，R/C/L 一个不落，单位是 SI；
2. 自动命名 —— 从**位号与结点名**派生（``R1`` / ``i_R1`` / ``u_R1`` / ``u_3``），重名避让；
3. 改名 —— 连带改写所有引用它的表达式，**且不重排用户手写的格式**；
4. 表达式 —— 只认线性组合，未知名词 / 非线性 / 函数一律当场拦住；
5. 控制关系编辑 —— 受控源的控制端与被采样支路必须能改（否则界面上把元件
   改成受控源之后就"没法用了"：求解只会说"控制端没定"，用户找不到地方去定它）；
6. 原子性 —— 一个包里有一项非法，整包不动（不然会留下"一半新一半旧"的参数表，
   而它长得跟正常的一模一样）；
7. 契约 —— ``param_values_by_binder`` 的键与 ``params_view()['by_binder']`` 完全一致
   （跨模块的键名一旦分家，报告就会去表里找一个不存在的名字）。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_params.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import Circuit, CircuitError, Component, Control  # noqa: E402
from app.ir.params import (                                         # noqa: E402
    BINDER_KINDS, CONTROL_MODE, MAX_EXPR_DEPTH, VALUE_UNIT,
    apply_param_edits, auto_symbol, is_valid_symbol, params_view,
    rewrite_symbol,
)
from app.ir.probes import ensure_sense_sources                       # noqa: E402
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


def C(ref, kind, a, b, value=None, ctrl=None):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value, ctrl=ctrl)


# ---------------------------------------------------------------- 样机


def ckt_controlled() -> Circuit:
    """带自定义表达式的受控源电路（表达式**用参数名**书写）。"""
    return Circuit(name="受控源", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 2000.0),
        C("E1", "E", "3", "0", None,
          ctrl=Control(mode="V", nodes=("2", "0"), expr="3*u_2 + 5")),
        C("R3", "R", "3", "0", 1000.0),
    ])


def ckt_passive() -> Circuit:
    """含 R/C/L 的无源网络 —— C/L 不参与直流求解，但**必须在参数表里有位置**。"""
    return Circuit(name="RCL", components=[
        C("V1", "V", "1", "0", 5.0),
        C("R1", "R", "1", "2", 100.0),
        C("C1", "C", "2", "0", 1e-6),
        C("L1", "L", "2", "0", 1e-3),
    ])


def ckt_ccvs() -> Circuit:
    """电流控制型（会插 0V 探针），用于控制关系编辑。"""
    return Circuit(name="CCVS", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 1000.0),
        C("H1", "H", "3", "0", 2000.0, ctrl=Control(mode="I", ref="R1")),
        C("R3", "R", "3", "0", 1000.0),
    ])


def solved(circ: Circuit) -> dict:
    circ.validate()
    return run_all(circ)


# ---------------------------------------------------------------- 1. 完整性


def test_coverage() -> None:
    banner("1. 参数体系完整性：每个结点、每条支路、每种量都有名字与 SI 单位")

    circ = ckt_passive()
    solvable, _ = ensure_sense_sources(circ)
    pv = params_view(circ)
    items = pv["items"]
    by_binder = {f"{e['binder_kind']}:{e['binder'][1]}": e for e in items}

    # ---- 结点电压：每个结点一个
    want_nodes = set(solvable.nodes)
    got_nodes = {e["binder"][1] for e in items if e["binder_kind"] == "node_u"}
    check("每个结点都有一个电压参数", got_nodes == want_nodes,
          f"缺 {want_nodes - got_nodes}，多 {got_nodes - want_nodes}")

    # ---- 支路：每条支路各有一个电流、一个电压、一个"自身值"
    refs = {c.ref for c in circ.components}
    for kind, label in (("branch_i", "支路电流"), ("branch_u", "支路电压"),
                        ("value", "元件值")):
        got = {e["binder"][1] for e in items if e["binder_kind"] == kind}
        check(f"每条支路都有一个{label}参数（{kind}）", got == refs,
              f"缺 {refs - got}，多 {got - refs}")

    # ---- R / C / L 的元件值都在，且单位各自正确（用户点名的三项）
    for ref, unit in (("R1", "Ω"), ("C1", "F"), ("L1", "H"), ("V1", "V")):
        e = by_binder.get(f"value:{ref}")
        check(f"{ref} 的元件值在参数表里，单位 {unit}",
              e is not None and e["unit"] == unit,
              f"{e and (e['unit'], e['value'])}")

    # ---- C/L 不参与直流求解，但**不许**因此从参数表里消失
    check("C/L 虽然不参与直流求解，仍在参数表里占位（不静默消失）",
          "value:C1" in by_binder and "value:L1" in by_binder,
          f"{sorted(by_binder)}")

    # ---- 分组：按量分组，组内单位的量纲一致
    groups = {g["quantity"]: g for g in pv["groups"]}
    check("分组按 电压 / 电流 / 元件值 三类，且每类带量纲级单位",
          set(groups) == {"u", "i", "value"}
          and groups["u"]["unit"] == "V" and groups["i"]["unit"] == "A",
          f"{[(g['quantity'], g['unit'], len(g['symbols'])) for g in pv['groups']]}")
    check("每组只装自己那一类的参数",
          all(all(by_binder[f"{e['binder_kind']}:{e['binder'][1]}"]["quantity"]
                  == g["quantity"] for e in items if e["quantity"] == g["quantity"])
              for g in pv["groups"]), "")

    # ---- 单位表本身要说得清（界面表头直接用它）
    units = pv["units"]
    check("单位表头明示 SI 与受控源增益的四种单位",
          "SI" in units["header"] and "V/V" in units["header"]
          and "A/A" in units["header"], units["header"])
    check("受控源增益单位按种类推出（V/V · S · Ω · A/A）",
          all(VALUE_UNIT[k] == u for k, u in
              (("E", "V/V"), ("G", "S"), ("H", "Ω"), ("F", "A/A"))),
          f"{ {k: VALUE_UNIT[k] for k in 'EGHF'} }")
    check("绑定种类只有这四种（没有野生类别）",
          set(BINDER_KINDS) == {"node_u", "value", "branch_i", "branch_u"},
          f"{BINDER_KINDS}")

    # ---- 绑定键的写法处处一致
    check("绑定键统一写成 kind:key",
          all(k in by_binder for k in
              ("node_u:0", "branch_i:R1", "branch_u:R1", "value:R1")),
          f"{sorted(by_binder)[:8]}…")

    # ---- 未手动命名 → 自动命名，且都从身份派生
    check("没手动命名的参数全部自动命名（auto=True）",
          all(e["auto"] for e in items) and pv["auto_count"] == len(items)
          and pv["custom_count"] == 0,
          f"auto={pv['auto_count']} custom={pv['custom_count']} count={pv['count']}")
    for binder, want in ((("value", "R1"), "R1"), (("branch_i", "R1"), "i_R1"),
                         (("branch_u", "R1"), "u_R1"), (("node_u", "0"), "u_0")):
        check(f"默认名从绑定派生：{binder[0]}:{binder[1]} → {want}",
              auto_symbol(binder, []) == want, auto_symbol(binder, []))
    check("默认名撞车时加序号（不重名 —— 重名会让表达式静默指向错的那个）",
          auto_symbol(("value", "R1"), ["R1"]) == "R1_2"
          and auto_symbol(("value", "R1"), ["R1", "R1_2"]) == "R1_3",
          auto_symbol(("value", "R1"), ["R1", "R1_2"]))
    check("结点名里有不能进表达式的字符时会被压成合法标识符",
          auto_symbol(("node_u", "out+"), []) == "u_out"
          and is_valid_symbol(auto_symbol(("node_u", "3.3"), [])),
          auto_symbol(("node_u", "3.3"), []))

    # ---- 每条参数都能说出"它是什么"
    check("每条参数都有可读的绑定描述与中文量名",
          all(e["binder_text"] and e["binder_label"] and e["quantity_label"]
              for e in items),
          f"{[(e['symbol'], e['binder_text']) for e in items[:4]]}")


# ---------------------------------------------------------------- 2. 改名


def test_rename() -> None:
    banner("2. 改名：连带改写表达式、保留原文格式、按绑定追回取值")

    # ---- 基本：改名生效、auto 翻成 False
    circ = ckt_controlled()
    circ.validate()
    res = apply_param_edits(circ, renames={"u_2": "uA", "i_R1": "i1"})
    pv = res["params"]
    by_sym = {e["symbol"]: e for e in pv["items"]}
    check("改名后新名字在表里、旧名字不在",
          "uA" in by_sym and "u_2" not in by_sym and "i1" in by_sym,
          f"{sorted(by_sym)[:8]}…")
    check("改过名的参数不再算自动命名（界面要能一眼分清谁起的）",
          by_sym["uA"]["auto"] is False and by_sym["i1"]["auto"] is False,
          f"auto={{'uA': {by_sym['uA']['auto']}, 'i1': {by_sym['i1']['auto']}}}")
    check("没动过的参数仍是自动命名", by_sym["R1"]["auto"] is True, "")

    # ---- ★ 表达式必须跟着改，否则名字一改表达式就指向空气
    check("改 u_2 时引用它的表达式被同步改写",
          circ.by_ref("E1").ctrl.expr.replace(" ", "") == "3*uA+5",
          repr(circ.by_ref("E1").ctrl.expr))
    check("rewritten 里报出被改写的受控源（不静默）",
          any(r["symbol"] == "uA" and r["rewritten"] == ["E1"] for r in res["renames"]),
          f"{res['renames']}")

    # ---- ★ 改写只动命中的那一段字面量，用户手写的空格原样保留
    circ2 = ckt_controlled()
    circ2.validate()
    apply_param_edits(circ2, renames={"u_2": "uA"})
    check("改写表达式**不重排用户手写的格式**（空格原样保留）",
          circ2.by_ref("E1").ctrl.expr == "3*uA + 5",
          f"原文 '3*u_2 + 5' → 现在 {circ2.by_ref('E1').ctrl.expr!r}")
    check("按记号替换：把 u_1 改成 u_10 不会把已有的 u_10 再改成 u_100",
          rewrite_symbol("u_1 + u_10", "u_1", "u_10") == "u_10 + u_10",
          rewrite_symbol("u_1 + u_10", "u_1", "u_10"))
    check("不碰数字里的字母（4k7 不会被当成含 k 的记号）",
          rewrite_symbol("4k7 + k", "k", "kk") == "4k7 + kk",
          rewrite_symbol("4k7 + k", "k", "kk"))

    # ---- 改名之后求解照旧，且**按绑定**能把值追回来
    pack = solved(ckt_controlled())
    pack2 = solved(circ)
    pvb = pack2["param_values_by_binder"]
    check("改名后求解仍通过总判定", bool(pack2["overall_pass"]),
          "；".join(f"{k}={v}" for k, v in pack2["failures"].items()))
    check("改名不改绑定：按 node_u:2 仍能取到 V(2) 的值（键不随名字走）",
          pvb.get("node_u:2") == "20/3", f"node_u:2 = {pvb.get('node_u:2')!r}")
    check("改名后按新名字取值也与原来一致（表达式仍然算得对）",
          pack2["param_values"].get("uA") == pack["param_values"].get("u_2")
          == "20/3",
          f"uA={pack2['param_values'].get('uA')!r}")
    check("改名后解的整体数值与改名前逐项相同",
          pack2["param_values_by_binder"] == pack["param_values_by_binder"],
          f"{ {k: pack2['param_values_by_binder'][k] for k in ('node_u:3',) } }")

    # ---- 契约：binder 索引的键 == 参数表的键
    pv3 = params_view(circ)
    check("★ param_values_by_binder 的键与 params_view()['by_binder'] 完全一致",
          set(pvb) == set(pv3["by_binder"]),
          f"只在解里 {sorted(set(pvb) - set(pv3['by_binder']))}；"
          f"只在表里 {sorted(set(pv3['by_binder']) - set(pvb))}")

    # ---- 非法名字：当场拒绝，并说清为什么
    for bad, why in (("2x", "以数字开头"), ("a-b", "含减号"), ("a b", "含空格"),
                     ("", "空名字"), ("R1", "与已有参数重名")):
        try:
            apply_param_edits(ckt_controlled(), renames={"u_2": bad})
            check(f"非法参数名 {bad!r}（{why}）被拒绝", False, "没有报错")
        except CircuitError as e:
            check(f"非法参数名 {bad!r}（{why}）被拒绝", True, str(e)[:76])
    try:
        apply_param_edits(ckt_controlled(), renames={"zz": "a"})
        check("改一个不存在的参数名被拒绝", False, "没有报错")
    except CircuitError as e:
        check("改一个不存在的参数名被拒绝", "没有" in str(e), str(e)[:76])
    # ★ 同一个包里先改出一个名字、再把另一个也改成它 —— 必须撞车被拦。
    #   这一条测的是"事务内部"的重名检查，只测单个改名是漏的。
    try:
        apply_param_edits(ckt_controlled(), renames={"u_2": "uA", "u_1": "uA"})
        check("同一个包里改出两个同名参数被拒绝", False, "没有报错")
    except CircuitError as e:
        check("同一个包里改出两个同名参数被拒绝", "占用" in str(e), str(e)[:76])


# ---------------------------------------------------------------- 3. 表达式


def test_expression() -> None:
    banner("3. 受控源表达式：只认线性组合，坏的一律当场拦住")

    # ---- 合法：线性组合 + 常数项
    circ = ckt_controlled()
    res = apply_param_edits(circ, exprs={"E1": "3*u_2 + 5"})
    check("按参数名书写的线性表达式被接受",
          circ.by_ref("E1").ctrl.expr == "3*u_2 + 5"
          and res["exprs"][0]["mode"] == "自定义表达式",
          f"{res['exprs']}")
    pack = solved(circ)
    check("表达式被真正用上：V(3) = 3·V(2) + 5 = 25（不是静默忽略）",
          pack["solutions"]["mna"]["node_voltages"]["3"] == "25",
          f"V(3)={pack['solutions']['mna']['node_voltages']['3']}")

    # ---- 合法：引用支路电流（走的是另一条代码路径）
    tot = Circuit(name="tot", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("R2", "R", "1", "2", 1000.0),
        C("F1", "F", "2", "0", None,
          ctrl=Control(mode="I", ref="R1", expr="3*i_R1")),
    ])
    apply_param_edits(tot, exprs={"F1": "3*i_R1"})
    pack = solved(tot)
    check("表达式里引用支路电流也生效：i(F1) = 3·i_R1 = 0.03 A",
          pack["param_values_by_binder"]["branch_i:F1"] == "3/100",
          f"{pack['param_values_by_binder']['branch_i:F1']!r}")

    # ---- 留空 = 回到标准形（且会连带把"待改"标记清掉）
    circ = ckt_controlled()
    circ.by_ref("E1").value = 3.0                     # 先给它一个增益
    res = apply_param_edits(circ, exprs={"E1": ""})
    check("表达式留空 = 回到增益的标准形",
          circ.by_ref("E1").ctrl.expr == ""
          and res["exprs"][0]["mode"] == "标准形（增益）",
          f"{res['exprs']}")

    # ---- 坏表达式：必须是 CircuitError（不是拖到求解阶段变成"算不出来"）
    bad_cases = [
        ("3*uQ", "未知名词"),
        ("u_1*u_2", "两个含参量相乘（非线性）"),
        ("1/u_1", "含参量作除数（非线性）"),
        ("sin(u_1)", "函数"),
        ("u_1^2", "幂运算"),
        ("3**u_1", "连续运算符"),
        ("(3*u_2", "括号没配平"),
        ("3 u_2", "漏了运算符"),
        ("2u_2", "漏了乘号（2u_2 不是合法记号）"),
    ]
    for expr, why in bad_cases:
        try:
            apply_param_edits(ckt_controlled(), exprs={"E1": expr})
            check(f"坏表达式 {expr!r}（{why}）被拒绝", False, "没有报错")
        except CircuitError as e:
            check(f"坏表达式 {expr!r}（{why}）被拒绝", True, str(e).splitlines()[0][:74])

    # ---- 未知名词的报错要**列出可用的名字**（否则用户只能猜）
    try:
        apply_param_edits(ckt_controlled(), exprs={"E1": "3*uQ"})
    except CircuitError as e:
        check("撞上未知名词时报错里列出了可用参数名", "可用的名字有" in str(e),
              str(e).splitlines()[0][:100])
        check("撞上未知名词时不会静默当成 0", "不是已知参数名" in str(e), "")

    # ---- 非受控源不许填表达式
    try:
        apply_param_edits(ckt_controlled(), exprs={"R1": "3*u_2"})
        check("给非受控源填控制表达式被拒绝", False, "没有报错")
    except CircuitError as e:
        check("给非受控源填控制表达式被拒绝", "不是受控源" in str(e), str(e)[:78])

    # ---- 清掉表达式又没有增益：不许留下一个"什么都不是"的受控源
    try:
        apply_param_edits(ckt_controlled(), exprs={"E1": ""})
        check("清掉表达式又没有增益时被拒绝（不许留下无来源的受控源）", False,
              "没有报错")
    except CircuitError as e:
        check("清掉表达式又没有增益时被拒绝（不许留下无来源的受控源）",
              "增益" in str(e), str(e)[:78])

    check("表达式嵌套深度有上限（防受控源互相引用绕成环）",
          isinstance(MAX_EXPR_DEPTH, int) and MAX_EXPR_DEPTH >= 2,
          f"MAX_EXPR_DEPTH={MAX_EXPR_DEPTH}")


# ---------------------------------------------------------------- 4. 控制关系


def test_control_edit() -> None:
    banner("4. 控制关系可编辑：控制端、被采样支路、以及各类拒绝")

    # ---- 电压控制：改控制端，解必须跟着变
    circ = ckt_controlled()
    circ.by_ref("E1").ctrl.expr = ""                  # 回到标准形，增益 3
    circ.by_ref("E1").value = 3.0
    before = solved(circ)["param_values_by_binder"]["node_u:3"]
    res = apply_param_edits(circ, ctrls={"E1": {"nodes": ["1", "0"]}})
    after = solved(circ)["param_values_by_binder"]["node_u:3"]
    check("改电压控制端后控制端真的换了（Ctrl.nodes 已更新）",
          tuple(circ.by_ref("E1").ctrl.nodes) == ("1", "0"),
          f"{circ.by_ref('E1').ctrl.nodes}")
    check("改控制端后返回值里逐条报出新控制关系（不静默）",
          len(res["ctrls"]) == 1 and res["ctrls"][0]["nodes"] == ["1", "0"]
          and res["ctrls"][0]["mode"] == "V",
          f"{res['ctrls']}")
    check("改控制端后解真的变了（说明改的是被求解的那一份，不是显示层）",
          before != after, f"改前 V(3)={before} 改后 V(3)={after}")
    check("新控制端的值算得对：3·V(1) = 3·10 = 30", after == "30", f"{after!r}")

    # ---- 电流控制：改被采样支路
    circ = ckt_ccvs()
    circ.validate()
    res = apply_param_edits(circ, ctrls={"H1": {"ref": "R2"}})
    check("改被采样支路生效", circ.by_ref("H1").ctrl.ref == "R2",
          f"ref={circ.by_ref('H1').ctrl.ref!r}")
    check("改被采样支路时旧探针记录被清掉（不留下为 A 而插、现在采 B 的 0V 源）",
          circ.by_ref("H1").ctrl.sense_ref == "",
          f"sense_ref={circ.by_ref('H1').ctrl.sense_ref!r}")
    pack = solved(circ)
    check("换采样支路后求解照常通过", bool(pack["overall_pass"]),
          "；".join(f"{k}={v}" for k, v in pack["failures"].items()))
    # i(R2) 沿 2→0 = 5/1000；H1 输出 2000·0.005 = 10 V → 与采 R1 时同值（本题对称）
    check("换采样支路后的解与手算一致（i(R2) = 5 mA → V(3) = 10 V）",
          pack["param_values_by_binder"]["node_u:3"] == "10",
          f"V(3)={pack['param_values_by_binder']['node_u:3']!r}")

    # ---- 拒绝：控制端不成对 / 两点相同 / 结点不存在 / 自指 / 控制方式
    rejects = [
        ({"E1": {"nodes": ["2"]}}, "控制端只有一个结点", "两个"),
        ({"E1": {"nodes": ["2", "2"]}}, "控制端两点相同", "恒为"),
        ({"E1": {"nodes": ["2", "99"]}}, "控制端结点不存在", "不在电路里"),
        ({"E1": {"mode": "I"}}, "想把压控改成流控", "由元件类型决定"),
        ({"R1": {"nodes": ["1", "0"]}}, "给非受控源指定控制支路", "不是受控源"),
    ]
    for spec, why, hint in rejects:
        try:
            apply_param_edits(ckt_controlled(), ctrls=spec)
            check(f"{why}被拒绝", False, "没有报错")
        except CircuitError as e:
            check(f"{why}被拒绝", hint in str(e), str(e).splitlines()[0][:78])

    for spec, why in (({"H1": {"ref": "H1"}}, "采样自己（自指）"),
                      ({"H1": {"ref": "RX"}}, "采样不存在的支路"),
                      ({"H1": {"ref": ""}}, "被采样支路留空")):
        try:
            apply_param_edits(ckt_ccvs(), ctrls=spec)
            check(f"{why}被拒绝", False, "没有报错")
        except CircuitError as e:
            check(f"{why}被拒绝", True, str(e).splitlines()[0][:78])


# ---------------------------------------------------------------- 5. 原子性


def test_atomicity() -> None:
    banner("5. 原子性：一个包里有一项非法，整包不动")

    # ---- 合法改名 + 非法控制端：改名**不许**生效
    circ = ckt_controlled()
    circ.validate()
    before_expr = circ.by_ref("E1").ctrl.expr
    before_syms = set(params_view(circ)["by_binder"].values())
    try:
        apply_param_edits(circ,
                          renames={"u_2": "zz"},
                          ctrls={"E1": {"nodes": ["2", "99"]}})
        check("合法改名 + 非法控制端：整包被拒绝", False, "居然成功了")
    except CircuitError:
        pv = params_view(circ)
        check("★ 整包被拒后，改名没有留下痕迹（要么全成、要么原样）",
              "u_2" in {e["symbol"] for e in pv["items"]}
              and "zz" not in {e["symbol"] for e in pv["items"]},
              f"zz 是否出现={'zz' in {e['symbol'] for e in pv['items']}}")
        check("整包被拒后，表达式也没被改写",
              circ.by_ref("E1").ctrl.expr == before_expr,
              repr(circ.by_ref("E1").ctrl.expr))
        check("整包被拒后，参数名集合与失败前完全一致",
              set(pv["by_binder"].values()) == before_syms,
              f"{sorted(set(pv['by_binder'].values()) ^ before_syms)}")

    # ---- 非法表达式 + 合法改名：同样整包不动
    circ = ckt_controlled()
    circ.validate()
    try:
        apply_param_edits(circ, renames={"u_2": "zz"}, exprs={"E1": "3*unknownname"})
        check("合法改名 + 非法表达式：整包被拒绝", False, "居然成功了")
    except CircuitError:
        check("同上：改名同样没有留下痕迹",
              "u_2" in {e["symbol"] for e in params_view(circ)["items"]}
              and "zz" not in {e["symbol"] for e in params_view(circ)["items"]},
              "")

    # ---- 真实生效的那一包：三样一起提交也必须都对
    circ = ckt_controlled()
    circ.validate()
    res = apply_param_edits(circ,
                            renames={"u_2": "uA", "i_R1": "i1"},
                            exprs={"E1": "3*uA + 5"},
                            ctrls={"E1": {"nodes": ["2", "0"]}})
    check("改名 + 表达式 + 控制关系一次提交成功（三者互相引用，必须同一个事务）",
          circ.by_ref("E1").ctrl.expr == "3*uA + 5"
          and tuple(circ.by_ref("E1").ctrl.nodes) == ("2", "0"),
          f"expr={circ.by_ref('E1').ctrl.expr!r} nodes={circ.by_ref('E1').ctrl.nodes}")
    check("一次提交的返回值里三样都逐条报出", all(res[k] for k in
                                              ("renames", "exprs", "ctrls")),
          f"{sorted(res)}")
    pack = solved(circ)
    check("一次提交后求解通过，且 uA = 20/3、V(3) = 25",
          pack["param_values"].get("uA") == "20/3"
          and pack["param_values_by_binder"]["node_u:3"] == "25",
          f"uA={pack['param_values'].get('uA')!r} "
          f"V(3)={pack['param_values_by_binder']['node_u:3']!r}")


# ---------------------------------------------------------------- 6. 孤儿


def test_orphans() -> None:
    banner("6. 孤儿参数：元件删了，名字留着并显式标出（不悄悄消失）")

    circ = ckt_controlled()
    circ.validate()
    apply_param_edits(circ, renames={"u_2": "uA"})
    circ.components = [c for c in circ.components if c.ref != "R1"]
    circ.validate(allow_incomplete=True)
    circ.sync_params()
    pv = params_view(circ)

    orphan_syms = set(pv["orphans"])
    check("R1 被删后，它的三个参数都变成孤儿并报出来",
          orphan_syms == {"R1", "i_R1", "u_R1"}, f"{sorted(orphan_syms)}")
    check("孤儿参数没有被悄悄删掉（用户的表达式可能还引用着它）",
          all(s in {e["symbol"] for e in pv["items"]} for s in orphan_syms),
          f"{sorted(e['symbol'] for e in pv['items'])}")
    check("孤儿参数带 orphan 标记，界面才能单独列一栏",
          all(e["orphan"] for e in pv["items"] if e["symbol"] in orphan_syms)
          and not any(e["orphan"] for e in pv["items"]
                      if e["symbol"] not in orphan_syms), "")
    check("★ 用户改过名的参数（uA）不受影响，仍在表里且不标孤儿",
          any(e["symbol"] == "uA" and not e["orphan"] for e in pv["items"]), "")
    check("孤儿参数仍带完整绑定信息（否则用户不知道它原来指哪）",
          all(e["binder_text"] and e["binder_kind"] for e in pv["items"]
              if e["symbol"] in orphan_syms), "")


# ---------------------------------------------------------------- 7. 受控源视图


def test_controlled_view() -> None:
    banner("7. controlled 视图：控制方式、增益单位、探针与网表器件名")

    circ = Circuit(name="四类", components=[
        C("V1", "V", "1", "0", 10.0),
        C("R1", "R", "1", "2", 1000.0),
        C("R2", "R", "2", "0", 1000.0),
        C("E1", "E", "3", "0", 2.0, ctrl=Control(mode="V", nodes=("2", "0"))),
        C("G1", "G", "3", "0", 0.001, ctrl=Control(mode="V", nodes=("2", "0"))),
        C("H1", "H", "3", "0", 500.0, ctrl=Control(mode="I", ref="R1")),
        C("F1", "F", "3", "0", 2.0,
          ctrl=Control(mode="I", ref="R1", expr="2*i_R1")),
        C("R3", "R", "3", "0", 1000.0),
    ])
    circ.validate()
    pv = params_view(circ)
    rows = {c["ref"]: c for c in pv["controlled"]}
    check("四种受控源都在 controlled 视图里",
          set(rows) == {"E1", "G1", "H1", "F1"}, f"{sorted(rows)}")

    for ref, mode, gain_sym, unit in (("E1", "V", "μ", "V/V"),
                                      ("G1", "V", "gm", "S"),
                                      ("H1", "I", "rm", "Ω"),
                                      ("F1", "I", "α", "A/A")):
        r = rows[ref]
        check(f"{ref}：控制方式 {mode} / 增益符号 {gain_sym} / 单位 {unit}",
              r["mode"] == mode and r["gain_symbol"] == gain_sym
              and r["unit"] == unit,
              f"{ {k: r[k] for k in ('mode', 'gain_symbol', 'unit')} }")
        check(f"{ref}：增益在参数表里有一个名字（界面要显示它）",
              bool(r["value_param"]) and r["value_param"] in pv["by_binder"].values(),
              f"value_param={r['value_param']!r}")

    check("CONTROL_MODE 与 controlled 视图里的 mode 一致",
          all(rows[k]["mode"] == CONTROL_MODE[v["kind"]]
              for k, v in (("E1", {"kind": "E"}), ("G1", {"kind": "G"}),
                           ("H1", {"kind": "H"}), ("F1", {"kind": "F"}))), "")

    # ---- 自定义表达式：网表器件名会变成行为源 B<ref>，必须提前说清
    check("用了自定义表达式的受控源，网表器件名标成 B<ref>",
          rows["F1"]["net_name"] == "BF1" and rows["F1"]["expr"] == "2*i_R1",
          f"net_name={rows['F1']['net_name']!r}")
    check("没用表达式的受控源，网表器件名就是位号",
          rows["E1"]["net_name"] == "E1", f"{rows['E1']['net_name']!r}")

    # ---- 人读控制关系必须指向用户认得的东西
    check("电压控制：人读写法给出两个控制端结点",
          rows["E1"]["control"] == "2·(V(2) − V(0))", f"{rows['E1']['control']!r}")
    check("表达式：人读写法里带着表达式原文与标准形对照",
          "2*i_R1" in rows["F1"]["control"]
          and "标准形" in rows["F1"]["control"], f"{rows['F1']['control']!r}")

    # ---- 三个名字必须分开：受控源自己 / 题目里的被采样支路 / 实际取电流的支路
    check("★ 未展开时：受控源位号 ≠ 被采样支路，实际取电流的就是题目那条",
          rows["H1"]["ref"] == "H1" and rows["H1"]["sampled_ref"] == "R1"
          and rows["H1"]["sampling_ref"] == "R1" and rows["H1"]["probe_ref"] == ""
          and rows["H1"]["sampling_is_probe"] is False,
          f"{ {k: rows['H1'][k] for k in ('ref', 'sampled_ref', 'probe_ref', 'sampling_ref', 'sampling_is_probe')} }")

    # ---- 展开（求解/导出网表时做的）之后：实际取电流的是探针，但被采样支路仍是 R1
    expanded, _ = ensure_sense_sources(circ)
    pv2 = params_view(expanded)
    r2 = {c["ref"]: c for c in pv2["controlled"]}
    check("★ 展开后 sampled_ref 仍是题目支路 R1，probe_ref 才是探针，且显式标出",
          r2["H1"]["ref"] == "H1" and r2["H1"]["sampled_ref"] == "R1"
          and r2["H1"]["probe_ref"].startswith("Vsense_")
          and r2["H1"]["sampling_ref"] == r2["H1"]["probe_ref"]
          and r2["H1"]["sampling_is_probe"] is True,
          f"{ {k: r2['H1'][k] for k in ('ref', 'sampled_ref', 'probe_ref', 'sampling_ref', 'sampling_is_probe')} }")
    check("展开后人读控制关系里出现的是题目支路名，探针另加说明",
          "i(R1)" in r2["H1"]["control"] and "探针" in r2["H1"]["control"],
          f"{r2['H1']['control']!r}")
    check("展开后受控源的增益参数名不受影响（名字跟绑定走，不跟探针走）",
          r2["H1"]["value_param"] == rows["H1"]["value_param"] == "H1",
          f"{r2['H1']['value_param']!r}")
    # ---- 被采样支路本身就是电压源：probe_ref 等于它是**有意的**语义
    #      （表示"直接取它"，不是"插了个探针"），界面据此显示不同说明
    c2 = Circuit(name="直采", components=[
        C("Vs", "V", "1", "0", 10.0),
        C("R1", "R", "1", "0", 1000.0),
        C("H1", "H", "2", "0", 2000.0, ctrl=Control(mode="I", ref="Vs")),
        C("R2", "R", "2", "0", 1000.0),
    ])
    e2, _ = ensure_sense_sources(c2)
    h = {c["ref"]: c for c in params_view(e2)["controlled"]}["H1"]
    check("被采样支路本身就是电压源时：probe_ref == sampled_ref 且不标成探针",
          h["sampled_ref"] == "Vs" and h["probe_ref"] == "Vs"
          and h["sampling_is_probe"] is False,
          f"{ {k: h[k] for k in ('sampled_ref', 'probe_ref', 'sampling_is_probe')} }")


# ---------------------------------------------------------------- 跑


def main() -> int:
    print("参数体系：命名 / 映射 / 单位 / 表达式 / 控制关系")
    test_coverage()
    test_rename()
    test_expression()
    test_control_edit()
    test_atomicity()
    test_orphans()
    test_controlled_view()
    print("\n" + "=" * 72)
    print(f"总计失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
