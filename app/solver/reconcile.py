"""三法对账：把三条独立代码路径的结果摆在同一张表上比对。

**不允许静默降级**：任何一条路径失败或缺失，都要在表里显式占一行并写清原因。
少了 ngspice 而报告看起来"全部通过"，比没有这张表更危险 ——
它会让人以为验过了。

判定阈值：
- 前两法都是精确有理数，互比应当**严格为 0**（不是"很小"）
- 与 ngspice 比，用相对偏差 1e-9 / 绝对偏差 1e-12。
  实测 ngspice 对线性电阻网络直接求解，精度接近机器精度
  （`10·120/220` 给出 5.454545455，真值 5.4545454545，相对偏差 ~1e-10），
  所以 1e-9 这个阈值是留了余量而不是放水。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..ir.model import Circuit, CircuitError
from ..ir.params import params_view
from .base import Solution
from .branch import branch_current_method
from .controlled import parameter_values
from .equivalence import superposition
from .linalg import frac_float, frac_str
from .mna import node_voltage_method
from .ngspice import ngspice_method, probe_availability
from .verify import verify_all

REL_TOL = 1e-9
ABS_TOL = 1e-12

METHODS = ("mna", "branch", "ngspice")


def _fl(x: Any) -> float:
    if isinstance(x, Fraction):
        return frac_float(x)
    return float(x)


def _close(a: Any, b: Any) -> bool:
    try:
        fa, fb = _fl(a), _fl(b)
    except (TypeError, ValueError):
        return False
    if fa != fa or fb != fb:                     # NaN
        return False
    return abs(fa - fb) <= max(ABS_TOL, REL_TOL * max(abs(fa), abs(fb)))


def _by_binder(circuit: Circuit, values: dict[str, Any]) -> dict[str, Any]:
    """``{绑定 -> 取值}``，键写作 ``"branch_i:R1"`` / ``"node_u:3"``。

    ★ **为什么要有这一份**：用户随时能改参数名，而改名会让所有
    "按名字索引"的表当场失效 —— 界面上就变成"这个参数还没有取值"。
    可电路没变、解也没变，变的只是名字：那是**假信息**，会让人回头去改一张
    本来没问题的图。绑定（哪条支路的电流、哪个结点的电压）不随改名而变，
    用它当键就绕开了这个问题。

    键的写法与 ``params_view()["by_binder"]`` 完全一致（都是 ``kind:key``），
    界面拿 ``item.binder`` 拼一下就能查到 —— 两边一旦分家就是"查不到值"。

    查不到值的参数**直接略过**，不填 0、不填 None：``i`` 待定本身就是
    一条正常的物理结论（理想短线并联时电流不唯一），界面上显示"—"才是对的。
    """
    if not values:
        return {}
    out: dict[str, Any] = {}
    for e in (params_view(circuit).get("items") or []):
        sym = e.get("symbol")
        if sym not in values:
            continue
        bk, key = e.get("binder") or ("", "")
        out[f"{bk}:{key}"] = values[sym]
    return out


def run_all(
    circuit: Circuit, *, want_superposition: bool = True, reduce_dc: bool = True
) -> dict[str, Any]:
    """跑满三条路径 + 验算 + 叠加，产出完整对账包。

    ``reduce_dc``：电路里有 C/L 时先做直流稳态化简（C 开路、L 短路），
    化简过程会写进返回包的 ``reduction`` 字段并出现在报告里 ——
    **静默化简等于篡改题目**，必须留下痕迹。
    """
    circuit.validate()

    reduction: dict[str, Any] | None = None
    rep = None
    original = circuit
    if reduce_dc and any(c.kind in ("C", "L") for c in circuit.components):
        from .dc_reduce import reduce_to_dc
        # ★ 约束矛盾（如电感把理想电压源短路）会在这里抛 CircuitUnsatisfiable。
        #   绝不能当场吞掉 —— 那正是本项目历史上最严重的一次静默降级。
        circuit, rep = reduce_to_dc(circuit)
        if not circuit.components:
            raise CircuitError("直流稳态化简后电路为空")

    solutions: dict[str, Solution] = {}
    failures: dict[str, str] = {}

    for key, fn, label in (
        ("mna", node_voltage_method, "节点电压法"),
        ("branch", branch_current_method, "支路电流法"),
        ("ngspice", ngspice_method, "ngspice"),
    ):
        try:
            solutions[key] = fn(circuit)
        except CircuitError as e:
            failures[key] = str(e)
        except Exception as e:                   # pragma: no cover
            failures[key] = f"{type(e).__name__}: {e}"

    if "mna" not in solutions:
        raise CircuitError(
            "主解（节点电压法）失败，无法继续："
            + failures.get("mna", "未知原因")
        )

    # ------------------------------------------------ 反算被短路移除支路的电流
    # 为什么放在这里而不是 dc_reduce 里：反算要拿**已解出的**支路电流当已知量。
    # 不补这一步的话，电感（直流下即短路）合并节点后会从支路表里整体消失，
    # 而学生要的往往正是 i_L —— 少一条支路还不说明，就是静默丢信息。
    if rep is not None:
        from .dc_reduce import recover_shorted_currents
        recover_shorted_currents(original, circuit, rep, solutions["mna"])
        reduction = rep.to_dict()
        reduction["summary"] = rep.summary()

    exact_keys = [k for k in ("mna", "branch") if k in solutions]
    valid_keys = [k for k in METHODS if k in solutions]

    # ------------------------------------------------ 参数名 -> 数值
    # ★ 求解跑的是**化简后/展开后**的电路，而参数表属于**原始**电路，
    #   两者不是同一张。所以要显式把两条对应关系补上：
    #     * 被电感合并掉的节点名 → 代表元（不补，``u_3`` 就"没有值"）；
    #     * 被电容开路移除的支路 → 已由 recover_shorted_currents 反算出来
    #       （不补，``i_L1`` 也"没有值"）。
    #   漏掉它们，界面上就会冒出一串"该参数暂无取值"，而电路其实算得好好的 ——
    #   用户会以为是自己画错了，转头去改一张没问题的图。
    param_values: dict[str, Any] = {}
    try:
        alias: dict[str, str] = {}
        if rep is not None:
            for group in rep.merged_groups:       # [代表元, 被合掉的...]
                for n in group[1:]:
                    alias[str(n)] = str(group[0])
        base_sol = solutions["mna"]
        node_v: dict[str, Any] = {}
        for n in original.nodes:
            node_v[n] = base_sol.node_voltages.get(alias.get(n, n))

        # ★ 化简层移除的支路不在 mna 的解里，但它们的取值**是已知的**，
        #   只是存在别处：
        #     * 电感（被短路合并）→ `recover_shorted_currents` 用原节点 KCL
        #       反算出的电流，落在化简报告的 `shorted[*]["current"]`；
        #     * 电容（视为开路）→ 直流稳态下 i_C = C·du/dt = 0，这是定义。
        #   不并进来的话，界面上 `i_L1`／`i_C1` 会显示成"暂无取值"，
        #   而用户要的恰恰是 i_L1（电感电流是这类题的常见答案）。
        currents: dict[str, Any] = dict(base_sol.branch_currents)
        if rep is not None:
            for e in rep.shorted:
                raw = e.get("current")
                if raw is None:
                    continue                     # "i 待定"是正常结论，不猜
                try:
                    currents.setdefault(e["ref"], Fraction(str(raw)))
                except (ValueError, ZeroDivisionError):
                    currents.setdefault(e["ref"], raw)
            for e in rep.opened:
                currents.setdefault(e["ref"], Fraction(0))

        param_values = parameter_values(original, node_v, currents)
        # ★ 精确值一律走 frac_str：这是个**响应体**，Fraction 直接塞进
        #   JSONResponse 会抛 "Object of type Fraction is not JSON serializable"
        #   —— 而且那句不在任何 try 里，整个 /api/solve 就变 500 了。
        #   界面要显示小数时读 param_values_float，两栏同时给，不靠前端解析分数。
        #   注意参数取值里**混着**分数与浮点：元件值本来就是 float（"R1 = 4700"），
        #   所以两个转换都要能接住 float，不能直接调 frac_float(v)。
        def _as_float(v: Any) -> Any:
            if isinstance(v, Fraction):
                return frac_float(v)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        param_values_float = {k: _as_float(v) for k, v in param_values.items()}
        param_values = {
            k: (frac_str(v) if isinstance(v, Fraction) else v)
            for k, v in param_values.items()
        }
    except Exception as e:                         # noqa: BLE001
        # 参数取值是**派生视图**，坏了不该毁掉已经算好的三法对账。
        # 但也不能装没发生 —— 显式留痕，界面上会看到这一条。
        param_values = {}
        param_values_float = {}
        failures["params"] = f"参数取值渲染失败：{type(e).__name__}: {e}"

    # ------------------------------------------------ 节点电压对账表
    node_rows = []
    for node in circuit.nodes:
        row: dict[str, Any] = {"node": node}
        ref_val = solutions["mna"].get_node(node)
        row["exact"] = frac_str(ref_val) if isinstance(ref_val, Fraction) else _fl(ref_val)
        row["exact_float"] = _fl(ref_val)
        worst = 0.0
        worst_against = None
        for k in METHODS:
            if k not in solutions:
                row[k] = None
                continue
            v = solutions[k].get_node(node)
            row[k] = (frac_str(v) if isinstance(v, Fraction) else round(_fl(v), 10))
            row[k + "_float"] = _fl(v)
            if k != "mna":
                d = abs(_fl(v) - _fl(ref_val))
                if d > worst:
                    worst, worst_against = d, k
        row["max_deviation"] = worst
        row["worst_method"] = worst_against
        row["consistent"] = worst <= max(ABS_TOL, REL_TOL * abs(row["exact_float"]))
        node_rows.append(row)

    # ------------------------------------------------ 支路电流对账表
    br_rows = []
    for c in circuit.components:
        row: dict[str, Any] = {"ref": c.ref, "kind": c.kind}
        ref_i = solutions["mna"].get_current(c.ref)
        row["exact"] = frac_str(ref_i) if isinstance(ref_i, Fraction) else _fl(ref_i)
        row["exact_float"] = _fl(ref_i)
        worst = 0.0
        for k in METHODS:
            if k not in solutions:
                row[k] = None
                continue
            iv = solutions[k].get_current(c.ref)
            row[k] = (frac_str(iv) if isinstance(iv, Fraction)
                      else (round(_fl(iv), 10) if iv is not None else None))
            if iv is not None and k != "mna":
                d = abs(_fl(iv) - _fl(ref_i))
                worst = max(worst, d)
        row["max_deviation"] = worst
        row["consistent"] = worst <= max(ABS_TOL, REL_TOL * max(abs(row["exact_float"]), 1e-30))
        br_rows.append(row)

    # ------------------------------------------------ 精确互校（前两法必须严格相等）
    exact_match = None
    exact_detail = ""
    if len(exact_keys) == 2:
        diffs = []
        for node in circuit.nodes:
            a = solutions["mna"].get_node(node)
            b = solutions["branch"].get_node(node)
            if a != b:
                diffs.append(f"V({node}): 节点电压法={frac_str(a)} 支路电流法={frac_str(b)}")
        for c in circuit.components:
            a = solutions["mna"].get_current(c.ref)
            b = solutions["branch"].get_current(c.ref)
            if a != b:
                diffs.append(f"i({c.ref}): 节点电压法={frac_str(a)} 支路电流法={frac_str(b)}")
        exact_match = not diffs
        exact_detail = (
            "两条精确路径逐项严格相等（差为 0，不是「足够小」）。"
            if exact_match else "逐项差集：" + "；".join(diffs)
        )
    else:
        exact_match = None
        exact_detail = f"精确互校不可用，仅 {len(exact_keys)} 条精确路径求解成功"

    # ------------------------------------------------ 验算
    verifications = {k: verify_all(circuit, s) for k, s in solutions.items()}

    # ------------------------------------------------ 叠加
    super_info: dict[str, Any]
    if want_superposition:
        try:
            super_info = superposition(circuit)
        except CircuitError as e:
            super_info = {"applicable": False, "reason": str(e)}
    else:
        super_info = {"applicable": False, "reason": "未请求"}

    # ------------------------------------------------ 结论
    all_consistent = all(r["consistent"] for r in node_rows) and \
        all(r["consistent"] for r in br_rows)
    all_verified = all(v["passed"] for v in verifications.values())
    missing = [k for k in METHODS if k not in solutions]

    overall = all_consistent and all_verified and not missing and exact_match is not False

    return {
        "methods_run": valid_keys,
        "failures": failures,
        "reduction": reduction,
        "circuit_after_reduction": circuit.to_dict() if reduction else None,
        "ngspice_availability": probe_availability(),
        "node_table": node_rows,
        "branch_table": br_rows,
        # ---- 参数体系：界面、报告、"哪个名字对应哪个量"都读这一份，
        #   不各自拼名字（拼法一旦和参数表的自动命名分家就会出现
        #   "报告里有个名字、参数表里没有"这种错位）。
        "params": params_view(original),
        # 按**参数名**索引：报告文本、"i_R1 等于多少"这类按名取值的场合读它。
        "param_values": param_values,
        "param_values_float": param_values_float,
        # ★ 按**绑定**索引：`"branch_i:R1" -> "1/20"`。
        #   界面读这一份而不是上面那份 —— 因为用户随时可以改名，
        #   而"改名"恰恰会让所有按名索引的表当场失效：界面上会显示成
        #   "这个参数还没有取值"，可电路没变、解也没变，变的只是名字。
        #   那是**假信息**，会让人以为自己画错了电路。
        #   绑定（哪条支路的电流、哪个结点的电压）是不随改名而变的，用它当键就没事。
        #   `params_view()` 的 `by_binder` 与这里的键是同一套写法（`kind:key`），
        #   一一对得上，界面拿 `item.binder` 拼一下就能查到值。
        "param_values_by_binder": _by_binder(original, param_values),
        "param_values_by_binder_float": _by_binder(original, param_values_float),
        "exact_crosscheck": {"ok": exact_match, "detail": exact_detail},
        "verifications": verifications,
        "superposition": super_info,
        "thresholds": {"rel": REL_TOL, "abs": ABS_TOL},
        "overall_pass": overall,
        "conclusion": _conclusion(overall, all_consistent, all_verified,
                                  exact_match, missing, failures),
        "solutions": {
            k: {
                "method": s.method,
                "node_voltages": {n: (frac_str(v) if isinstance(v, Fraction) else _fl(v))
                                  for n, v in s.node_voltages.items()},
                "branch_currents": {r: (frac_str(v) if isinstance(v, Fraction) else _fl(v))
                                    for r, v in s.branch_currents.items()},
                "detail": s.detail,
            } for k, s in solutions.items()
        },
    }


def _conclusion(overall, all_consistent, all_verified, exact_match, missing, failures) -> str:
    if overall:
        return ("三条独立代码路径结果一致，KCL/KVL 回代与功率守恒全部通过，"
                "前两法逐项严格相等 —— 这张电路的读图与建模可以采信。")
    bits = []
    if not all_verified:
        bits.append("有解未能通过 KCL/KVL 或功率守恒校核")
    if exact_match is False:
        bits.append("两条精确路径的结果不完全相等（说明它们至少有一条算错了）")
    if not all_consistent:
        bits.append("三条路径之间偏差超出阈值")
    if missing:
        detail = "；".join(f"{k}: {failures.get(k,'?')}" for k in missing)
        bits.append(f"缺失 {missing} 路对账（不允许当作通过）—— {detail}")
    # 结论文本会同时出现在纯文本报告和 WebUI 的卡片里 —— 后者是**纯文本渲染**，
    # 所以这里不用 Markdown 星号强调，改用「」。
    return "未通过全部校核：" + "；".join(bits) + "。"


def format_text_report(circuit: Circuit, pack: dict[str, Any]) -> str:
    """把对账包渲染成纯文本报告（给终端/日志用；WebUI 另有结构化渲染）。

    ★ 这个函数曾经把 ``/api/solve`` 打成 **HTTP 500**：它按**原电路**的节点名/
    位号去 ``next()`` 查**化简后**电路生成的对账表，而任何含 C/L 的电路里，
    被电感合并掉的节点名、被电容移除的支路位号在表中都不存在 → ``StopIteration``。
    实测：含 L 的电路 500、纯 R/V 的电路 200。

    所以现在主解一节**直接遍历对账表**（表内自带 ``node`` / ``ref``），不再反查；
    被化简移除的支路另列一行交代清楚 —— 报告"少了几条支路还不说明"同样是静默降级。
    """
    red = pack.get("reduction") or {}
    removed_why: dict[str, str] = {}
    for e in red.get("opened") or []:
        removed_why[e["ref"]] = "电容视为开路"
    for e in red.get("shorted") or []:
        cur = e.get("current")
        removed_why[e["ref"]] = (
            f"两端被理想短路路径短接，i = {cur}" if cur is not None
            else "两端被理想短路路径短接，i 待定"
        )

    L: list[str] = []
    L.append("=" * 72)
    L.append(f"电路：{circuit.name}")
    L.append(f"节点：{', '.join(circuit.nodes)}   参考节点：{circuit.ref_node}")
    L.append(f"元件：{len(circuit.components)} 个   支路数 b={len(circuit.components)}   "
             f"节点数 n={len(circuit.nodes)}   基本回路数 b−n+1="
             f"{max(0, len(circuit.components)-len(circuit.nodes)+1)}")
    L.append("=" * 72)

    L.append("\n【1】读图依据")
    for c in circuit.components:
        ev = c.evidence
        mark = f"   ← 直流化简后移除：{removed_why[c.ref]}" if c.ref in removed_why else ""
        L.append(f"  {c.ref:<5} {c.kind}  {c.nodes[0]} -> {c.nodes[1]}  "
                 f"值={c.value}  出处={ev.source}  置信度={ev.confidence:.2f}{mark}")
    if circuit.diagnostics:
        for d in circuit.diagnostics:
            L.append(f"  · {d}")

    # ---- 化简留痕（原题含 C/L 时才出现）。静默化简等于篡改题目，
    #      所以摘要与逐条明细都必须进报告，而不只是躺在返回包的 JSON 里。
    if red.get("applied"):
        L.append("\n【1b】直流稳态化简（原题含 C/L）")
        L.append(f"  {red.get('summary') or ''}")
        for e in red.get("opened") or []:
            L.append(f"  [开路] {e['ref']}  {e['nodes'][0]}-{e['nodes'][1]}"
                     f"  —— {e['reason']}")
        for e in red.get("shorted") or []:
            cur = e.get("current")
            line = (f"  [短接] {e['ref']}  {e['nodes'][0]}-{e['nodes'][1]}  "
                    f"i({e['ref']}) = {cur}" if cur is not None
                    else f"  [短接] {e['ref']}  {e['nodes'][0]}-{e['nodes'][1]}  "
                         f"i({e['ref']}) 待定")
            line += f"  —— {e['reason']}"
            if e.get("current_method"):
                line += f"（依据：{e['current_method']}）"
            L.append(line)
            if cur is None and e.get("current_note"):
                L.append(f"         为什么不定：{e['current_note']}")
        for note in red.get("notes") or []:
            L.append(f"  · {note}")

    L.append("\n【2】约定")
    from .base import declared_direction
    for c in circuit.components:
        if c.ref in removed_why:
            continue
        f, t = declared_direction(c)
        extra = "（电源内部由 − 流向 +，故为正 = 该电源在供电）" if c.kind == "V" else ""
        L.append(f"  i({c.ref}) 参考方向：{f} -> {t}{extra}")

    L.append("\n【3】主解（节点电压法，精确有理数）")
    for row in pack["node_table"]:
        L.append(f"  V({row['node']}) = {row['exact']}")
    for row in pack["branch_table"]:
        L.append(f"  i({row['ref']}) = {row['exact']}")
    if removed_why:
        L.append("  注：以下支路在直流化简后被移除，不在上表内 —— "
                 + "；".join(f"{r}（{removed_why[r]}）" for r in removed_why))

    L.append("\n【4】验算")
    for k, v in pack["verifications"].items():
        for chk in v["checks"]:
            flag = "通过" if chk["ok"] else "不通过"
            L.append(f"  [{flag}] {v['method']} · {chk['name']}")
    pw = pack["verifications"]["mna"]["checks"][0]
    L.append("  功率表：")
    for r in pw["rows"]:
        L.append(f"    {r['ref']:<5} P={r['power']:<16} ({r['role']})")
    L.append(f"  {pw['conclusion']}")

    L.append("\n【5】三法对账表")
    hdr = f"  {'量':<10}{'节点电压法':<18}{'支路电流法':<18}{'ngspice':<18}{'最大偏差'}"
    L.append(hdr)
    for r in pack["node_table"]:
        L.append(f"  V({r['node']})" .ljust(10) +
                 f"{str(r.get('mna')):<18}{str(r.get('branch')):<18}"
                 f"{str(r.get('ngspice')):<18}{r['max_deviation']:.3g}")
    L.append(f"  精确互校：{'通过' if pack['exact_crosscheck']['ok'] else '未通过'} —— "
             f"{pack['exact_crosscheck']['detail']}")
    for k in METHOD_KEYS:
        if k in pack["failures"]:
            L.append(f"  [缺失] {k} 路对账失败：{pack['failures'][k]}")

    if pack["superposition"].get("applicable"):
        L.append("\n【6】叠加原理复核")
        sup = pack["superposition"]
        # 单独立源时 superposition() 走退化分支，不产出 conclusion，只有 note
        concl = sup.get("conclusion") or sup.get("note") or "无可比对信息。"
        L.append(f"  {concl}")
        if sup.get("conclusion") and sup.get("note"):
            L.append(f"  注意：{sup['note']}")
        for s in sup.get("sources") or []:
            L.append(f"  独立源：{s}")
        if sup.get("rows"):
            L.append(f"  逐源对账最大偏差：{sup['max_deviation']:.3g}"
                     f" —— {'通过' if sup.get('ok') else '未通过'}")

    L.append("\n【结论】" + pack["conclusion"])
    return "\n".join(L)


METHOD_KEYS = ("mna", "branch", "ngspice")
