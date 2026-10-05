"""直流稳态化简：把 C 视为开路、L 视为短路。

**为什么必须有这一步**：真实电路（尤其 KiCad 里的实际工程）到处都是退耦电容。
如果一见到 C 就报"不支持"，这个工具在实际原理图上完全不能用。
而 C/L 在直流稳态下的处理是**教材标准动作**，不是我发明的近似：

- **电容 → 开路**：直流稳态下 i_C = C·du/dt = 0，电容支路断开
- **电感 → 短路**：直流稳态下 u_L = L·di/dt = 0，电感两端等电位，节点合并

★ 这一步会**改变节点集合**（电感合并节点、电容拆除支路导致孤立节点），
所以化简过程必须留痕并写进报告 —— 用户看到的解是化简后电路的解，
他有权知道中间做了什么变换。**静默化简等于篡改题目。**

★ 化简后若出现孤立节点，那不是错误，是**有意义的结论**：
说明该节点与其余部分之间只剩电容相连，直流稳态下它悬浮（这正是"隔直电容"的作用）。
报告里会显式说明，而不是报一个"图不连通"的错就完事。

======================================================================
「两端落进同一个节点」的元件，五种处理各不相同 —— 这里是最容易出错的地方
======================================================================

电感合并节点后，原本跨在这两个节点上的元件，两端就变成了同一个节点。它们的
**支路方程给出的信息量完全不同**，绝不能一概当成"被短接、等价于消失"删掉：

==================  ==================  ================================
元件                支路方程            两端等电位（u = 0）后的结论
==================  ==================  ================================
``R``               u = R·i             R ≠ 0 ⇒ **i = 0**（信息确定）
``I``               i = I_s             i 由源本身给定，与 u 无关 ⇒ **i = I_s**
``V`` (E ≠ 0)       u = E               0 = E ⇒ **约束矛盾，整张电路无解**
``V`` (E = 0)       u = E = 0           0 = 0，给不出 i ⇒ i 待定
``L``               u = 0（恒成立）      无信息 ⇒ i 待定
==================  ==================  ================================

"i 待定"不是没办法 —— 见 :func:`recover_shorted_currents`：用 KCL 在**合并前的
原节点**上把它反算出来。学生要的往往正是 ``i_L``，报告里少一条支路还不说明，
就是静默丢信息。反算确实不唯一时（纯理想短路回路）明说"电流不定"，不许猜。

★ 血泪案例（本条规则就是为它写的）：``V1(10V, 1-0)`` 与 ``L1(1-0)`` 并联，
再串 ``R1(1-2)``、``R2(2-0)``。旧的化简把 V1 当"被短接即消失"删掉，
剩下的电路照常可解 → 三条路径一致地给出全零解 → 功率守恒平凡成立 →
``overall_pass = True``，并输出"这张电路的读图与建模**可以采信**"。
而 ngspice 独立复核给出 ``singular matrix: check node l1#branch``，
Dynamic/True gmin stepping 与 source stepping 全部失败 —— **该电路确实无解**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..ir.model import (
    Circuit,
    CircuitError,
    CircuitUnsatisfiable,
    Component,
)
from .base import declared_direction
from .linalg import frac_str, to_frac

__all__ = [
    "ReductionReport",
    "reduce_to_dc",
    "recover_shorted_currents",
]


class _UF:
    """并查集。**参考节点的代表固定为参考节点本身**（地不改名）。

    不改名这件事比看起来重要：若 ``L1(1-0)`` 让合并后的代表元取到 ``"1"``，
    化简后的电路里"节点 1"其实是地，报告和插图都会变得极难读，
    而且 ``ref_node`` 会在电路内部被悄悄换掉。
    """

    def __init__(self, keys, *, prefer: str | None = None):
        self.p = {k: k for k in keys}
        self.prefer = prefer

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.prefer is not None and rb == self.prefer:
            ra, rb = rb, ra          # 让参考节点的根当新根
        self.p[rb] = ra


@dataclass
class ReductionReport:
    """化简留痕。

    ★ **不同原因的移除必须分开存放。** 早先把"电容开路移除"和"被短路而移除"
    塞进同一个 ``removed`` 列表，``summary()`` 又整列冠以"电容视为开路"，
    于是**没有电容的电路**会输出"电容视为开路，移除 2 条支路（L1, R1)"。
    报告文本张冠李戴比不做报告更坏：它让学生以为程序看懂了。
    """

    applied: bool = False

    #: 电容 → 开路而移除的支路
    opened: list[dict[str, Any]] = field(default_factory=list)
    #: 两端被理想短路路径合一而移除的支路。每条都带 ``current``：
    #: 已确定的写成字符串（如 ``"0"``），确实不唯一的写 ``None`` 并附 ``current_note``。
    shorted: list[dict[str, Any]] = field(default_factory=list)
    #: 电感短路造成的节点合并组。**每组第一个元素是该组的代表元**（保留的节点名）
    merged_groups: list[list[str]] = field(default_factory=list)
    #: 被合并掉、不再单独出现的原节点名。
    #: ★ 这与"孤立节点"是两回事 —— 这些点**依然存在**，只是与代表元等电位。
    merged_away: list[str] = field(default_factory=list)
    #: 真正的孤立节点：化简后不属于任何支路
    isolated_nodes: list[str] = field(default_factory=list)
    #: 约束矛盾（理想电压源被短路等）。出现任意一条即整张电路无解。
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def indeterminate(self) -> list[dict[str, Any]]:
        """被移除但电流**无法由直流稳态唯一确定**的支路。"""
        return [e for e in self.shorted if e.get("current") is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "opened": self.opened,
            "shorted": self.shorted,
            "merged_groups": self.merged_groups,
            "merged_away": self.merged_away,
            "isolated_nodes": self.isolated_nodes,
            "indeterminate": self.indeterminate,
            "contradictions": self.contradictions,
            "notes": self.notes,
        }

    def summary(self) -> str:
        """一句（分号连接的）人类可读摘要。**只陈述真实发生过的变换。**"""
        if not self.applied:
            return "未做直流稳态化简（电路里没有 C/L）。"
        bits: list[str] = []

        if self.opened:
            refs = ", ".join(e["ref"] for e in self.opened)
            bits.append(f"电容视为开路，移除 {len(self.opened)} 条支路（{refs}）")

        if self.merged_groups:
            gs = "; ".join("≡".join(g) + f"（以 {g[0]} 为准）" for g in self.merged_groups)
            bits.append(f"电感视为短路，合并节点组：{gs}")

        if self.shorted:
            parts = []
            for e in self.shorted:
                cur = e.get("current")
                parts.append(f"{e['ref']}（i = {cur}）" if cur is not None
                             else f"{e['ref']}（i 待定）")
            bits.append(
                f"两端被理想短路路径短接而移除 {len(self.shorted)} 条支路："
                + ", ".join(parts)
            )

        if self.merged_away:
            bits.append(
                f"被合并掉的节点名：{self.merged_away}"
                "（这些点仍然存在，只是与同组的代表元等电位，报告里不再单独列出）"
            )

        if self.isolated_nodes:
            bits.append(
                f"化简后孤立节点：{self.isolated_nodes}"
                "（与其余部分之间只剩电容相连，直流稳态下没有电流 —— "
                "这是正确结论，不是错误）"
            )

        ind = self.indeterminate
        if ind:
            bits.append(
                "电流不唯一的支路："
                + ", ".join(e["ref"] for e in ind)
                + "（处于纯理想 0 Ω 回路中，直流稳态下 u = 0 对电流不构成约束）"
            )

        if self.contradictions:
            bits.append("约束矛盾：" + ", ".join(c["ref"] for c in self.contradictions))

        return "直流稳态化简：" + "；".join(bits) + "。"


# ---------------------------------------------------------------- 单条支路定性


def _classify_shorted(c: Component, merged_into: str, report: ReductionReport) -> None:
    """一条支路的两端落进同一个代表元 → 它已被理想短路路径短接。

    逐种元件按上表定性；``V`` 且 E ≠ 0 的记进 ``contradictions``（稍后抛无解）。
    """
    entry: dict[str, Any] = {
        "ref": c.ref,
        "kind": c.kind,
        "nodes": list(c.nodes),
        "value": c.value,
        "merged_into": merged_into,
        "current": None,
    }

    if c.kind == "R":
        entry["reason"] = "与理想短路路径并联；两端等电位 ⇒ u = 0 ⇒ i = 0"
        entry["current"] = "0"
        entry["current_method"] = "元件方程 u = R·i，u = 0 且 R ≠ 0"

    elif c.kind == "I":
        entry["reason"] = (
            "被理想短路路径短接；理想电流源的电流由源本身给定，与两端电压无关"
        )
        if c.value is None:
            entry["current_note"] = "该电流源缺数值，电流无法确定"
        else:
            entry["current"] = frac_str(to_frac(c.value))
            entry["current_method"] = "理想电流源的定义 i = I_s"

    elif c.kind == "V":
        val = None if c.value is None else to_frac(c.value)
        if val is None or val != 0:
            entry["reason"] = "理想电压源被理想短路路径短接"
            report.contradictions.append({
                "kind": "ideal_voltage_source_shorted",
                "ref": c.ref,
                "nodes": list(c.nodes),
                "value": c.value,
                "merged_into": merged_into,
            })
        else:
            entry["reason"] = (
                "0 V 电压源在直流下等价于导线；u = E = 0 给不出电流信息"
            )

    elif c.kind == "L":
        entry["reason"] = (
            "电感本身就是直流短路路径；u_L = L·di/dt = 0 给不出电流信息"
        )

    else:
        # ★ 绝不靠"剩下的一定是电感"来兜底。受控源（E/G/H/F）一旦走到这里，
        #   就会被套上一句"电感本身是短路路径"的解释 —— 而它根本不是电感，
        #   报告里的理由就是编的。宁可在这里炸掉。
        #   （正常路径下到不了：reduce_to_dc 见到"受控源 + C/L"会先明确报不支持。）
        raise CircuitError(
            f"{c.ref}: 化简层不认识元件类型 {c.kind} 落在同一节点上的情形，"
            "无法判断它是否已被理想短路路径短接（内部一致性错误，不应触发）")

    report.shorted.append(entry)


def _contradiction_text(cs: list[dict[str, Any]]) -> str:
    lines = ["这张电路在直流稳态下无解（约束集自相矛盾）："]
    for i, c in enumerate(cs, 1):
        a, b = c["nodes"]
        lines.append(
            f"  {i}. 理想电压源 {c['ref']}（{c['value']} V，{a} → {b}）的两端被理想"
            f"短路路径短接（合并到节点 {c['merged_into']}）。"
            f"电路同时要求 V({a}) − V({b}) = {c['value']} V 与 V({a}) = V({b})，"
            f"二者不能同时成立。"
        )
    lines += [
        "",
        "为什么会这样：直流稳态下电感是理想短路（u_L = L·di/dt = 0），"
        "而理想电压源不允许被短路 —— 那要求它输出无穷大电流。",
        "怎么办：",
        "  · 若题图里的线圈有直流电阻，请把它作为串联电阻画出来（现实中线圈总有电阻）；",
        "  · 若原题求的是 u_L(t) / i_L(t) 这类暂态量，本工具只解直流工作点，"
        "请先判断题目意图再决定是否保留该电感；",
        "  · 也请顺带核对这条短路线是否真的画到了电压源两端 —— "
        "读图错误同样会造出这个矛盾。",
        "",
        "★ 注意：这不是「算不出来」，而是这张理想化电路本身没有解。"
        "本工具不会为了给出一个答案而删掉那条电压源。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程


def reduce_to_dc(circuit: Circuit, *, require_solvable: bool = True):
    """返回 ``(化简后的电路, ReductionReport)``。

    ``require_solvable=False`` 时即使化简后仍不连通也照样返回（供界面展示用）。

    遇约束矛盾（如电感把理想电压源短路）抛 :class:`CircuitUnsatisfiable`，
    **绝不**把它当成"某条元件被短接就删掉"糊过去。
    """
    comps = circuit.components
    has_c = any(c.kind == "C" for c in comps)
    has_l = any(c.kind == "L" for c in comps)
    report = ReductionReport()

    # ★★ 受控源 + 储能元件：**明确报不支持，不许硬算**。
    #
    #   化简发生在三条求解路径**之前**，所以这里的任何差错都拦不住：
    #   三法互校 / 跨实现对账 / 功率守恒，大家解的都是这张被化简过的电路，
    #   会一致地给出同一个错答案（C→开路、L→短路那次事故就是这么发生的）。
    #   而受控源正好是最怕这一点的元件：它的**控制量**可能就落在被移除的
    #   电容支路或与电感合并掉的那对节点上 —— 化简后那个量已经不存在了，
    #   却仍会被当成一个"零"或"某个别的量"参与进来，算出一个看着正常的解。
    #   在把"受控源的控制量在化简下如何变换"想清楚之前，宁可不做。
    controlled = [c.ref for c in comps if c.is_controlled]
    storages = [c.ref for c in comps if c.kind in ("C", "L")]
    if controlled and storages:
        raise CircuitError(
            f"本题同时含受控源（{', '.join(controlled)}）与储能元件"
            f"（{', '.join(storages)}），本版本暂不支持求解这种组合。"
            "原因是直流稳态化简发生在三条求解路径之前：电容被视为开路、"
            "电感被视为短路之后，受控源的**控制量**可能正好落在被移除或被合并的"
            "那条支路上，而化简后的电路仍然算得出一个「看起来正常」的解 ——"
            "三法互校与跨实现对账都发现不了。"
            "请按直流稳态手算化简（电容开路、电感短路）后再输入，"
            "或改画成不含受控源的等效电路。"
        )

    if not has_c and not has_l:
        return circuit.copy(), report

    report.applied = True

    # ---- 1. 电感短路 -> 节点合并
    uf = _UF(circuit.nodes, prefer=circuit.ref_node)
    for c in comps:
        if c.kind == "L":
            uf.union(*c.nodes)

    def rep(n: str) -> str:
        return uf.find(n)

    groups: dict[str, list[str]] = {}
    for n in circuit.nodes:
        groups.setdefault(rep(n), []).append(n)
    # 代表元排在首位，"2≡3" 读作"以 2 为准"
    report.merged_groups = [
        [r] + sorted(x for x in members if x != r)
        for r, members in sorted(groups.items())
        if len(members) > 1
    ]
    report.merged_away = sorted(n for n in circuit.nodes if rep(n) != n)
    if report.merged_groups:
        report.notes.append(
            "电感视为短路 ⇒ 两端节点等电位，已合并。被合并掉的节点名 "
            f"{report.merged_away} 并不是消失了，而是与同组代表元等电位，"
            "报告里以代表元的名字出现。"
        )

    # ---- 2. 电容开路 -> 移除支路
    kept: list[Component] = []
    for c in comps:
        if c.kind == "C":
            report.opened.append({
                "ref": c.ref, "kind": "C", "nodes": list(c.nodes), "value": c.value,
                "reason": "直流稳态下 i_C = C·du/dt = 0，视为开路",
            })
            continue
        kept.append(c)

    # ---- 3. 重写节点名；两端落进同一代表元的支路已被理想短路路径短接
    new_ref = rep(circuit.ref_node)
    rewritten: list[Component] = []
    for c in kept:
        a, b = rep(c.nodes[0]), rep(c.nodes[1])
        if a != b:
            rewritten.append(Component(
                ref=c.ref, kind=c.kind, nodes=(a, b), value=c.value,
                evidence=c.evidence, geom=c.geom, note=c.note,
                # ★ ctrl 必须一起带上。漏掉它 = 受控源悄悄退化成"独立源"，
                #   而网表照样生成、解照样算得出来，只是把它按 0 处理了。
                ctrl=c.ctrl,
            ))
            continue
        _classify_shorted(c, a, report)

    # ---- 4. 约束矛盾优先：先报无解，再谈别的
    if report.contradictions:
        raise CircuitUnsatisfiable(
            _contradiction_text(report.contradictions), report.contradictions
        )

    if not rewritten:
        raise CircuitError(
            "直流稳态化简后没有任何剩余元件：原电路的每一条支路都在直流稳态下"
            "退化了（电容开路、电感短路、被短路的理想元件被移除）。"
            "这类电路不是「无解」，但化简后已没有可供求解的电阻网络。"
            "请检查是否漏画了元件（例如电感回路里通常还有电阻）。"
        )

    present = {n for c in rewritten for n in c.nodes}
    if new_ref not in present:
        raise CircuitError(
            f"直流稳态化简后参考节点 {new_ref!r} 不再接任何支路"
            "（它原来只经电容与电路相连）。请把参考节点改到一个直流上确实"
            "有通路的节点，或补上被省略的直流通路。"
        )

    # ---- 5. 孤立节点（≠ 被合并掉的节点）
    #      ★ 必须用**合并后的代表元**集合去比：rewritten 里的节点名已被换成代表元，
    #        若拿它与原节点名相减，被合并掉的原节点名会全部被误报成"孤立节点"。
    reps_all = {rep(n) for n in circuit.nodes}
    report.isolated_nodes = sorted(reps_all - present - {new_ref})

    out = Circuit(
        name=circuit.name, components=rewritten, ref_node=new_ref,
        probes=circuit.probes,
        diagnostics=list(circuit.diagnostics) + [{
            "kind": "dc_reduction", "text": report.summary(),
        }],
        origin={**circuit.origin, "dc_reduced": True},
    )

    # ---- 6. 连通性检查
    if require_solvable:
        adj: dict[str, set[str]] = {n: set() for n in out.nodes}
        for c in rewritten:
            adj[c.nodes[0]].add(c.nodes[1])
            adj[c.nodes[1]].add(c.nodes[0])
        seen = {new_ref}
        stack = [new_ref]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        island = [n for n in out.nodes if n not in seen]
        if island:
            raise CircuitError(
                f"直流稳态化简后电路不连通，以下节点与参考节点 {new_ref} 之间"
                f"没有直流通路：{island}。若这是电容隔直造成的，那本身是正确结论"
                "（该支路没有直流电流），但本版本还不会自动分段求解 —— "
                "请把不需要的部分删掉后重试。"
            )

    return out, report


# ---------------------------------------------------------------- 反算被短路支路的电流


def _rref_solve(
    A: list[list[Fraction]], b: list[Fraction], ncols: int
) -> tuple[str, list[Fraction] | None]:
    """精确有理数高斯-约当，返回 ``("unique"|"underdetermined"|"inconsistent", 解)``。

    ★ 不能直接用 :func:`app.solver.linalg.solve_linear`：那里**奇异就抛异常**，
    而这里"奇异"是有物理含义的正常结论 —— 纯理想 0 Ω 回路里电流本来就不唯一。
    必须区分两种情况，因为报告要说的话完全不同：

    - ``underdetermined`` → "电流不定"（电路的固有性质，不是错）
    - ``inconsistent``    → 化简层或求解层有 bug（必须报警，不许当"不定"糊过去）
    """
    rows = len(A)
    if ncols == 0:
        return "unique", []
    if rows == 0:
        return "underdetermined", None

    M = [[to_frac(v) for v in row] + [to_frac(b[i])] for i, row in enumerate(A)]
    piv: list[int] = []
    r = 0
    for col in range(ncols):
        p = next((rr for rr in range(r, rows) if M[rr][col] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][col]
        M[r] = [v / pv for v in M[r]]
        for rr in range(rows):
            if rr != r and M[rr][col] != 0:
                f = M[rr][col]
                M[rr] = [M[rr][c] - f * M[r][c] for c in range(ncols + 1)]
        piv.append(col)
        r += 1
        if r == rows:
            break

    for rr in range(r, rows):
        if M[rr][ncols] != 0 and all(M[rr][c] == 0 for c in range(ncols)):
            return "inconsistent", None
    if len(piv) < ncols:
        return "underdetermined", None

    x: list[Fraction] = [Fraction(0)] * ncols
    for i, c in enumerate(piv):
        x[c] = M[i][ncols]
    return "unique", x


def recover_shorted_currents(
    original: Circuit,
    reduced: Circuit,
    report: ReductionReport,
    sol: Any,
) -> None:
    """用 KCL 在**合并前的原节点**上反算"被短路移除"支路的电流，就地写进 ``report``。

    为什么非做不可：电感在直流下是短路，被判成"节点合并"而从支路表里消失，
    可**学生要的往往正是 i_L**。报告里少一条支路还不说明，就是静默丢信息。

    已经在 :func:`_classify_shorted` 里由**元件方程**定下的（``R`` 的 i = 0、
    ``I`` 的 i = I_s）不再参与反算，只有"元件方程给不出信息"的（电感、0 V 电压源）
    才作为未知量，用合并前每个原节点的 KCL 解出来：

    - 未知量 = 这些支路的电流；
    - 已知量 = 化简后每条已解出支路在各原节点上的注入（正负按
      :func:`~app.solver.base.declared_direction`，**用原件的方向** ——
      节点重写只换了名字，没有换 ``nodes`` 次序，所以方向与原来一致）；
    - 方程 = 每个涉及到的原节点上"流出之和 = 0"。

    这是**纯后处理**：只往报告里补数字，不改动化简后的电路，也不改任何已解出的量，
    所以它不可能把对的答案改错。反算不唯一时（纯理想短路回路）明说"电流不定"；
    反算不相容时（说明化简层或求解层有 bug）显式报警 —— 两种情况都不猜。
    """
    if not report.shorted:
        return

    orig_by_ref = {c.ref: c for c in original.components}

    def _give_up(msg: str) -> None:
        for e in report.shorted:
            if e.get("current") is None:
                e["current_note"] = msg

    if sol is None:
        _give_up("主解缺失，未做反算")
        return

    # ---- 已知量：化简后已解出的支路在原节点上的注入
    known: dict[str, Fraction] = {n: Fraction(0) for n in original.nodes}
    missing: list[str] = []
    for c in reduced.components:
        oc = orig_by_ref.get(c.ref)
        i = sol.get_current(c.ref) if oc is not None else None
        if oc is None or i is None:
            missing.append(c.ref)
            continue
        iv = to_frac(i)
        f, t = declared_direction(oc)          # ★ 用原件的方向
        known[f] += iv
        known[t] -= iv

    if missing:
        _give_up(f"化简后支路 {missing} 的电流不可用，未做反算")
        report.notes.append(
            f"被短路移除的支路电流未能反算：化简后支路 {missing} 的电流不可用。"
        )
        return

    # ---- 未知量：元件方程给不出信息的那几条
    unk = [e["ref"] for e in report.shorted if e.get("current") is None]
    if not unk:
        return
    col = {r: k for k, r in enumerate(unk)}

    involved = sorted({
        n
        for e in report.shorted
        if e.get("current") is None
        for n in e["nodes"]
    })
    A: list[list[Fraction]] = []
    b: list[Fraction] = []
    for n in involved:
        row = [Fraction(0)] * len(unk)
        for e in report.shorted:
            r = col.get(e["ref"])
            if r is None:
                continue
            f, t = declared_direction(orig_by_ref[e["ref"]])
            if n == f:
                row[r] += 1
            elif n == t:
                row[r] -= 1
        A.append(row)
        b.append(-known[n])

    status, values = _rref_solve(A, b, len(unk))

    if status == "unique":
        by_ref = {e["ref"]: e for e in report.shorted}
        for r, v in zip(unk, values or []):
            by_ref[r]["current"] = frac_str(v)
            by_ref[r]["current_method"] = (
                "由 KCL 在合并前的原节点上反算（精确有理数）—— "
                "与三条解法路径无关，是化简层的独立后处理"
            )
        report.notes.append(
            "被短路移除支路的电流已由原节点 KCL 反算补全："
            + ", ".join(f"{r} = {by_ref[r]['current']}" for r in unk)
            + "。这些是化简前支路的电流，报告中可见的解均为化简后电路的解。"
        )

    elif status == "underdetermined":
        _give_up(
            "该支路处于纯理想 0 Ω 回路中：直流稳态下 u = 0 对电流不构成约束，"
            "需要线圈/导线的实际电阻或暂态信息才能确定"
        )
        report.notes.append(
            "有支路（" + ", ".join(unk) + "）的电流在直流稳态下不唯一 —— "
            "它们构成纯理想（0 Ω）回路，u = 0 给不出电流信息。"
            "这是电路的固有性质，不是算错；报告里标为「i 待定」，不予猜测。"
        )

    else:
        _give_up("反算方程与原节点 KCL 不相容，未给出数值")
        report.notes.append(
            "★ 反算 KCL 与原节点不相容 —— 正确的化简不会出现这种情况，"
            "它表明化简层或求解层有缺陷。请把这张电路和此报告一起反馈。"
        )
