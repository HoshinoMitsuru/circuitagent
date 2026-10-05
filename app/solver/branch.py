"""支路电流法（独立复算路径）。

未知量 = **全部支路电流**（电流源/电流型受控源的电流不自由，所以改用它的电压作未知量）；
方程 = (n−1) 条结点 KCL + **基本回路** KVL。

基本回路从**生成树**导出：每条非树边（弦）对应恰好一个回路，
条数恒为 ``b − n + 1``，**天生线性无关** —— 不用自己挑回路，
也就不会像手挑网孔那样挑到两条相关的。

★ KVL 的符号只按"压降"一条规则走：``drop(a→c) = V_a − V_c``。
电阻沿参考方向是 ``+R·i``；电压源从 + 走到 − 是 ``+E``。
**绝不混用"压降/电势升"两套说法** —— 技能文档里记录过一次真实事故：
混用之后出现"KCL 全对、KVL 差 250V"这种看起来很玄的错误。

为什么这条路径真的独立于 MNA：未知量是**支路电流**而不是节点电压，
方程是**回路 KVL** 而不是电压源约束行，矩阵的维度、稀疏结构、奇异条件
全都不一样。两者的共性错误（比如同一处符号约定写反）概率远低于
"同一份代码跑两遍"。

## 受控源在这里怎么进方程

受控源的量要么含"节点电压差"、要么含"别处支路电流"，两者都不是本方法的
未知量，**必须换基底**。做法是把每一个量都表示成**未知量列的线性组合**：

* 支路压降 ``drop_c`` 与支路电流 ``i_c`` 各有一个"系数表 + 常数"的表达式；
* **节点电压差 ``V_x − V_y``** 沿生成树的**唯一路径**累加压降即可
  （树路径唯一，所以这个表示是确定的，不依赖挑哪条回路）；
* 受控源的控制量、以及电压型受控源给出的压降 ``−控制量``，
  都在这个基底上展开。

★ 关键在于：**这些系数表在求解之前就完全确定**（它们只是"未知量的线性
组合"，与未知量的取值无关）。所以解出 ``sol_vec`` 之后直接代入即可 ——
不需要"先知道节点电压才能算受控源压降"这种循环。整个流程单向，没有不动点迭代。

★★ 展开深度有上限（``MAX_EXPR_DEPTH``）。受控源互相控制、或控制量绕回自己，
会构成代数环；超过深度就**明确报错**，而不是让递归撞上 Python 的栈上限
（那时用户看到的是 RecursionError，完全指不到真正的问题）。
"""

from __future__ import annotations

from fractions import Fraction

from ..ir.model import Circuit, CircuitError
from ..ir.params import LinearExpr
from .base import (Solution, declared_direction,
                   build_node_voltages_from_drops, reject_non_dc)
from .controlled import ensure_sense_sources
from ..ir.params import MAX_EXPR_DEPTH
from .linalg import solve_linear, to_frac


# ---------------------------------------------------------------- 图工具


def _build_edges(circuit: Circuit) -> list[tuple[int, int, str]]:
    """把元件表变成边表 ``(u, v, ref)``，u/v 是节点下标。

    平行边（同一对节点之间的多个元件）必须各自成边 —— 用"节点对"去重
    会悄悄吃掉一个元件，这是位图解析最容易踩的坑之一。
    """
    node_index = {n: i for i, n in enumerate(circuit.nodes)}
    edges: list[tuple[int, int, str]] = []
    for c in circuit.components:
        f, t = declared_direction(c)
        edges.append((node_index[f], node_index[t], c.ref))
    return edges


def _spanning_tree(n_nodes: int, edges: list[tuple[int, int, str]], root: int):
    """BFS 生成树，返回 (树边下标集合, 弦边下标集合, 父边表)。"""
    adj: dict[int, list[tuple[int, int]]] = {}
    for i, (u, v, _) in enumerate(edges):
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))

    tree: set[int] = set()
    parent: dict[int, tuple[int, int]] = {}     # 子节点 -> (父节点, 边下标)
    seen = {root}
    queue = [root]
    while queue:
        x = queue.pop(0)
        for y, ei in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                tree.add(ei)
                parent[y] = (x, ei)
                queue.append(y)
    if len(seen) != n_nodes:
        raise CircuitError(
            f"支路电流法：图不连通（连通 {len(seen)}/{n_nodes} 个节点）。"
            "位图解析漏线时最常见。"
        )
    chords = [i for i in range(len(edges)) if i not in tree]
    return tree, chords, parent


def _tree_path(parent: dict[int, tuple[int, int]], a: int, b: int):
    """树中 a -> b 的路径，返回 ``[(起点, 终点, 边下标), ...]``。

    ★ 这里踩过一次坑，写下来：早先的写法用
    ``seg = chain(x)[:idx_lca]`` 再逐个 ``parent[]`` 取边，
    结果第一条边取成了 ``(a, a, parent[a])`` —— 一个**自环**，
    给 KVL 方程塞进一个多余的 ``−drop`` 项。
    症状是：KCL 全对、KVL 全错，而且只在"a 不是 LCA 且离根较深"的图上发作，
    简单的链式电路一点事没有。电桥（有环）才把它逼出来。
    """
    def chain(x: int) -> list[int]:
        c = [x]
        while x in parent:
            x = parent[x][0]
            c.append(x)
        return c

    ca, cb = chain(a), chain(b)
    setb = {n: i for i, n in enumerate(cb)}
    lca = next(n for n in ca if n in setb)
    ia, ib = ca.index(lca), setb[lca]

    out: list[tuple[int, int, int]] = []
    # a -> LCA：沿 ca 逐段向下走
    for i in range(ia):
        u, v = ca[i], ca[i + 1]
        out.append((u, v, parent[u][1]))
    # LCA -> b：cb 存的是 b -> LCA，倒过来走
    for i in range(ib - 1, -1, -1):
        u, v = cb[i + 1], cb[i]
        out.append((u, v, parent[cb[i]][1]))
    return out


# ---------------------------------------------------------------- 主解


def fundamental_loops(circuit: Circuit):
    """导出全部基本回路，每条是 ``(弦边位号, [(起点下标, 终点下标, 边下标), ...])``。

    ★ 返回前**强制自证闭合**：相邻段的"终点"必须等于下一段的"起点"，
    最后一段的终点必须回到第一段的起点。不闭合就直接抛错，
    绝不把一条断开的"回路"交给 KVL —— 否则残差会是个看着挺像样的非零数，
    让人以为是电路算错了，而不是回路构造错了（本项目真踩过这个坑）。
    """
    nodes = circuit.nodes
    root = nodes.index(circuit.ref_node)
    edges = _build_edges(circuit)
    _, chords, parent = _spanning_tree(len(nodes), edges, root)

    loops = []
    for ci in chords:
        u, v, ref = edges[ci]
        walk = _tree_path(parent, u, v) + [(v, u, ci)]
        # 闭合自证
        for i in range(len(walk) - 1):
            if walk[i][1] != walk[i + 1][0]:
                raise CircuitError(
                    f"基本回路构造失败（回路{ref}）：第 {i} 段终点 {walk[i][1]} "
                    f"≠ 第 {i+1} 段起点 {walk[i+1][0]}，回路断开。"
                    "这是内部一致性错误，不应触发。"
                )
        if walk and walk[-1][1] != walk[0][0]:
            raise CircuitError(
                f"基本回路构造失败（回路{ref}）：末段未回到起点 "
                f"（{walk[-1][1]} ≠ {walk[0][0]}）。"
            )
        loops.append((ref, walk))
    return loops


def _declared_drop(c, val: Fraction) -> Fraction | None:
    """沿 declared_direction 的**恒定**压降（与支路电流无关的那部分）。

    只有电压源有恒定压降：declared_direction(电压源) = (nodes[1], nodes[0]) = (−, +)，
    所以沿参考方向的压降是 ``V_− − V_+ = −E``。
    电阻的压降是 ``R·i``（随电流变），电流源的压降本身是未知量 —— 两者都返回 None。

    ★ 这个函数是"**独立电压源**压降符号"的唯一出口。KVL 回路方程与由电流反推压降
    两处都调它，就不会出现"两处各写一遍、其中一处写反"的事故。
    （受控源走 ``_Base`` 里更一般的 ``drop_terms``，因为它的压降不是常数。）
    """
    if c.kind == "V":
        return -val
    return None


class _Base:
    """把"支路电流法的各个量"都表达成**未知量列的线性组合**。

    ★ 为什么单独一个类：这几组表达式互相引用（节点电压差要压降、
    受控源压降要节点电压差、支路电流又要受控源），递进关系容易绕晕。
    集中到一处 + 一个显式的深度上限，比散在装配循环里安全得多。
    """

    def __init__(self, circuit: Circuit, edges, parent, unk, cols, n_nodes):
        self.circuit = circuit
        self.edges = edges
        self.parent = parent
        self.unk = unk
        self.cols = cols
        self.n_nodes = n_nodes
        self.known = {c.ref: c for c in circuit.components}
        self._delta_cache: dict[tuple[int, int], tuple[dict[int, Fraction], Fraction]] = {}

    # ---- 基础量的表达

    def current_terms(self, c, depth: int = 0):
        """``i_c`` = 未知量线性组合 + 常数。"""
        if c.kind in ("R",) or c.outputs_voltage:
            return {self.unk[c.ref]: Fraction(1)}, Fraction(0)
        if c.kind == "I":
            return {}, to_frac(c.value)
        if c.is_controlled:
            return self.control_terms(c, depth + 1)
        raise CircuitError(f"{c.ref}: 元件类型 {c.kind} 的电流无法表达")

    def drop_terms(self, c, depth: int = 0):
        """``drop_c``（沿 declared_direction）= 未知量线性组合 + 常数。"""
        if c.kind == "R":
            return {self.unk[c.ref]: to_frac(c.value)}, Fraction(0)
        if c.kind == "I":
            return {self.unk[c.ref]: Fraction(1)}, Fraction(0)     # 未知的是它的电压
        if c.outputs_voltage:
            if c.kind == "V":
                return {}, -to_frac(c.value)
            # ★ 受控源的输出电压就是"控制量"，而沿参考方向（−→+）的压降是
            #   它的相反数 —— 与理想电压源 `−E` 同一个来路。
            n, cnst = self.control_terms(c, depth + 1)
            return {k: -v for k, v in n.items()}, -cnst
        if c.is_controlled:                     # G / F：电流型，电压是未知量
            return {self.unk[c.ref]: Fraction(1)}, Fraction(0)
        raise CircuitError(f"{c.ref}: 元件类型 {c.kind} 的压降无法表达")

    def delta(self, x: int, y: int, depth: int = 0):
        """``V_x − V_y`` = 未知量线性组合 + 常数（沿生成树唯一路径累加压降）。

        ★ 缓存是必要的：受控源的控制端差、每个回路里的每一段，都会反复问
        同一对节点。没有缓存时表达式一多就是指数级重复展开。
        """
        key = (x, y)
        hit = self._delta_cache.get(key)
        if hit is not None:
            return hit
        if depth > MAX_EXPR_DEPTH:
            raise CircuitError(
                "受控源的控制关系绕成了环（控制量最终又指回它自己）。"
                "请检查是不是有受控源互相控制，或控制端落在含它自己的回路上。")
        if x == y:
            res = ({}, Fraction(0))
            self._delta_cache[key] = res
            return res

        acc: dict[int, Fraction] = {}
        const = Fraction(0)
        for (u, v, ei) in _tree_path(self.parent, x, y):
            e_u, e_v, e_ref = self.edges[ei]
            c = self.known[e_ref]
            coeffs, cnst = self.drop_terms(c, depth + 1)
            forward = (u == e_u and v == e_v)
            sign = 1 if forward else -1
            for col, co in coeffs.items():
                acc[col] = acc.get(col, Fraction(0)) + sign * co
            const += sign * cnst
        res = ({k: v for k, v in acc.items() if v != 0}, const)
        self._delta_cache[key] = res
        return res

    def control_terms(self, c, depth: int = 0):
        """受控源的**输出量** = 未知量线性组合 + 常数。"""
        ctrl = c.ctrl
        if ctrl is None:
            raise CircuitError(f"{c.ref}: 受控源缺少控制支路")
        if depth > MAX_EXPR_DEPTH:
            raise CircuitError(
                "受控源的控制关系绕成了环（控制量最终又指回它自己）。"
                "请检查是不是有受控源互相控制。")

        if not ctrl.expr:
            gain = to_frac(c.value)
            if ctrl.mode == "V":
                assert ctrl.nodes is not None
                x = self.circuit.nodes.index(ctrl.nodes[0])
                y = self.circuit.nodes.index(ctrl.nodes[1])
                coeffs, cnst = self.delta(x, y, depth + 1)
                return ({k: gain * v for k, v in coeffs.items()}, gain * cnst)
            # 电流控制：采样支路已被 ensure_sense_sources 保证为电压输出元件
            sense = ctrl.sampling
            col = self.unk.get(sense)
            if col is None:
                raise CircuitError(
                    f"{c.ref}: 采样支路 {sense!r} 的电流不是可用的未知量"
                    "（内部一致性错误：探针源应当已经插好）")
            return {col: gain}, Fraction(0)

        lin = self._resolve_expr(ctrl.expr)
        acc: dict[int, Fraction] = {}
        const = Fraction(lin.const)
        for sym, k in lin.coeffs.items():
            coeffs, cnst = self._symbol_terms(sym, depth + 1)
            for col, co in coeffs.items():
                acc[col] = acc.get(col, Fraction(0)) + k * co
            const += k * cnst
        return ({k: v for k, v in acc.items() if v != 0}, const)

    # ---- 表达式的符号翻回未知量

    def _resolve_expr(self, expr: str) -> LinearExpr:
        return self.circuit.params.resolve_expression(expr)

    def _symbol_terms(self, sym: str, depth: int):
        table = self.circuit.params
        p = table.by_symbol(sym)
        if p is None:
            raise CircuitError(f"表达式里的 {sym!r} 不在当前参数表里")
        kind, key = p.binder
        if kind == "node_u":
            x = self.circuit.nodes.index(key)
            y = self.circuit.nodes.index(self.circuit.ref_node)
            return self.delta(x, y, depth)
        if kind == "branch_u":
            return self.drop_terms(self.circuit.by_ref(key), depth)
        if kind == "branch_i":
            return self.current_terms(self.circuit.by_ref(key), depth)
        if kind == "value":
            comp = self.circuit.by_ref(key)
            if comp.value is None:
                raise CircuitError(f"{sym!r} 对应的元件 {key} 还没有数值")
            return {}, to_frac(comp.value)
        raise CircuitError(f"未知的绑定种类 {kind!r}")


def branch_current_method(circuit: Circuit) -> Solution:
    """支路电流法。返回精确有理数解。"""
    circuit, sense = ensure_sense_sources(circuit)
    circuit.validate()
    reject_non_dc(circuit)

    nodes = circuit.nodes
    n = len(nodes)
    root = nodes.index(circuit.ref_node)

    # 弦边必须与回路一一对应，所以电流源若落在弦上会让回路电流被强制已知、
    # 减少未知量。这里不做这种优化 —— 一律当未知量处理，
    # 未知量与方程数严格配平（见下方 assert），逻辑最简单也最不容易错。
    edges = _build_edges(circuit)
    b = len(edges)
    tree, chords, parent = _spanning_tree(n, edges, root)
    n_loops = len(chords)
    if n_loops != b - n + 1:
        raise CircuitError(
            f"支路电流法：基本回路数 {n_loops} != b−n+1 = {b-n+1}，"
            "生成树构造有误（这是内部一致性断言，不应触发）"
        )

    known = {c.ref: c for c in circuit.components}
    # 未知量编号：电流自由的支路取电流；电流不自由的（I/G/F）取其电压
    unk: dict[str, int] = {}
    cols: list[tuple[str, str]] = []
    for c in circuit.components:
        unk[c.ref] = len(cols)
        cols.append(("i", c.ref) if c.kind not in ("I", "G", "F") else ("u", c.ref))
    n_unk = len(cols)

    n_kcl = n - 1
    n_eq = n_kcl + n_loops
    if n_eq != n_unk:
        raise CircuitError(
            f"支路电流法：方程数 {n_eq} != 未知量数 {n_unk}，"
            "说明支路集与未知量集配平有误（内部断言）"
        )

    ctx = _Base(circuit, edges, parent, unk, cols, n)
    A: list[list[Fraction]] = [[Fraction(0)] * n_unk for _ in range(n_eq)]
    rhs: list[Fraction] = [Fraction(0)] * n_eq

    # ---- (n−1) 条 KCL：除参考节点外，每个节点"流入 = 流出"（写成流出之和 = 0）
    kcl_row = {}
    for i, nd in enumerate(nodes):
        if i == root:
            continue
        kcl_row[i] = len(kcl_row)
    for i, (u, v, ref) in enumerate(edges):
        c = known[ref]
        # ★ 统一走 current_terms：电阻/电压源是"自身未知量"，
        #   电流源是常数，G/F 是控制量的线性组合。三种来源一条出口，
        #   免得"理想源写一处、受控源写另一处"然后其中一处符号写反。
        coeffs, const = ctx.current_terms(c)
        for node_i, sign in ((u, +1), (v, -1)):
            r = kcl_row.get(node_i)
            if r is None:
                continue
            for col, co in coeffs.items():
                A[r][col] += sign * co
            rhs[r] -= sign * const

    # ---- (b − n + 1) 条基本回路 KVL：沿回路逐支路累加"压降 = 0"
    # 回路统一由 fundamental_loops() 出，那里带闭合自证，避免拿到断开的"回路"
    loops = fundamental_loops(circuit)
    if len(loops) != n_loops:
        raise CircuitError(
            f"支路电流法：回路数 {len(loops)} 与弦边数 {n_loops} 不一致（内部幂等性断言）"
        )
    for li, (_chord_ref, walk) in enumerate(loops):
        row = n_kcl + li
        for (x, y, ei) in walk:
            e_u, e_v, e_ref = edges[ei]
            c = known[e_ref]
            forward = (x == e_u and y == e_v)
            sign = 1 if forward else -1
            coeffs, const = ctx.drop_terms(c)
            for col, co in coeffs.items():
                A[row][col] += sign * co
            rhs[row] -= sign * const

    sol_vec = solve_linear(A, rhs, what="支路电流法")

    # ---- 回代：把每一组"未知量线性组合"代入解向量
    # ★ 直接代入即可，不需要先用节点电压反推受控量 —— 那些组合在装配前
    #   就已经是"未知量的线性组合"了（见 _Base 的设计说明）。
    def value_of(coeffs: dict[int, Fraction], const: Fraction) -> Fraction:
        total = Fraction(const)
        for col, co in coeffs.items():
            total += co * sol_vec[col]
        return total

    currents: dict[str, Fraction] = {}
    drops: dict[str, Fraction] = {}
    for c in circuit.components:
        cf, cc = ctx.current_terms(c)
        currents[c.ref] = value_of(cf, cc)
        df, dc = ctx.drop_terms(c)
        drops[c.ref] = value_of(df, dc)

    node_v = build_node_voltages_from_drops(circuit, drops)

    detail = {
        "branches": b, "nodes": n, "tree_edges": len(tree),
        "fundamental_loops": n_loops,
        "unknowns": [f"{k}({r})" for k, r in cols],
    }
    if circuit.controlled:
        detail["controlled"] = [
            {"ref": c.ref, "kind": c.kind, "control": c.control_text(),
             "mode": "自定义表达式" if (c.ctrl and c.ctrl.expr) else "标准形",
             "i": str(currents.get(c.ref, "")), "u": str(drops.get(c.ref, ""))}
            for c in circuit.controlled
        ]

    return Solution(
        method="支路电流法(生成树基本回路/精确有理数)",
        node_voltages=node_v,
        branch_currents=currents,
        branch_drops=drops,
        exact=True,
        detail=detail,
    )
