"""支路电流法（独立复算路径）。

未知量 = **全部支路电流**（电流源的电流已知，所以改用它的电压作未知量）；
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
"""

from __future__ import annotations

from fractions import Fraction

from ..ir.model import Circuit, CircuitError
from .base import (Solution, declared_direction,
                   build_node_voltages_from_drops, reject_non_dc)
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

    ★ 这个函数是"电压源压降符号"的**唯一出口**。KVL 回路方程与由电流反推压降
    两处都调它，就不会出现"两处各写一遍、其中一处写反"的事故。
    """
    if c.kind == "V":
        return -val
    return None


def branch_current_method(circuit: Circuit) -> Solution:
    """支路电流法。返回精确有理数解。"""
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
    # 未知量编号：R/V 支路的电流 + I 支路的电压
    unk: dict[str, int] = {}
    cols: list[tuple[str, str]] = []
    for c in circuit.components:
        if c.kind in ("R", "V"):
            unk[c.ref] = len(cols)
            cols.append(("i", c.ref))
        else:                                    # I：电流已知，取其电压为未知量
            unk[c.ref] = len(cols)
            cols.append(("u", c.ref))
    n_unk = len(cols)

    n_kcl = n - 1
    n_eq = n_kcl + n_loops
    if n_eq != n_unk:
        raise CircuitError(
            f"支路电流法：方程数 {n_eq} != 未知量数 {n_unk}，"
            "说明支路集与未知量集配平有误（内部断言）"
        )

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
        val = to_frac(c.value)
        # 参考方向 from->to 就是 (u, v)
        for node_i, sign in ((u, +1), (v, -1)):
            r = kcl_row.get(node_i)
            if r is None:
                continue
            if c.kind == "I":
                rhs[r] -= sign * val          # 电流已知，移到右端
            else:
                A[r][unk[ref]] += sign

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
            val = to_frac(c.value)
            # 沿 x->y 行进；支路参考方向是 e_u->e_v
            forward = (x == e_u and y == e_v)
            if c.kind == "R":
                # drop = ±R·i
                A[row][unk[e_ref]] += (val if forward else -val)
            elif c.kind == "V":
                d = _declared_drop(c, val)      # = −E
                rhs[row] -= (d if forward else -d)
            else:                                  # I：未知的是它的电压
                A[row][unk[e_ref]] += (1 if forward else -1)

    sol_vec = solve_linear(A, rhs, what="支路电流法")

    currents: dict[str, Fraction] = {}
    drops: dict[str, Fraction] = {}
    for kind, ref in cols:
        if kind == "i":
            currents[ref] = sol_vec[unk[ref]]
    # 电流源的电流是已知量，显式补上（不靠解向量）
    for c in circuit.components:
        if c.kind == "I":
            currents[c.ref] = to_frac(c.value)

    # ---- 由支路电流与源值算各支路压降，再沿生成树推全节点电压
    for c in circuit.components:
        if c.kind == "R":
            drops[c.ref] = to_frac(c.value) * currents[c.ref]
        elif c.kind == "V":
            drops[c.ref] = _declared_drop(c, to_frac(c.value))   # = −E
        else:
            drops[c.ref] = sol_vec[unk[c.ref]]  # I 支路电压 = V_from − V_to

    node_v = build_node_voltages_from_drops(circuit, drops)

    return Solution(
        method="支路电流法(生成树基本回路/精确有理数)",
        node_voltages=node_v,
        branch_currents=currents,
        branch_drops=drops,
        exact=True,
        detail={
            "branches": b, "nodes": n, "tree_edges": len(tree),
            "fundamental_loops": n_loops,
            "unknowns": [f"{k}({r})" for k, r in cols],
        },
    )
