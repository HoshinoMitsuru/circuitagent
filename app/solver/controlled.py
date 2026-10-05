"""受控源在**求解**这一侧的公共设施。

三件事，三法（节点电压法 / 支路电流法 / ngspice）全都依赖这里：

1. ``ensure_sense_sources()`` —— 给电流控制型受控源在采样支路里插 0V 探针源。
   这是"为什么"要插：SPICE 的 ``H``/``F`` 卡只能引用**电压源**的电流；
   而教科书题目常写"受 ``R1`` 上的电流控制"。插一个 0V 源进去，
   那条支路就变成可以取电流的支路，且**电路的解完全不变**
   （0V 源对回路 KVL 的贡献恒为 0）。

   ★ 插进来的元件会改变节点数与支路数，所以它必须**留痕**：
   报告里要标明"测量探针引入、不是题目元件"。否则就是
   "题目里没有的东西偷偷进了答案"。

2. ``control_terms()`` —— 把受控源的受控量写成**线性项**：
   ``(节点电压系数, 电压源电流系数, 常数)``。MNA 直接按列装配，
   支路电流法换个基底再装配。**两种标准形 + 用户自定义表达式走同一条出口**，
   免得"标准形写在一处、表达式写在另一处"然后其中一处符号写反。

3. ``MnaCtx`` —— 列号的上下文（节点→列、电压输出元件→电流未知量列）。

符号约定的唯一出口是 :func:`app.ir.model.declared_direction`（已住到 IR 层）；
这里只做组合，不重新定义方向。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping

from ..ir.model import (Circuit, Component, CircuitError, Control,
                        CONTROLLED_KINDS, GAIN_SYMBOL)
from ..ir.params import MAX_EXPR_DEPTH, LinearExpr, Param
from .base import declared_direction
from .linalg import to_frac


@dataclass
class MnaCtx:
    """MNA 的列号上下文。

    ``node_col`` 不含参考节点（它的电压恒为 0，不占未知量）。
    ``vslot`` 是"输出为电压"的元件（V/E/H）到它们**支路电流未知量**列号的映射。
    """

    node_col: dict[str, int]
    vslot: dict[str, int]
    #: 取精确有理数的函数（``to_frac``）。放进来是为了 **不在这里再写一份**
    #: 数值转换 —— 三条路径必须用同一个转换，否则精度口径就分家了。
    frac: Callable[[Any], Fraction] = field(default=to_frac)


# ---------------------------------------------------------------- 探针源

# ★ 整段搬到 :mod:`app.ir.probes` 了：三个求解路径**和网表落盘**都要用它，
#   留在求解层就会让 IR 反过来依赖求解层。这里只做转发（mna/branch/ngspice
#   都还按老路径 import），实现只有那一份。
from ..ir.probes import SENSE_PREFIX, ensure_sense_sources, sense_ref_for  # noqa: F401
# ---------------------------------------------------------------- 控制项

def _param_table(circuit: Circuit):
    # 参数表是"名字 ↔ 绑定"的派生层。表达式用它把名字翻回 ref / 节点名。
    # 万一表还没同步（表里一个名字都没有），表达式就会被判成"未知符号" ——
    # 所以这里先补一次同步，代价是一次 O(n) 遍历。
    if not circuit.params.items:
        circuit.sync_params()
    return circuit.params


def _branch_current_terms(comp: Component, circuit: Circuit, ctx: MnaCtx,
                          depth: int) -> tuple[dict[int, Fraction],
                                               dict[int, Fraction], Fraction]:
    """把"一条支路的电流"写成 MNA 未知量的线性组合。

    返回 ``(节点系数, 电压源电流系数, 常数)``，键都是**列号**。

    这是受控源在 MNA 里最核心的一步：``F``（CCCS）的控制量是别处的电流，
    而 MNA 的未知量是节点电压 —— 两者之间的翻译就靠这里。
    """
    if comp.outputs_voltage:
        col = ctx.vslot.get(comp.ref)
        if col is None:
            raise CircuitError(
                f"{comp.ref}: 电压输出元件却没有分配到电流未知量列"
                "（内部一致性错误，不应触发）")
        return {}, {col: Fraction(1)}, Fraction(0)

    if comp.kind == "R":
        val = ctx.frac(comp.value)
        if val == 0:
            raise CircuitError(f"{comp.ref}: 电阻为 0，取电流时会除零")
        g = Fraction(1) / val
        a, b = comp.nodes
        out: dict[int, Fraction] = {}
        if a in ctx.node_col:
            out[ctx.node_col[a]] = out.get(ctx.node_col[a], Fraction(0)) + g
        if b in ctx.node_col:
            out[ctx.node_col[b]] = out.get(ctx.node_col[b], Fraction(0)) - g
        return out, {}, Fraction(0)

    if comp.kind == "I":
        return {}, {}, ctx.frac(comp.value)

    if comp.is_controlled:
        # G / F 的电流本身是受控量：按它们的控制关系展开
        node_c, vs_c, const = control_terms(comp, circuit, ctx, depth=depth + 1)
        return node_c, vs_c, const

    raise CircuitError(
        f"{comp.ref}: 元件类型 {comp.kind} 的电流无法用节点电压法表达")


def _symbol_terms(sym: str, circuit: Circuit, ctx: MnaCtx,
                  depth: int) -> tuple[dict[int, Fraction],
                                       dict[int, Fraction], Fraction]:
    """把一个**参数名**翻成 MNA 的线性项。

    ``u_3`` → 节点 3 的列；``i_R1`` → 由 R1 的电阻值换成节点电压差；
    ``R1`` → 常数。名字用哪一套由参数表决定，所以用户改了名字也不影响这里。
    """
    if depth > MAX_EXPR_DEPTH:
        raise CircuitError(
            "受控源的控制关系绕成了环（控制量最终又指回它自己）。"
            "请检查是不是有受控源之间互相控制。")

    table = _param_table(circuit)
    p: Param | None = table.by_symbol(sym)
    if p is None:
        raise CircuitError(
            f"表达式里的 {sym!r} 不在当前参数表里。"
            "（若刚改过名字，请先在「参数」页确认名字已保存）")

    kind, key = p.binder

    # ---- 节点电压
    if kind == "node_u":
        if key == circuit.ref_node:
            return {}, {}, Fraction(0)
        col = ctx.node_col.get(key)
        if col is None:
            raise CircuitError(f"节点 {key} 不在未知量里（内部一致性错误）")
        return {col: Fraction(1)}, {}, Fraction(0)

    # ---- 支路电压 = V(from) − V(to)，from/to 取 declared_direction
    if kind == "branch_u":
        comp = circuit.by_ref(key)
        if comp.ref in ctx.vslot:
            # 电压输出元件的压降：V_p − V_n，p/n 是元件自己的端子
            raise CircuitError(
                f"{sym!r} 是受控源的输出电压，本版本不支持在表达式里引用它："
                "它的值与控制量构成代数环，展开会不收敛。"
                "若要引用，请改用节点电压（如 u_3）来写。")
        f, t = declared_direction(comp)
        out: dict[int, Fraction] = {}
        if f in ctx.node_col:
            out[ctx.node_col[f]] = out.get(ctx.node_col[f], Fraction(0)) + 1
        if t in ctx.node_col:
            out[ctx.node_col[t]] = out.get(ctx.node_col[t], Fraction(0)) - 1
        return out, {}, Fraction(0)

    # ---- 支路电流
    if kind == "branch_i":
        comp = circuit.by_ref(key)
        return _branch_current_terms(comp, circuit, ctx, depth)

    # ---- 元件值：常数
    if kind == "value":
        comp = circuit.by_ref(key)
        if comp.value is None:
            raise CircuitError(f"{sym!r} 对应的元件 {key} 还没有数值")
        return {}, {}, ctx.frac(comp.value)

    raise CircuitError(f"未知的绑定种类 {kind!r}")


def expr_terms(lin: LinearExpr, circuit: Circuit, ctx: MnaCtx,
               depth: int = 0) -> tuple[dict[int, Fraction],
                                        dict[int, Fraction], Fraction]:
    """把一个线性表达式整体翻成 MNA 线性项。"""
    node_c: dict[int, Fraction] = {}
    vs_c: dict[int, Fraction] = {}
    const = Fraction(lin.const)

    for sym, k in lin.coeffs.items():
        n, v, c = _symbol_terms(sym, circuit, ctx, depth)
        for col, co in n.items():
            node_c[col] = node_c.get(col, Fraction(0)) + k * co
        for col, co in v.items():
            vs_c[col] = vs_c.get(col, Fraction(0)) + k * co
        const += k * c

    return ({c: v for c, v in node_c.items() if v != 0},
            {c: v for c, v in vs_c.items() if v != 0},
            const)


def standard_control_terms(comp: Component, ctx: MnaCtx,
                           gain: Fraction) -> tuple[dict[int, Fraction],
                                                    dict[int, Fraction], Fraction]:
    """**标准形**（``输出 = 增益 × 控制量``）的控制项。

    电压控制 → ``gain·(V(x) − V(y))``；电流控制 → ``gain·i(采样支路)``。
    后者要求采样支路是电压输出元件（``ensure_sense_sources`` 已保证）。
    """
    ctrl = comp.ctrl
    assert ctrl is not None
    if ctrl.mode == "V":
        assert ctrl.nodes is not None
        x, y = ctrl.nodes
        out: dict[int, Fraction] = {}
        if x in ctx.node_col:
            out[ctx.node_col[x]] = out.get(ctx.node_col[x], Fraction(0)) + gain
        if y in ctx.node_col:
            out[ctx.node_col[y]] = out.get(ctx.node_col[y], Fraction(0)) - gain
        return out, {}, Fraction(0)

    sense = ctrl.sampling
    col = ctx.vslot.get(sense)
    if col is None:
        raise CircuitError(
            f"{comp.ref}: 电流控制型受控源的采样支路 {sense!r} 不是电压输出元件，"
            "取不到电流。这通常说明探针源没有插上（内部一致性错误）。")
    return {}, {col: gain}, Fraction(0)


def terms_value(node_c: dict[int, Fraction], vs_c: dict[int, Fraction],
                const: Fraction, node_v: dict[str, Any], hot: list[str],
                src_current: dict[str, Any],
                vsources: list[Component]) -> Fraction:
    """把线性项按**已解出**的节点电压与源电流回代，算出数值。

    给两处用：

    * 反推电流输出受控源（``G``/``F``）自己的电流 —— 它们不是未知量，
      不还代就等于把它们当 0，而功率守恒反而更容易成立（少算了一部分），
      于是"看起来一切正常"；
    * 报告里展示受控关系的实际取值。
    """
    total = Fraction(const)
    for col, co in node_c.items():
        total += co * node_v.get(hot[col], Fraction(0))
    for col, co in vs_c.items():
        idx = col - len(hot)
        if not 0 <= idx < len(vsources):
            raise CircuitError(f"源电流列号 {col} 越界（内部一致性错误）")
        total += co * src_current.get(vsources[idx].ref, Fraction(0))
    return total


class DependencyNotReady(Exception):
    """某个受控源的依赖还没算出来（内部重试信号，不对外）。

    只在"按数值回代受控源电流"的排序循环里用：``G``/``F`` 的电流可能依赖
    另一个受控源的电流，而参数表里的表达式是否引用它只有解析后才知道。
    拿它当重试信号，比事先建一张依赖图简单得多，也不会漏掉
    "表达式里间接引用"的情形。
    """

    def __init__(self, missing: list[str]) -> None:
        super().__init__("、".join(missing))
        self.missing = missing


def parameter_values(circuit: Circuit, node_v: Mapping[str, Any],
                     currents: Mapping[str, Any] | None = None,
                     *, need: Iterable[str] | None = None) -> dict[str, Any]:
    """参数名 -> 数值。给"按数值回代"与 UI 的参数面板共用。

    ``need`` 给定时只算这些名字（省掉无关的查表，也让"缺哪个"更清楚）。
    """
    cur = dict(currents or {})
    table = _param_table(circuit)
    out: dict[str, Any] = {}
    for p in table.live():
        if need is not None and p.symbol not in need:
            continue
        kind, key = p.binder
        if kind == "value":
            try:
                out[p.symbol] = circuit.by_ref(key).value
            except CircuitError:
                out[p.symbol] = None
        elif kind == "node_u":
            if key == circuit.ref_node:
                out[p.symbol] = 0
            else:
                out[p.symbol] = node_v.get(key)
        elif kind == "branch_u":
            try:
                comp = circuit.by_ref(key)
            except CircuitError:
                out[p.symbol] = None
                continue
            f, t = declared_direction(comp)
            a, b = node_v.get(f), node_v.get(t)
            out[p.symbol] = (None if a is None or b is None else a - b)
        elif kind == "branch_i":
            out[p.symbol] = cur.get(key)
    return out


def controlled_value(comp: Component, circuit: Circuit,
                     node_v: Mapping[str, Any],
                     currents: Mapping[str, Any]) -> Any:
    """按**数值**求一个受控源的输出量（电流型求电流、电压型求输出电压）。

    依赖没齐时抛 ``DependencyNotReady``，由调用方排序重试。

    ★ 有这条路径是因为 ngspice 那一路只能给出节点电压与电压源的电流：
    受控源自身的电流它不给，必须由我们按控制关系回代。**漏了这一步，
    受控源在功率表与 KCL 校验里就是 0** —— 而功率"守恒"反而更容易成立
    （少算了一部分），于是看起来一切正常。
    """
    ctrl = comp.ctrl
    if ctrl is None:
        raise CircuitError(f"{comp.ref}: 受控源缺少控制支路")

    if not ctrl.expr:
        gain = comp.value
        if gain is None:
            raise CircuitError(f"{comp.ref}: 受控源缺少增益")
        if ctrl.mode == "V":
            assert ctrl.nodes is not None
            x, y = node_v.get(ctrl.nodes[0]), node_v.get(ctrl.nodes[1])
            if x is None or y is None:
                raise DependencyNotReady([ctrl.nodes[0], ctrl.nodes[1]])
            return gain * (x - y)
        sense = ctrl.sampling
        iv = currents.get(sense)
        if iv is None:
            raise DependencyNotReady([f"i({sense})"])
        return gain * iv

    lin = _param_table(circuit).resolve_expression(ctrl.expr)
    wanted = list(lin.coeffs)
    vals = parameter_values(circuit, node_v, currents, need=wanted)
    missing = [s for s in wanted if vals.get(s) is None]
    if missing:
        raise DependencyNotReady(missing)
    return lin.eval(vals)


def resolve_controlled_currents(circuit: Circuit, node_v: Mapping[str, Any],
                                currents: dict[str, Any]) -> list[str]:
    """按依赖顺序把 ``G``/``F`` 的电流补进 ``currents``。返回补了哪些位号。

    最多迭代"受控源个数 + 1"轮。一轮下来毫无进展说明它们互相依赖成环 ——
    那就**明确报错**，而不是留一堆缺值去污染后面的功率表。
    """
    pending = {c.ref: c for c in circuit.components
               if c.is_controlled and c.kind in ("G", "F")}
    if not pending:
        return []
    done: list[str] = []

    for _ in range(len(pending) + 1):
        if not pending:
            break
        progressed = False
        for ref in list(pending):
            try:
                value = controlled_value(pending[ref], circuit, node_v, currents)
            except DependencyNotReady:
                continue
            # 直接写回 currents 就够了：parameter_values() 每次都按位号现查，
            # 下一轮的受控源立刻就能看到它。
            currents[ref] = value
            done.append(ref)
            del pending[ref]
            progressed = True
        if not pending:
            break
        if not progressed:
            raise CircuitError(
                "受控源的电流互相依赖成了环（"
                + "、".join(sorted(pending))
                + "），无法定出它们的取值。请检查是否有互相控制的受控源。")
    return done

# ---------------------------------------------------------------- 表达式渲染

# ★ "把线性表达式写成 SPICE 文本"整个搬到 :mod:`app.ir.spice_expr` 了：
#   网表层（``to_spice``）也要用它把受控源落盘，留在求解层就会让
#   IR 反过来依赖求解层。**这里不再留副本** —— 两份渲染器分叉的表现是
#   "网表看着没问题，ngspice 算出来却是另一个电路"。
from ..ir.spice_expr import spice_expression  # noqa: F401  (对外沿用旧导入路径)


# ---------------------------------------------------------------- 出口

def control_terms(comp: Component, circuit: Circuit, ctx: MnaCtx,
                  depth: int = 0) -> tuple[dict[int, Fraction],
                                           dict[int, Fraction], Fraction]:
    """受控源**输出量**的线性项（标准形与自定义表达式走同一条出口）。

    这是"两条精确路径必须与 ngspice 对得上"的关键：无论用户写了什么
    线性表达式，最终都归到同一组 ``(节点系数, 源电流系数, 常数)`` 上，
    三种表达方式不可能出现符号分叉。
    """
    ctrl = comp.ctrl
    if ctrl is None:
        raise CircuitError(f"{comp.ref}: 受控源缺少控制支路")
    if comp.value is None and not ctrl.expr:
        raise CircuitError(
            f"{comp.ref}: 受控源既没有增益（{GAIN_SYMBOL.get(comp.kind, '')}）"
            "也没有自定义表达式，无法确定它的受控关系")

    gain = ctx.frac(comp.value) if comp.value is not None else None

    if not ctrl.expr:
        assert gain is not None
        return standard_control_terms(comp, ctx, gain)

    lin = _param_table(circuit).resolve_expression(ctrl.expr)
    node_c, vs_c, const = expr_terms(lin, circuit, ctx, depth)

    # 标准形与自定义表达式不一致时**不报错**（用户是有意改写的），
    # 但要把"实际用了哪个"记在报告里 —— 交给 reconcile / 参数面板去呈现。
    return node_c, vs_c, const
