"""面向 LLM / Agent 的直接解题工具层（stdio JSON-RPC，MCP 2.0 协议）。

★ 定位：WebUI 是**给人**用的（叠图、确认、拍板）；这一层是给 **LLM** 用的 ——
agent 把题目文本/网表发进来，工具返回**带完整解题过程的结论**，LLM 拿去
直接转述给提问者。中间不需要任何界面操作。

为什么用 stdio JSON-RPC 而不是 HTTP：主流 agent 运行时（Claude/WorkBuddy/…）
的 MCP 客户端都把「stdio 子进程」当一等公民 —— 一行配置就能挂上，
不需要起端口、不需要管防火墙、生命周期随宿主走。

为什么工具这么少（就 3 个）：LLM 的失败模式是"选项太多乱试"。
这里刻意把整条流水线压成一个主工具 ``solve_circuit``：
    网表文本 → 解析 → 校验 → 三法求解 → 互校 → 文本报告
一次调用直接给最终形态。另外两个辅助工具（``list_supported_elements``、
``validate_netlist``）只负责"解题前的自检"，避免 LLM 拿格式问题
反复浪费完整求解。

★ 错误 philosophy（与整个项目一致）：**报错文案是给 LLM 读的提示词**。
每条错误都写清"是什么问题 + 怎么改"，LLM 拿到后通常一次就能自纠 ——
这比让 agent 盲目重试三次便宜得多。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 让脚本无论从哪里被调起都能找到 app 包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ir.model import Circuit, CircuitError  # noqa: E402
from app.ir.spice import from_spice  # noqa: E402
from app.solver.reconcile import format_text_report, run_all  # noqa: E402

# ---------------------------------------------------------------- 元数据

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "circuit-agent", "version": "1.0.0"}


def _tool_solve() -> dict[str, Any]:
    return {
        "name": "solve_circuit",
        "description": (
            "求解线性直流电阻电路（含受控源；支持电容/电感的直流稳态化简）。"
            "输入一张 SPICE 网表，返回带完整解题过程的文本报告："
            "读图依据、参考方向约定、节点电压与支路电流（精确分数）、"
            "功率表、三法（节点电压法/支路电流法/ngspice）逐项互校、"
            "叠加原理复核与最终结论。元件：R/L/C/V/I + 受控源 E/G/H/F。"
            "流控源（H/F）的控制端必须引用电压源位号；"
            "受控源与储能元件（C/L）不可同图。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": (
                        "SPICE 网表文本。每行一个元件卡："
                        "R1 1 2 1k / V1 1 0 DC 12 / I1 2 0 DC 0.002 / "
                        "E1 3 0 2 0 3（压控压源） / G1 3 0 2 0 0.001（压控流源） / "
                        "H1 3 0 Vs 200（流控压源，控制端是电压源位号） / "
                        "F1 3 0 Vs 2（流控流源，同前）。节点名为任意整数或短标识。"
                    ),
                },
                "reduce_dc": {
                    "type": "boolean",
                    "description": "含 C/L 时是否做直流稳态化简（C 开路、L 短路，全程留痕）。默认 true。",
                },
            },
            "required": ["netlist"],
        },
    }


def _tool_validate() -> dict[str, Any]:
    return {
        "name": "validate_netlist",
        "description": (
            "只校验不求解：检查网表能否解析、结构是否合法（悬空节点、"
            "受控源与储能元件冲突、控制端引用错误等），并回显解析出的元件清单。"
            "适合解题前先确认格式，或求解报错后排查。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "netlist": {"type": "string", "description": "SPICE 网表文本。"},
            },
            "required": ["netlist"],
        },
    }


def _tool_elements() -> dict[str, Any]:
    return {
        "name": "list_supported_elements",
        "description": "列出支持的元件类型、写法示例与已知限制（何时会拒绝求解）。",
        "inputSchema": {"type": "object", "properties": {}},
    }


TOOLS = [_tool_solve(), _tool_validate(), _tool_elements()]

# ---------------------------------------------------------------- 实现


def _fmt_value(v: Any) -> str:
    """元件值统一成可读字符串（LLM 读的，别给 Python repr）。"""
    if v is None:
        return "（未给值）"
    return str(v)


def do_solve(args: dict[str, Any]) -> str:
    """主工具：一次调用给完整解题报告。错误直接作为文本返回（不抛到协议层）。"""
    text = str(args.get("netlist") or "")
    if not text.strip():
        return "错误：netlist 是空的。把 SPICE 网表文本传进来，例如 'V1 1 0 DC 12\\nR1 1 2 1k'。"
    reduce_dc = bool(args.get("reduce_dc", True))
    try:
        circ = from_spice(text, name="agent")
    except CircuitError as e:
        return (
            "解析失败（输入有问题，请修正后重发）：\n"
            f"{e}\n\n"
            "写法速查：R1 1 2 1k ｜ V1 1 0 DC 12 ｜ I1 2 0 DC 0.002 ｜ "
            "E1 3 0 2 0 3 ｜ G1 3 0 2 0 0.001 ｜ H1 3 0 Vs 200 ｜ F1 3 0 Vs 2"
        )
    except Exception as e:                          # noqa: BLE001
        return f"解析失败（意外错误）：{type(e).__name__}: {e}"

    # 先做一次结构校验，把"能解析但拓扑有问题"的情况用短文案挡在求解前
    try:
        warns = circ.validate()
    except CircuitError as e:
        return f"结构校验未通过（输入有问题，请修正后重发）：\n{e}\n"
    if warns:
        head = "结构校验有提醒（不阻断，但请先确认这些是不是本意）：\n"
        body = "\n".join(f"  · {w}" for w in warns[:8])
        pre = head + body + "\n\n"
    else:
        pre = ""

    try:
        pack = run_all(circ, reduce_dc=reduce_dc)
    except CircuitError as e:
        # CircuitError 的文案本来就是"问题 + 怎么办"双段式，直接给 LLM
        return pre + f"无法求解：\n{e}"
    except Exception as e:                          # noqa: BLE001
        return pre + f"求解失败（意外错误）：{type(e).__name__}: {e}"

    try:
        report = format_text_report(circ, pack)
    except Exception as e:                          # noqa: BLE001
        # 报告渲染失败不能吞掉已经算好的对账包 —— 给出精简版结论兜底
        rows = pack.get("node_table") or []
        brief = "\n".join(f"  V({r['node']}) = {r['exact']}" for r in rows)
        brs = pack.get("branch_table") or []
        brief += "\n" + "\n".join(f"  i({r['ref']}) = {r['exact']}" for r in brs)
        return (pre + "求解完成，但报告渲染失败（以下是精简结果）：\n"
                + brief + f"\n\n渲染错误：{type(e).__name__}: {e}")

    return pre + report


def do_validate(args: dict[str, Any]) -> str:
    text = str(args.get("netlist") or "")
    if not text.strip():
        return "错误：netlist 是空的。"
    try:
        circ = from_spice(text, name="agent")
    except CircuitError as e:
        return f"解析失败：\n{e}"
    except Exception as e:                          # noqa: BLE001
        return f"解析失败（意外错误）：{type(e).__name__}: {e}"

    lines = [f"解析成功：{len(circ.components)} 个元件，节点 {', '.join(circ.nodes)}，"
             f"参考节点 {circ.ref_node}。元件清单："]
    for c in circ.components:
        ctrl = ""
        if c.ctrl is not None:
            ctrl = f"  控制关系：mode={c.ctrl.mode} ref={c.ctrl.ref or '（未定）'}"
        lines.append(f"  {c.ref:<6} {c.kind:<2} {c.nodes[0]}->{c.nodes[1]}  "
                     f"值={_fmt_value(c.value)}{ctrl}")

    try:
        warns = circ.validate()
    except CircuitError as e:
        lines.append(f"\n结构校验未通过：\n{e}")
        return "\n".join(lines)
    if warns:
        lines.append("\n结构提醒：")
        lines += [f"  · {w}" for w in warns[:10]]
    else:
        lines.append("\n结构校验通过。")
    # 受控源+C/L 冲突提前说（这是最常见的"解析都好、求解必炸"项）
    controlled = [c.ref for c in circ.components if c.is_controlled]
    storages = [c.ref for c in circ.components if c.kind in ("C", "L")]
    if controlled and storages:
        lines.append(f"\n注意：受控源（{', '.join(controlled)}）与储能元件"
                     f"（{', '.join(storages)}）同图 —— 本工具会拒绝求解。"
                     "请先手算化简（电容开路、电感短路）再改写网表。")
    return "\n".join(lines)


def do_elements(_args: dict[str, Any]) -> str:
    return """支持的元件与写法（SPICE 网表，每行一张卡）：

二端元件：
  电阻   R1 1 2 1k        （支持工程记法 1k / 4k7 / 1meg / 10u）
  电容   C1 2 0 1u        直流稳态下视为开路（留痕）
  电感   L1 2 3 1m        直流稳态下视为短路（合并节点，留痕，电流会反算）
独立源：
  电压源 V1 1 0 DC 12     （DC 可省略；i(V1) 正值 = 供电）
  电流源 I1 2 0 DC 0.002  （方向从首节点流向次节点）
受控源（四类，控制端写节点对或电压源位号）：
  E1 3 0 2 0 3    压控压源 VCVS：u3 = 3 × (V2 − V0)
  G1 3 0 2 0 0.001 压控流源 VCCS：i = 0.001 × (V2 − V0)
  H1 3 0 Vs 200   流控压源 CCVS：u = 200 × i(Vs)  ★控制端必须是电压源位号
  F1 3 0 Vs 2     流控流源 CCCS：i = 2 × i(Vs)   ★同上

已知限制（会明确拒绝并说明，不会硬算）：
  1. 只做线性直流：AC/SIN/PULSE 等激励一律拒绝；
  2. 受控源与储能元件（C/L）不可同图 —— 请先手算直流稳态化简再输入；
  3. 电感把理想电压源短路等理想化矛盾 → 报"题目自相矛盾"，请改题；
  4. 行为源 B 卡不支持，自定义表达式请在 WebUI 参数页使用。

节点命名：任意整数或短标识（1/2/A/B…）；地节点建议用 0。
输出内容：完整文本报告（读图依据/参考方向/精确分数解/功率表/三法互校/叠加复核/结论）。"""


# ---------------------------------------------------------------- 协议层


def _ok(id_: Any, result: dict[str, Any]) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id_, "result": result},
                      ensure_ascii=False)


def _err(id_: Any, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id_,
                       "error": {"code": code, "message": message}},
                      ensure_ascii=False)


TOOLS_MAP = {
    "solve_circuit": do_solve,
    "validate_netlist": do_validate,
    "list_supported_elements": do_elements,
}


def handle(req: dict[str, Any]) -> str | None:
    """处理一条 JSON-RPC 请求。通知类（无 id）返回 None。"""
    method = req.get("method")
    id_ = req.get("id")

    if method == "initialize":
        return _ok(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _ok(id_, {})
    if method == "tools/list":
        return _ok(id_, {"tools": TOOLS})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        fn = TOOLS_MAP.get(name)
        if fn is None:
            return _err(id_, -32602, f"未知工具：{name}。可用：{list(TOOLS_MAP)}")
        try:
            text = fn(params.get("arguments") or {})
        except Exception as e:                       # noqa: BLE001
            # 工具内部连兜底都失败 —— 报给协议层，但不带栈（LLM 不需要栈）
            text = f"工具内部错误：{type(e).__name__}: {e}"
        return _ok(id_, {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        })
    if id_ is not None:
        return _err(id_, -32601, f"未知方法：{method}")
    return None


def main() -> int:
    """stdio 事件循环：每行一条 JSON-RPC。EOF 或 exit 请求退出。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(_err(None, -32700, f"JSON 解析失败：{e}") + "\n")
            sys.stdout.flush()
            continue
        out = handle(req)
        if out is not None:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
        if req.get("method") == "exit":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
