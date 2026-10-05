"""MCP 工具层的端到端测试：不起任何网络服务，直接在进程内打协议。

覆盖三类缺陷（都是"静默出错、LLM 拿到错误信息也没法自纠"的形态）：
  1. 协议层：initialize 握手、tools/list 与实现一致、未知方法/工具的报错形态；
  2. 主工具：可解题、C/L 化简留痕、受控源混合、四类错误文案可自纠
     （空输入/坏语法/受控源+C/L 冲突/H 卡引用电阻）；
  3. 辅助工具：validate 的元件清单与冲突预警、elements 的示例自洽
     （说明里的示例必须真的能解，否则 LLM 照着抄就是错的）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.server import TOOLS, TOOLS_MAP, handle  # noqa: E402

FAIL = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global FAIL
    if not ok:
        FAIL += 1
    tag = "[通过]" if ok else "[失败]"
    print(f"  {tag} {name}" + (f"  —— {detail}" if detail != "" else ""))


def banner(t: str) -> None:
    print("\n" + "#" * 72)
    print("# " + t)
    print("#" * 72)


def call_tool(name: str, args: dict) -> str:
    """模拟一次 tools/call，取回文本内容。"""
    raw = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": args}})
    resp = json.loads(raw)
    assert "result" in resp, resp
    return resp["result"]["content"][0]["text"]


def main() -> int:
    banner("A 协议层：握手与工具清单")
    r = json.loads(handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                           "params": {}}))
    check("initialize 返回协议版本与 serverInfo",
          "protocolVersion" in r["result"] and r["result"]["serverInfo"]["name"]
          == "circuit-agent")
    r = json.loads(handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
    names = {t["name"] for t in r["result"]["tools"]}
    check("tools/list 与实现表一致", names == set(TOOLS_MAP), f"{names}")
    check("每个工具都有 inputSchema 与 description",
          all(t.get("inputSchema") is not None and t.get("description")
              for t in r["result"]["tools"]))
    check("通知（无 id）不回包",
          handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
          is None)
    check("未知方法返回 -32601",
          json.loads(handle({"jsonrpc": "2.0", "id": 2,
                             "method": "no/such"}))["error"]["code"] == -32601)

    banner("B 主工具 solve_circuit：能解题、给过程")
    net_good = "V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 2k\n"
    out = call_tool("solve_circuit", {"netlist": net_good})
    check("正常解题：报告含分压精确值 V(2)=20/3",
          "V(2) = 20/3" in out, out[:80])
    check("报告含三法对账表", "三法对账表" in out and "ngspice" in out)
    check("报告含功率表与结论", "功率表" in out and "【结论】" in out)
    check("报告含参考方向约定（LLM 转述时必须讲清符号）",
          "参考方向" in out)

    # C/L 化简留痕
    net_cl = "V1 1 0 DC 12\nR1 1 2 100\nC1 2 0 1u\nL1 2 3 1m\nR2 3 0 300\nI1 3 0 DC 0.01\n"
    out = call_tool("solve_circuit", {"netlist": net_cl})
    check("含 C/L：解出来且化简留痕完整",
          "直流稳态化简" in out and "开路" in out and "短接" in out
          and "【结论】" in out)

    # 受控源混合
    net_ctrl = ("V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 2k\nE1 3 0 2 0 3\n"
                "R3 3 0 1k\nVseg 1 4 DC 0\nR4 4 0 500\nF1 3 0 Vseg 2\n"
                "I1 3 0 DC 0.001\n")
    out = call_tool("solve_circuit", {"netlist": net_ctrl})
    check("受控源混合（E+F+V+I+R）：解出来且互校通过",
          "【结论】" in out and ("通过" in out or "consistent" in out),
          out[-120:])

    banner("C 错误文案可自纠（LLM 拿到后一次改对的形态）")
    out = call_tool("solve_circuit", {"netlist": "   "})
    check("空输入：给出写法示例", "netlist 是空的" in out and "R1 1 2" in out)
    out = call_tool("solve_circuit", {"netlist": "X1 1 0 1k\n"})
    check("未知元件类型：报错点名位号", "解析失败" in out and ("X1" in out or "X" in out))
    out = call_tool("solve_circuit",
                    {"netlist": "V1 1 0 DC 10\nR1 1 2 1k\nC1 2 0 1u\n"
                                "E1 3 0 2 0 2\nR3 3 0 1k\n"})
    check("受控源+C/L 冲突：说明原因与出路（手算化简）",
          "无法求解" in out and ("电容" in out or "化简" in out))
    out = call_tool("solve_circuit",
                    {"netlist": "V1 1 0 DC 10\nR1 1 2 1k\nH1 3 0 R1 200\n"
                                "R3 3 0 1k\n"})
    check("H 卡引用电阻：指出只许引用电压源并给两条改法",
          "电压源" in out and ("0V" in out or "改" in out))

    banner("D 辅助工具")
    out = call_tool("validate_netlist", {"netlist": net_good})
    check("validate：清单+参考节点+结构通过",
          "元件清单" in out and "参考节点 0" in out and "结构校验通过" in out)
    out = call_tool("validate_netlist",
                    {"netlist": "V1 1 0 DC 10\nC1 2 0 1u\nE1 3 0 1 0 2\n"})
    check("validate：受控源+储能冲突提前预警（不用等求解）",
          "本工具会拒绝求解" in out)
    out = call_tool("validate_netlist", {"netlist": "V1 1 0 5\n"})
    check("validate：悬空节点被抓出来",
          ("疑似悬空" in out), out[-150:])

    # elements 里的每个示例都必须真的能解析/求解 —— LLM 会照抄
    out = call_tool("list_supported_elements", {})
    check("elements：返回示例与限制", "E1 3 0 2 0 3" in out and "已知限制" in out)
    examples = ["R1 1 2 1k", "V1 1 0 DC 12", "I1 2 0 DC 0.002",
                "C1 2 0 1u", "L1 2 3 1m", "E1 3 0 2 0 3",
                "G1 3 0 2 0 0.001"]
    from app.ir.spice import from_spice
    bad = []
    for ex in examples:
        try:
            from_spice(ex + "\n", name="doc")
        except Exception as e:                       # noqa: BLE001
            bad.append(f"{ex}: {e}")
    check("★ elements 的写法示例全部可解析（照抄即用）", not bad, str(bad[:3]))

    print("\n" + "=" * 72)
    print(f"MCP 工具层测试失败项：{FAIL}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
