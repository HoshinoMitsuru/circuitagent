# circuit-agent MCP Server

给 **LLM / Agent** 用的直接解题工具层。宿主 agent（WorkBuddy / Claude 等）
把 SPICE 网表发进来，工具返回**带完整解题过程**的文本报告，agent 直接转述
给提问者 —— 全程不需要打开 WebUI。

## 三个工具

| 工具 | 用途 |
|---|---|
| `solve_circuit` | 主工具：网表 → 解析 → 三法求解 → 互校 → 文本报告（读图依据/参考方向/精确分数解/功率表/三法互校/叠加复核/结论），一次调用给最终形态 |
| `validate_netlist` | 只校验不求解：解析错误、悬空节点、受控源与储能元件冲突等，回显元件清单 |
| `list_supported_elements` | 元件写法示例与已知限制（示例全部实解析过，LLM 照抄即用） |

## 支持范围（直流）

- 二端：R / L / C（直流稳态：C 开路、L 短路，全程留痕）
- 独立源：V / I
- 受控源：E / G / H / F（H/F 控制端必须引用电压源位号）
- **明确拒绝**：交流/瞬态激励；受控源与 C/L 同图（安全闸门，报错文案写明原因与出路）

## 接入（stdio，零依赖）

```json
{
  "mcpServers": {
    "circuit-agent": {
      "command": "C:/Users/Psyche/.workbuddy/binaries/python/envs/circuit_agent/Scripts/python.exe",
      "args": ["D:/Psyche/trial/circuit_agent/mcp_server/server.py"]
    }
  }
}
```

依赖只有项目自己（`app/`），不需要任何额外 pip 包 —— 协议是手写的
JSON-RPC over stdio（MCP `2025-06-18`），每行一条请求。

## 手动冒烟

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"solve_circuit","arguments":{"netlist":"V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 2k"}}}' \
  | python mcp_server/server.py
```

回归测试：`python tests/test_mcp_server.py`（协议/解题/错误文案可自纠/示例自洽，0 失败为过）。

## 设计取舍

- **工具只留 3 个**：LLM 的失败模式是选项太多乱试，主流程压成一个 `solve_circuit`；
- **错误文案是提示词**：每条报错都带"问题 + 怎么改"，agent 拿到通常一次自纠，
  不需要盲目重试；
- **报告直接复用 `format_text_report`**：给人看的和给 LLM 看的是同一份，
  避免两套渲染各错各的；
- **isError 永远为 False**：解析失败/无解这类"业务错误"以文本返回，让 LLM
  能读懂并改写网表重试；只有协议级错误才走 JSON-RPC error。
