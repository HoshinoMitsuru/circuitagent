# Circuit Agent

电路智能分析代理 —— 从电路图照片、SVG 或 KiCad 文件到完整求解结果的一站式解决方案。

## 功能特性

### 多种输入方式
- **图片 OCR 识别**：从照片/截图自动识别电路元件和连接
- **SVG 解析**：支持从 SVG 电路图导入
- **KiCad 网表**：支持 KiCad 原理图和网表导入
- **SPICE 网表**：支持读取和导出 SPICE 网表

### 电路求解
- **节点电压法 (MNA)**：经典矩阵求解
- **支路电流法**：基于基本回路的求解
- **DC 化简**：自动合并串/并联元件
- **ngspice 求解**：集成开源 SPICE 引擎
- **三法互校**：三种方法交叉验证结果

### 受控源与参数体系
- 电压控制电压源 (VCVS/E)
- 电压控制电流源 (VCCS/G)
- 电流控制电压源 (CCVS/H)
- 电流控制电流源 (CCCS/F)
- 参数表达式支持

### 验证功能
- KCL (基尔霍夫电流定律) 验证
- KVL (基尔霍夫电压定律) 验证
- 功率平衡验证

### 可视化
- SVG 渲染电路图
- 显示节点电压和支路电流
- 暗色/亮色主题切换

## 安装

### 环境要求
- Python 3.10+
- Windows 10/11 (支持打包为单文件 exe)

### 本地运行

```bash
# 克隆仓库
git clone https://gitee.com/psycheclaritas/circuit_agent.git
cd circuit_agent

# 安装依赖
pip install -e .

# 启动 Web 服务
python -m app.api.server
```

服务启动后访问 http://127.0.0.1:8765

### 打包为单文件 exe

```bash
python tools/build_exe.py
```

打包完成后可执行文件位于 `dist/circuit_agent.exe`，已内置 ngspice 引擎。

## 使用方式

### Web 界面

启动服务后打开浏览器访问：
```
http://127.0.0.1:8765
```

支持：
- 拖拽上传电路图片/SVG 文件
- OCR 自动识别元件
- 手动编辑元件参数
- 查看求解结果和验证信息

### API 接口

#### 健康检查
```bash
GET /api/health
```

#### 导入电路
```bash
POST /api/import
Content-Type: multipart/form-data

file: <电路文件>
tol: 6.0  # 拓扑容差
ref_node: "0"  # 参考节点（可选）
```

#### 求解电路
```bash
POST /api/solve
Content-Type: application/json

{
  "method": "mna",  # mna | branch | ngspice | auto
  "verify": true
}
```

#### 渲染 SVG
```bash
POST /api/render
Content-Type: application/json

{
  "mode": "auto",  # auto | grid | radial
  "dark": false
}
```

#### SPICE 互转
```bash
POST /api/from-spice
Content-Type: application/json

{
  "netlist": "* 示例电路\nV1 1 0 10\nR1 1 2 1k\nR2 2 0 2k"
}
```

### MCP Server (LLM 集成)

支持作为 MCP 工具供大语言模型调用：

```bash
python mcp_server/server.py
```

提供三个工具：
- `solve`: 求解电路
- `validate`: 验证求解结果
- `elements`: 获取电路元件列表

## 项目结构

```
circuit_agent/
├── app/
│   ├── api/          # FastAPI Web 服务
│   ├── ingest/       # 电路输入解析
│   │   ├── kicad_in.py    # KiCad 网表解析
│   │   ├── svg_in.py      # SVG 解析
│   │   ├── topology.py    # 拓扑分析
│   │   └── values.py      # 参数解析
│   ├── ir/           # 内部电路表示
│   │   ├── model.py       # 数据模型
│   │   ├── params.py      # 参数体系
│   │   ├── render.py      # SVG 渲染
│   │   ├── spice.py       # SPICE 互转
│   │   └── probes.py      # 探针处理
│   ├── solver/       # 求解器
│   │   ├── mna.py         # 节点电压法
│   │   ├── branch.py      # 支路电流法
│   │   ├── dc_reduce.py   # DC 化简
│   │   ├── controlled.py  # 受控源处理
│   │   ├── ngspice.py     # ngspice 接口
│   │   └── reconcile.py   # 结果对账
│   └── vision/       # 计算机视觉
│       ├── ocr.py         # OCR 识别
│       ├── symbols.py     # 元件检测
│       ├── wires.py       # 连线提取
│       └── pipeline.py    # 处理流程
├── tests/            # 测试套件
├── tools/            # 构建工具
├── web/              # 前端页面
└── mcp_server/       # MCP 服务
```

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 单独测试模块
python tests/test_solver.py
python tests/test_controlled.py
python tests/test_vision.py
python tests/test_api.py
```

## 配置

配置文件位于 `config/secrets.json`（需从示例创建）：

```json
{
  "vlm_api_key": "your-api-key",
  "vlm_endpoint": "https://api.example.com/vision"
}
```

OCR 配置可通过 Web 界面或 API 修改。

## 许可证

MIT License - 详见 LICENSE 文件。