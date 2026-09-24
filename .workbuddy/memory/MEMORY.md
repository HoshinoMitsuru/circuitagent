# circuit_agent — 项目长期约定

## 一句话
照片/SVG/KiCad → 规范中间表示（IR）→ 三法独立求解并对账，服务"算题"。
根目录 `D:\Psyche\on campus\trial\circuit_agent`。

## 环境（**venv 不在项目目录里**）
- 解释器：`C:\Users\Psyche\.workbuddy\binaries\python\envs\circuit_agent\Scripts\python.exe`（Python 3.13.14）
- 已装：numpy / Pillow / opencv-python-headless / scipy / sympy / lxml / fastapi / uvicorn[standard] / python-multipart / PySpice / **httpx**（`tests/test_api.py` 用 FastAPI TestClient 需要它）
- ngspice：`find_ngspice_dll()` 按 `CIRCUIT_AGENT_NGSPICE_DLL` → KiCad 10.0 安装目录 → PySpice 自带 顺序搜索。
  KiCad 目录 `C:\Users\Psyche\AppData\Local\Programs\KiCad\10.0`。
  `NgSpiceShared.LIBRARY_PATH` 必须指向 **dll 文件本身**；PySpice 1.5 会报 `Unsupported Ngspice version 46` 但 .op 可用。

## 自检（九套 + 一道跨工具对账 + 打包验证，改任何一层都要全跑）
```
PY=C:/Users/Psyche/.workbuddy/binaries/python/envs/circuit_agent/Scripts/python.exe
"$PY" tests/test_solver.py          # 求解五电路（含教材题 4-17）+ 戴维南
"$PY" tests/test_dc_reduce.py       # C 开路/L 短路化简：五类元件处理 + 无解判定 + 电流反算（含 ngspice 独立复核）
"$PY" tests/test_api.py             # 端点回归：含 C/L 必须 200、无解给可读原因、位图通道、视觉设置读写
"$PY" tests/test_ingest_svg.py      # 跨线/圆点/T接三规则 + 回绘往返 + 布局自检
"$PY" tests/test_ingest_kicad.py    # 手写网表断言 + 14 张 KiCad demo 冒烟
"$PY" tests/test_spice.py           # 网表读写往返 + DC前缀/工程记法 + 非直流激励拒绝
"$PY" tests/test_vision.py          # 视觉层：本地端到端、两条铁律、OCR 修正表、位号拆段拼接、升级闸门、请求契约
"$PY" tests/test_ngspice_threadsafe.py  # ★ 并发首调探针/求解（**必须单独起进程**，见下）
"$PY" tests/test_temp_sweep.py      # ★ 残留解压目录清理的四道闸门（沙箱里跑，不碰真实 %TEMP%）
"$PY" tools/crosscheck_sharedumbrella.py [--spice]   # 跨实现对账（见下）
"$PY" runs/_probe_draw_static.py    # 画布层静态自检（无头浏览器不可用时的替代，见下）
```
打包时另加（改了 `app/`、`web/`、`launcher.py`、spec 之后必跑）：
```
"$PY" tools/build_exe.py            # 归置 ngspice + 生成图标 + 打包 + 铺第三方声明
"$PY" tools/verify_exe.py [--keep]  # 在**空目录**里跑 exe：路径分家 / 三法 / OCR / 算题 / 落盘 / 横幅 / 收尾
```
**为什么要按这个粒度留测试**：每一条都对应一次"曾经真的错、而且不报错"的缺陷
（`dc_reduce` 静默删元件、`/api/solve` 的 500、`ocr.recognize` 不收 `Path`、
`config.relative_to(ROOT)` 越界、`blank_text` 死开关、PySpice 的 `api.h` 没被打进包、
并发首调假报求解失败、清理失败被 `ignore_errors=True` 藏住）。

**探针（量数据用，不是断言用）**：`tests/_probe_ocr_lanelen.py`（行长/字号/字距）、
`_probe_ocr_{drop,merge,confusion}.py`、`_probe_{symbols,wires,groupfeat,holeratio}.py`。
有断言的两个是 `_probe_symbols` / `_probe_wires`（打印 `FAILS = 0`）。

## ★ 视觉层（位图通道）两条硬约定
1. **配置路径一律走 `config.display_path()`**，永远不写 `p.relative_to(ROOT)` ——
   密钥文件一旦不在项目根下（测试重定向、挪配置目录、共享盘）就抛 `ValueError`，
   而报错的那句正是"告诉用户去哪填 key"的提示，等于自己把请求打成 500。
2. **参数型功能必须测往返**（写进去 → 重读盘 → 读回来一致），
   不能只断言"字段在响应里"。`blank_text` 当年就是 `to_dict`/`to_file_dict`/
   `save_config.known` 三处都漏了它 → 界面有勾选框、永远显示未勾、取消也静默无效。

### 位图通道的两级编排（细节见技能文档 §10）
- 本地层（几何 + 模板 + 系统 OCR）优先；**只有结构性失败**才升级给视觉大模型。
  非结构性失败（某数值读不清）本地照出网表 + 标 `needs_human`，不花 API 费用。
- 采信 VLM ≠ 隐去不确定性：必须记 `evidence.source="vlm"`、留原始返回、跑叠图核对。
- **位图 IR 没有节点坐标**，`render_overlay_payload` 对它无能为力；
  叠图走 `pipeline.overlay_payload()`（视觉层自己那份，含"有字没读出来""未解释墨迹"）。
- OCR 失败要分 soft / structural：`engine_missing`/`lang_missing` 是正常分支，
  `image_unreadable`/`engine_error` 是结构性 —— 后者若被记成 soft，
  就会出现"整图位号数值全空、网表照样出、还不升级"这种最危险的静默缺值。
- **`app/vision/__init__.py` 故意不导入子模块**（WinRT 依赖 + 循环导入），
  用法一律 `from app.vision import pipeline as VP` 显式指名。

### ★ Windows OCR 的实测硬约束（影响合成测试图与回绘 SVG）
- **短文本行会被整条丢弃且无解**：`"R1"` 在 18~56px 每一档都是 0 个词。
  → **位号不要单独占一行**（合成图、回绘 SVG、给用户的标注指引都要守这条）。
- **字号影响非单调**：`"R1 10k"` 在 620×480 图上 26px 一个字都没有；
  测试图字号必须挑实测干净的档位（现用 20px），而且**要把实得词表打印出来**。
- **字距大时一个 token 被切成多个词**（`R12`→`['R','1','2']`），不是"含空格的单词"。
- **缝宽不能当判据**：token 内缝 0.24~1.44 字高，正常词间空格 0.39~0.80 字高，**区间重叠**；
  判"被拆散"只能靠形状。
- 分界：**位号可拼，数值只报不拼**（`10k` 与 `1.0k` 小数点被吞后形状相同、差 10 倍，
  而这类错三法互校永远抓不到 —— 三条路径共用同一份 IR）。

## ★★ 化简层的静默降级（本项目最严重的一次缺陷，务必别再犯）

C→开路、L→短路的化简发生在**三条求解路径之前**，所以
**三法互校 / 跨实现对账 / 功率守恒全部拦不住**：大家解的都是那张被化简歪掉的电路，
而且全零解让 ΣP = 0 平凡成立。

**事故经过**：`V1=10V(1-0)` 与 `L1(1-0)` 并联，再串 `R1`、`R2`。
直流下 L 短路 → V1 被短路 → 约束矛盾 → **无解**。
旧 `dc_reduce` 把"两端落进同一节点"的元件一律当"被短接即消失"删掉 →
**电压源凭空消失** → 剩余电路可解 → 全零解 → `overall_pass=True`，
并输出"这张电路的读图与建模**可以采信**"。
（ngspice 独立复核原电路：`singular matrix: check node l1#branch`，三种收敛策略全失败。）

### 「两端落进同一节点」的五类元件，处理各不相同（**永不**一概删掉）
| 元件 | 支路方程 | 两端等电位后的结论 |
|---|---|---|
| `R` | `u = R·i` | `R≠0` ⇒ **i = 0**（元件方程） |
| `I` | `i = I_s` | 与 u 无关 ⇒ **i = I_s** |
| `V` (E≠0) | `u = E` | `0 = E` ⇒ **无解，抛 `CircuitUnsatisfiable`** |
| `V` (E=0) | `u = 0` | 给不出 i ⇒ **i 待定**（与别的理想短线并联时不唯一） |
| `L` | `u = 0` 恒成立 | ⇒ **i 待定** |

### 配套的三条硬规则
1. **`report` 里不同原因的移除必须分开存**（`opened` / `shorted` 两个 list）。
   早先混装一个 `removed` 列表、`summary()` 整列冠以"电容视为开路"，
   于是**没有电容的电路**输出"电容视为开路，移除 2 条支路（L1, R1)"。
2. **被合并掉的节点名 ≠ 孤立节点**。孤立节点要用**合并后的代表元**集合去比：
   `rewritten` 里的节点名已被换成代表元，拿它与原节点名相减会把被合并掉的原节点全误报成"孤立"。
3. **被移除支路的电流要物归原主**（`recover_shorted_currents`，在 `run_all` 拿到 MNA 解之后调用）：
   元件方程能定的直接给数，定不了的用 KCL 在**合并前的原节点**上反算
   （未知量=这些支路电流；已知量=已解出支路在各原节点上的注入，**用原件的 `declared_direction`**）。
   反算欠定 → 明说"待定"，不许猜；反算不相容 → 报警（那是化简/求解层的 bug）。
   `_rref_solve` 与 `linalg.solve_linear` **分开写**：后者奇异就抛，而这里"奇异"是有物理含义的正常结论。

### 参考节点不许改名
`_UF(prefer=circuit.ref_node)` 让参考节点恒为代表元。否则 `L1(1-0)` 会让地改名成"节点 1"，
报告与插图全都难读。

### 同类防护（可推广）
- 变换前后各列一次「约束清单」，核对是否有约束凭空消失；
- **把"全零解"当报警信号**（有独立源却全 0 ⇒ 某处约束被静默丢掉；ΣP 在这里恒成立，拦不住）；
- 用 ngspice 复核**原电路**（把 L 换成 0V 源，直流下二者等价）→ 无解会暴露成 singular matrix。

## ★ `/api/solve` 曾对任何含 C/L 的电路返回 HTTP 500
根因：`format_text_report` 用**原电路**的节点名/位号去 `next()` 查**化简后**电路生成的
对账表 → 被合并的节点、被移除的支路在表里都不存在 → `StopIteration`；
而那一句不在任何 `try` 里（`format_text_report` 是派生视图，失败不该毁掉整个响应）。
修法：主解一节**直接遍历对账表**（表内自带 `node`/`ref`），并把渲染异常收进
`text_report_error` 显式返回（不静默、也不是"没有报告就是没问题"）。
`api_solve` 现在统一 200 + `{ok, stage, error}`，`CircuitUnsatisfiable` 走 `CircuitError` 分支。

## 跨工具对账（补三法互校的盲区）
**三法互校有个共同盲区**：三条路径共用同一份 IR，所以「网表抄错 / 参考方向定反 /
参考节点选错」这类错误，三法会**一致地**给出同一个错答案，互校拦不住。
补法是换一套**独立写的实现**喂同一份网表 —— 用 SharedUmbrella 项目那份
`D:\Psyche\SharedUmbrella\.workbuddy\tools\circuit_dc_solve.py`（同一作者、独立编写）。
对账入口：`tools/crosscheck_sharedumbrella.py`，默认跑 `examples/` 下 5 份网表。
当前结果：**节点电压 / 支路电流 / 每元件功率 / 功率守恒 逐项严格相等，差异项 0**。

### ★ 两套实现的符号约定**不同**，对账前必须映射（否则把约定差异误报成算错）
| 量 | circuit_agent | SharedUmbrella | 映射 |
|---|---|---|---|
| 电压源 +端 | `nodes[0]` | 网表第一个节点 | 同 |
| **电压源电流参考方向** | 内部 `−→+`（`nodes[1]→nodes[0]`），`i>0` = **供电** | `+→−`（`n1→n2`），`i<0` = **供电** | **取相反数** |
| 电阻/电流源参考方向 | `nodes[0]→nodes[1]` | `n1→n2` | 同 |
| 功率 | `P = drop × i`，吸收为正 | 同 | 同 |

实测佐证：`net_isrc_2.txt` 里 V1 明确在供电，`nodal()` 给 `isrc=+3`、
`branch_current()` 给 `ib=−3` —— **同一文件内两个函数对同一物理量取了相反号**，
且都叫 `isrc`/`i(V1)`。circuit_agent 站在 `+3` 那一侧。
对账时 ngspice 一节**必须用容差**（double 两次运行本身有 ~1e-10 噪声），其余用精确 `==`。

## 不可动摇的设计约束（血泪换来的）
1. **不许靠肉眼定连接**：只把「线段端点」和「圆点圆心」当候选结点，**几何交点不生成结点**。
   于是纯 X 跨线天然不相连（构造的自然结果，非特判）；T 接无圆点判**相连**（教科书惯例）。
2. **不许只算一遍**：节点电压法（Fraction 精确 MNA）、支路电流法（生成树基本回路）、
   ngspice（浮点）三条**独立代码路径**，逐项对账。幂守恒 ΣP=0 是最强单条判据。
3. **符号本体不是导线**：电阻框/电容极板/电感弧串都是闭环导体，参与连通就会
   **把元件两端短接**。`build_topology` 的顺序固定为
   **挑槽位 → 算本体圆柱 `slot_region` → 剔除体内线段 → 建连通图 → 生成元件**。
4. **不静默降级**：跳过、剔除、置信度不足、需人工确认的，一律写进 `diagnostics` /
   `origin.warnings` / `Evidence`，并在 WebUI 上呈现。宁可报"需人工确认"，不许猜。

## IR 约定
- `Component.nodes` 次序即参考方向：`P = drop × i`（沿参考方向）。
  电阻 `R·i²`、电压源 `−E·i` 都由此推出。
- 电压源参考方向：`declared_direction(V) = (nodes[1], nodes[0])`（内部 −→+），
  故 `i > 0` 表示该电源**供电**。电压源符号的 `+` 画在 `nodes[0]` 一侧。
- `REF_NODE = "0"`；`ALLOWED_KINDS = {R,V,I,C,L}`；`SOLVABLE_KINDS = {R,V,I}`；
  `CONFIDENCE_GATE = 0.85`（低于此值强制人工确认）。
- 直流稳态化简：C→开路（移除支路）、L→短路（并查集并节点），必须留痕。
- `CircuitError` 是"**输入有问题**（漏画线/缺数值）→ 请改图"；
  `CircuitUnsatisfiable(CircuitError)` 是"**输入没问题、题目理想化模型自相矛盾** → 请改题"。
  两者话术必须分开，否则学生会一直去查自己的读图而其实题目本身无解。

## 渲染器 ↔ 解析器的往返契约
- 回绘 SVG 带语义标记 `data-ca-*`（ref/kind/a/b/p1/p2/value），
  **几何仍是权威**（坐标去定位几何结点、IR 节点名去命名），标记只提供身份与取值。
- 装饰必须显式声明：`class="ca-bg"`（背景）/ `ca-deco` / `ca-node-mark`（度为 2 的装饰小圆）。
  解析侧 `_is_decoration` 同时认 `stroke="none"`，**但 circle/ellipse 豁免**。
- 符号本体必须画到「导线从节点连到符号**本体边缘**」（`BODY_HALF` 与 `_glyph_*` 严格对齐）。
- 图形元素**不许伸到离端子 6px（容差）以内**，否则会粘住导线端点使其不算自由端点。
- `render_svg(..., diagnostics={})` 会回填 `{"layout","overlaps","roundtrip_safe"}`；
  **网格布局对同层密集图（电桥）会重叠，其输出不可用于几何回导。**

## ★ 手绘面板（第三类输入源；细节见技能文档 §11）
`web/index.html` 的 `data-p="draw"` 面板 + `app/api/server.py` 的
`/api/symbols`（元件定义，吃 `render.GLYPHS` + `render.style_block`）、
`/api/layout`（把已有电路打进画布，`mode=geom/radial/grid/auto`）、
`/api/from-svg`（画布提交入口，**走与导入 SVG 完全相同的 `svg_to_ir`**）。

四条最容易错的：
1. **画布绝不写 `data-ca-a/b`** —— 那是节点名的**强提示**，写上去等于把名字硬编码成
   1/0，用户画的接法反而作废。手绘图的名字正是要**算出来**的。只写
   `ref/kind/value/p1/p2`。
2. **T 接切开（`dNormalize`）不是导出时才做，而是每次几何一变就做** ——
   入口四个：`dPlace` / `dAddWire` / `onCanvasUp` / `loadToDraw`。这样画布看到的、
   `dAnalyze` 判的、提交出去的是**同一个几何**。（结点只认端点与圆点圆心，交点不是端点。）
3. **CSS 特异性**：画布要内联后端给的 `.circuit-svg` 规则，所以画布自己的规则
   必须带 `#dCanvas` 前缀；动态颜色走行内 style。容器 `.dwrap` **定高 440px**。
4. **拖动期间冻结视口**（viewBox 随内容外扩 → 不冻则每帧缩放比在变，"越拖越飘"），
   **`pointermove/up` 挂 `window`**（重画会换掉 `#dCanvas`，挂它身上连指针捕获一起失效）。

`/api/from-svg` 的会话处理：**沿用同一个 sid**（不换 id，避免留别名键）；
**绝不覆盖 `source.*`**，手绘版另存 `hand.svg`；换通道时清
`vision/vision_obj/vision_ocr_edits` 并把 `viewbox`/`image_size` 换成手绘那份；
`original_source`/`origin_channel` **从上一份报告继承**（第二次改时 `source_path`
已是 `hand.svg`，照抄它会让来源链断在第一环）。

**画布层静态自检** `runs/_probe_draw_static.py`：语法、标签↔面板 1:1、
JS 引用的 DOM id 都有定义、`d*` 函数都有定义、产出的 class 都有样式、
改几何的每条路径都归一化、撤销/重做接线齐、提交路径不含 `data-ca-a/b`。
**它覆盖不了手感**（拖拽跟不跟手、切开的线看着对不对）—— 那一步只能人看。

## ★★ ngspice 这一整块必须串行（`_NG_LOCK`）

PySpice 的 NgSpice 绑定有**两处进程级全局状态**，都不在我们手里：

1. `PySpice/Spice/NgSpice/Shared.py:110` 的 `ffi = FFI()` 是**模块级单例**，
   而 `_load_library()` 每次构造实例都无脑 `ffi.cdef(api.h 全文)`。
   cffi 不许同一个 FFI 重复声明同一 struct → **第二次 cdef 必抛**
   `CDefError: duplicate declaration of struct ngcomplex`。
2. `NgSpiceShared._instances` 按 id 缓存实例，但"查缓存→构造→写回"**不是原子的**。

叠加 → **两个线程同时第一次构造，必崩一个**。实测（8 线程同时首调探针）：
**1 成功 / 7 失败**。FastAPI 的同步端点跑在**线程池**里，启动那一刻启动器的
就绪轮询与外部检查会**同时**打 `/api/health`，`/api/solve` 之间也会并发 ——
于是表现为「同一道题偶尔报求解失败、重试又好了」，**单线程怎么跑都复现不出来**。

修法：`app/solver/ngspice.py` 的 `_NG_LOCK`，**探针与真解共用同一把锁**
（抢的是同一份全局状态，分成两把等于没加）；`probe_availability()` 用双检锁。
`ngspice_method` 只是薄壳，实体在 `_solve_locked`（**只在持锁时调用**）。
回归：`tests/test_ngspice_threadsafe.py`，**必须单独起进程** ——
`ffi` 一旦被 cdef 过竞态就再也复现不出来，该文件第 ⓪ 节会先自检这一点，
不满足就如实报"跳过"而不是假装通过。

## ★ 打包成单文件 exe（`dist/circuit_agent.exe`，~106 MB）

四个决策（用户确认）：**onefile** / **ngspice.dll 打进包** / **runs+config 放 exe 同级** /
**保留控制台**。工具链：`tools/build_exe.py`（4 步）+ `tools/verify_exe.py`（干净目录端到端）。

- **资源根 ≠ 数据根**（`app/paths.py`）：onefile 的资源根是 `_MEIPASS`（退出即失），
  数据根是 **exe 同级**。不分家的话 `runs/` 会落进临时目录、**退出就没了且不报错**。
  `resource_root()`（只读，web/ 与 vendor/ 在这）vs `data_root()`（可写，
  便携 → `%LOCALAPPDATA%\circuit_agent` → 临时目录，**每一级都带 `data_root_reason`**）。
  测试/探针仍 monkeypatch `server.RUNS_DIR`，所以 `RUNS_DIR` 保持模块级全局。
- **`bundled_ngspice()` 排在 KiCad 之前**（环境变量仍最优先）：先用自己的那份，
  版本才是测过的那个。`origin` 字段如实说明来源。
- **探针必须真跑一次 `.op`**，不能只 `import`。旧实现只 import → 横幅写
  「● ngspice 可用」而第三法其实是空的（见下条），**且没有任何地方告诉用户**。
  现在判据是「跑完 1V/1Ω 并核对 V(1)=1.0 与 |i|=1.0」，并把 `verified` 字段
  打上首屏。
- **★ PySpice 的 `api.h` / `logging.yml` 是运行时读的**数据文件，PyInstaller
  默认只收 `.py` 与二进制 → 必须 `collect_data_files("PySpice", include_py_files=False)`
  （实测恰好这两个）。**不打包时一切正常**，打出来之后每道题的第三法都失败。
- 另需 `collect_submodules("winrt")`（命名空间包 + 原生扩展，OCR 的 import 在函数体里，
  静态分析看不到；少收的表现是"OCR 不可用"这个**看起来完全正常**的降级分支）
  与 `collect_submodules("uvicorn")`（协议/事件循环是运行时按名挑的）。
- `vendor/ngspice/`（dll + `LICENSE.ngspice.txt`）**必须入库**：它是构建输入，
  不是产物。没它则构建出的包**悄悄只剩两法**。发版时 `dist/THIRD-PARTY-NOTICES.txt`
  要和 exe 一起给。
- 重建失败若只有一句 `PyInstaller 返回 1`：用 `_exe_locked()` 判——
  判据是 `open(path, "r+b")` **Permission denied**（实测：运行中的 exe
  **改得了名**、删不掉、打不开写；用 rename 判会永远漏判）。

### ★★ onefile 是"父进程 + 子进程"，只杀父进程会留孤儿

PyInstaller 的 onefile 形态：**父**进程把包解压到 `_MEI*` → 再 `CreateProcess`
一个**子**进程跑真应用，父进程只是等着。所以 `subprocess.Popen` 拿到的是**父**进程：

- 只 `terminate()` 父进程 → 正在跑服务、占着端口与 `%TEMP%\_MEI*` 的**子进程变孤儿**。
  症状：下次启动报"端口被占"、临时目录删不掉、`runs/` 行为异常。
  必须 `taskkill /PID <父pid> /T /F`，且**要在父进程还活着时执行**（父一死就找不到子）。
- `tools/verify_exe.py` 的 `stop_tree()` 就是干这个的；收尾还断言
  **端口真的释放了**（比"进程对象说它退了"可信）。

### ★ 残留解压目录：强杀一次漏 ~250 MB（`sweep_stale_mei`）

单文件模式每次启动往 `%TEMP%\_MEI*` 解压 ~250 MB（比 exe 本身大一倍），
正常退出由引导器删掉；**只要上次是被强杀的**就留在那里。实测本机被测一下午
攒了 **19 个 / 3.0 GB**。`launcher.sweep_stale_mei()` 在启动时清，**三道闸门**：

1. 只认临时目录**顶层**的 `_MEI*`，别的文件/目录一律不碰；
2. **时效门槛**（默认 2 小时）：太新的不动 —— 刚启动的实例正在往自己那个目录里
   解压，那时可能还没有任何打开的文件句柄，光靠第 3 条判不出来；
3. **改名试锁**：Windows 上**在被使用的目录改不了名**（实测「拒绝访问」，
   因为引导器持有目录句柄且未开 `FILE_SHARE_DELETE`）。改名成功 = 没人用它，才敢删；
   删不掉就改回原名，不留 `.stale` 怪名。自己的 `sys._MEIPASS` 永远跳过。

`CIRCUIT_AGENT_NO_TEMP_SWEEP=1` 可关；`CIRCUIT_AGENT_TEMP_SWEEP_AGE` 覆盖门槛（测试用）。
**绝不悄悄删**：清理了几个、回收多少 MB 都打在首屏横幅上。
测试 `tests/test_temp_sweep.py`（沙箱里跑，四道闸门逐个盯，含"持句柄时不许删"的对照）。

★ 同理教训：`tools/verify_exe.py` 原来写 `shutil.rmtree(work, ignore_errors=True)`，
把"删不掉"完全藏住了 —— 于是悄悄攒了 8 个临时目录。**清理失败必须能被看见。**
