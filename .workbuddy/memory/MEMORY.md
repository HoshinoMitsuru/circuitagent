# circuit_agent — 长期约定

照片/SVG/KiCad/手绘 → 规范中间表示（IR）→ 三法独立求解并对账，服务"算题"。
根目录 `D:\Psyche\on campus\trial\circuit_agent`。
**事故全程与推演在每日日志；可推广的方法论在技能文档 `circuit-photo-to-solution/SKILL.md`（§1~§12）。**
本文件只留"必须照做"的常量、规则与索引。

## 环境
- 解释器 `C:\Users\Psyche\.workbuddy\binaries\python\envs\circuit_agent\Scripts\python.exe`
  （**venv 不在项目目录里**）。已装 numpy/Pillow/opencv/…/PySpice/**httpx**（TestClient 需要）。
- `find_ngspice_dll()` 搜索序：`CIRCUIT_AGENT_NGSPICE_DLL` → **随包自带** → KiCad 10.0 → PySpice 自带。
  `NgSpiceShared.LIBRARY_PATH` 必须指向 **dll 文件本身**；PySpice 1.5 报
  `Unsupported Ngspice version 46` 但 `.op` 可用。

## 自检（改任何一层都要全跑）
```
PY=C:/Users/Psyche/.workbuddy/binaries/python/envs/circuit_agent/Scripts/python.exe
"$PY" tests/test_solver.py        # 五电路（含教材题 4-17）+ 戴维南
"$PY" tests/test_dc_reduce.py     # C 开路/L 短路化简 + 无解判定 + 电流反算（含 ngspice 复核）
"$PY" tests/test_api.py           # 端点回归 + 受控源网表端到端 + /api/params 缓存语义
"$PY" tests/test_ingest_svg.py    # 跨线/圆点/T接三规则 + 回绘往返 + 布局自检
"$PY" tests/test_ingest_kicad.py  # 手写网表断言 + 14 张 KiCad demo 冒烟
"$PY" tests/test_spice.py         # 网表读写往返 + DC 前缀/工程记法 + 非直流激励拒绝
"$PY" tests/test_vision.py        # 视觉层：本地端到端、两条铁律、OCR 修正表、升级闸门
"$PY" tests/test_controlled.py    # E/G/H/F + 表达式：三路径对账 + 探针 + 往返幂等 + 坏标记拒绝
"$PY" tests/test_params.py        # 参数体系：命名/映射/单位/改名/控制关系/原子性/孤儿
"$PY" tests/test_ngspice_threadsafe.py  # ★ 并发首调（**必须单独起进程**）
"$PY" tests/test_temp_sweep.py    # ★ 残留 _MEI 清理四道闸门（沙箱里跑，不碰真实 %TEMP%）
"$PY" tools/crosscheck_sharedumbrella.py [--spice]   # 跨实现对账
"$PY" runs/_probe_draw_static.py  # 画布层静态自检（含"参数页字段名 ↔ 后端"对齐）
"$PY" tools/build_exe.py && "$PY" tools/verify_exe.py [--keep]   # 改了 app/ web/ launcher.py spec 之后
```
**粒度为什么这么细**：每条都对应一次"曾经真的错、而且不报错"的缺陷。
**探针**（量数据用，非断言）：`tests/_probe_ocr_*.py`、`tests/_probe_{symbols,wires,groupfeat,holeratio}.py`；
有断言的是 `_probe_symbols`/`_probe_wires`。

## 不可动摇的设计约束
1. **不许靠肉眼定连接**：只把「线段端点」和「圆点圆心」当候选结点，**几何交点不生成结点**。
   → 纯 X 跨线天然不相连；T 接无圆点判**相连**（教科书惯例）。
2. **不许只算一遍**：节点电压法（Fraction 精确 MNA）、支路电流法（生成树基本回路）、
   ngspice（浮点）三条**独立代码路径**逐项对账。ΣP=0 是最强单条判据。
3. **符号本体不是导线**：电阻框/电容极板/电感弧串都是闭环导体，参与连通会把元件两端**短接**。
   `build_topology` 顺序固定：**挑槽位 → 算本体圆柱 `slot_region` → 剔除体内线段 → 建连通图 → 生成元件**。
4. **不静默降级**：跳过、剔除、置信度不足、需人工确认的，一律写进 `diagnostics` /
   `origin.warnings` / `Evidence`，并在 WebUI 上呈现。宁可报"需人工确认"，不许猜。
5. **任何技术分叉先问**：给出选项与取舍，用户拍板后才动代码。

## IR 约定
- `Component.nodes` 次序即参考方向：`P = drop × i`（沿参考方向）。
- 电压源 `declared_direction = (nodes[1], nodes[0])`（内部 −→+），`i > 0` = **供电**；
  符号的 `+` 画在 `nodes[0]` 一侧。**判"电压输出"必须用 `VOLTAGE_OUTPUT_KINDS`，不许写 `kind=="V"`。**
- `REF_NODE="0"`；`CONFIDENCE_GATE=0.85`；`INDEPENDENT_KINDS={R,V,I,C,L}`；
  `CONTROLLED_KINDS={E,G,H,F}`；`ALLOWED_KINDS`=两者并；`SOLVABLE_KINDS={R,V,I}|CONTROLLED_KINDS`；
  `BITMAP_KINDS=INDEPENDENT_KINDS`（位图/VLM 只有 R/V/I/C/L 的字形）；`VOLTAGE_OUTPUT_KINDS={V,E,H}`。
- `CircuitError` = "**输入有问题**（漏画线/缺数值）→ 请改图"；
  `CircuitUnsatisfiable(CircuitError)` = "**输入没问题、题目理想化模型自相矛盾** → 请改题"。话术必须分开。
- 直流稳态化简：C→开路（移除支路）、L→短路（并查集并节点），必须留痕。

## ★★ 受控源与参数体系（详细推演见 SKILL §12 / 2026-09-24.md）
**这一层的错不会算错，只会指错。** 每条纪律都围绕"名字指向谁"。

- **四类**：`CONTROL_MODE={E:"V",G:"V",H:"I",F:"I"}`（**元件类型固有属性**，只在 `ir/params.py`
  定义一次，网表层转发）；`VALUE_UNIT={E:"V/V",G:"S",H:"Ω",F:"A/A"}`；`GAIN_SYMBOL={E:"μ",G:"gm",H:"rm",F:"α"}`。
  **增益单位是判断"受控源画对没有"的硬标准。** 受控源与非受控源**互斥**（`Component.__post_init__` 判死）。
- **流控 H/F 自动插 0V 探针源**（`ir/probes.py`，`SENSE_PREFIX="Vsense_"`）：
  `target.nodes=(a,mid)`、探针 `nodes=(b,mid)`。★ **端子顺序决定符号**（此时探针电流与 target 参考方向
  `a→b` 上电流**同号**）；反过来写三条路径会**一致地**差一个负号，互校发现不了。
  **幂等**（靠 `ctrl.sense_ref`）；**只改副本、绝不写回会话**；探针**四处留痕**。
  不插的两种：被采样支路本身就是电压源；受控源用了自定义表达式。
- ★★ **三个"支路身份"必须分开给**：`ref`=受控源自己的位号（`H1`）｜`sampled_ref`=**题目里说的**
  被采样支路（`R1`）｜`probe_ref`=系统插的探针（可为空）｜`sampling_ref`=实际取电流的。
  挤成一个 `sampling` 的后果：界面拿受控源自己的位号去选中"被采样支路"下拉框，而选项里排除了自己
  → 永远显示"（未定）"。**人读文字也必须指向 `sampled_ref`**（写 `i(Vsense_R1)` 用户对不上题目）。
- ★ **网表语义标记** `* ca-ctrl <ref> mode=I ref=<支路> sense=<探针>`：H/F 卡写的是探针，
  光看卡片分不开"题目说的 R1"与"系统插的 Vsense_R1"→ 往返一次 `ctrl.ref` 就变错。
  ① 标记与卡片不符 → **以卡片为准** + 原因写 diagnostics；② 外来网表无标记 → **原样不动**；
  ③ ★ 顺序：**先**按卡片查合法性，**再**用标记还原 `ref`（反过来会把跑不起来的网表"修好"）。
- **参数表**：**绑定是唯一匹配依据** `("node_u",结点)`/`("branch_i",位号)`/`("branch_u",位号)`/`("value",位号)`。
  默认名从**身份**派生（`value→位号`、`branch_i→i_R1`、`branch_u→u_R1`、`node_u→u_3`，撞车加 `_2`）。
  每行带 **SI 单位**；**C/L 不参与直流求解但必须在表里占位**；删元件 → 名字留成**孤儿**并显式报出。
  `params_view()` 是**唯一**的"参数名 ↔ 绑定"翻译处（报告、界面、表达式校验三处都读它）。
  报告按绑定找名字读 `by_binder`，**不要自己拼名字字符串**。
- ★ **改名要连带改表达式，且不许重排用户手写的格式**：`str.replace` 会把 `u_1→u_10` 时已有的 `u_10`
  变成 `u_100`；"拆记号再拼回去"会**抹掉用户写的空格**（`3*u_2 + 5`→`3*uA+5`）；
  正解 = **带记号边界的正则** + 一道 tokenize 语法关卡（坏表达式原文不动）。
- ★ **改名保缓存、改方程清缓存**：缓存按**绑定**存（`"branch_i:R1"->"1/250"`）→ 改名不失效；
  只改名 → 留着；改表达式/改控制关系/换 IR → **当场清空**（否则拿旧数配新方程，长得和正确答案一样）。
  清空在**服务端**（`Session.put_ir()` 唯一入口 + `/api/params` 额外清一次）。
- ★ **参数编辑是一个事务**：副本上做完再写回，任一项非法 → **整包原样**；写回**逐字段**搬
  `c.ctrl.expr/nodes/ref/sense_ref`，**不整块** `c.ctrl = other.ctrl`。
- **表达式只认线性组合**（`params.parse_linear`，未知名词当场拦住并列出可用名字）；`MAX_EXPR_DEPTH=8`；
  渲染成 ngspice 文本走 `ir/spice_expr.py`。★★ **IR 的电压源电流与 SPICE 的 `I(Vx)` 互为相反数**，
  只此一处定义：`SPICE_SOURCE_CURRENT_SIGN = -1`（表达式 `I(...)` 片段 + H/F 卡增益两处共用）。

## 遍历/解析的两条通用纪律
- **解析器的元件类型判据必须跟着 `ALLOWED_KINDS` 走**。写死五种会**静默丢掉所有受控源**；
  而 `skipped_components` 若没有消费方，就是"图上少了个元件却一点提示都没有"。
  被跳过的必须**按原因分类逐条**给（位号/机器可读原因/人读细节）并一路送到界面。
- **配置/数据路径一律走专用函数**（`config.display_path()`、`paths.data_root()`），
  永不写 `p.relative_to(ROOT)`。**参数型功能必须测往返**（写入 → 重读盘 → 读回一致）——
  `blank_text` 当年三个序列化处都漏，表现为"界面有勾选框、永远显示未勾、取消也静默无效"。

## ★★ 三法互校的两个共同盲区（都补过了）
1. **化简层在其上游**：C→开路、L→短路发生在三条求解路径**之前** → 三法互校 / 跨实现对账 / 功率守恒
   全拦不住，且全零解让 ΣP=0 平凡成立。（旧 `dc_reduce` 把"两端落进同一节点"的元件一律删，
   于是**电压源凭空消失**、全零解、还宣称"读图可采信"。）
   - 「两端同一节点」五类元件**永不**一概删：`R`⇒i=0｜`I`⇒i=I_s｜`V(E≠0)`⇒**无解抛异常**｜
     `V(E=0)`/`L`⇒**i 待定**。
   - 移除原因**分开存**（`opened`/`shorted`）；**被合并掉的节点名 ≠ 孤立节点**（用代表元集合比）；
     被移除支路的电流要**物归原主**（`recover_shorted_currents`，KCL 在**合并前的原节点**上反算，
     欠定就说"待定"；`_rref_solve` 与 `linalg.solve_linear` **分开写**）。
   - **参考节点不许改名**（`_UF(prefer=circuit.ref_node)`）；**把"全零解"当报警信号**；
     ngspice 复核**原电路**（L 换 0V 源）。
2. **三法共用同一份 IR**：「网表抄错 / 参考方向定反 / 参考节点选错」三法会一致地给同一个错答案。
   补法 = 换一套**独立写的实现**喂同一份网表：SharedUmbrella 的
   `D:\Psyche\SharedUmbrella\.workbuddy\tools\circuit_dc_solve.py` → `tools/crosscheck_sharedumbrella.py`。
   当前差异项 0。**两套的电压源电流参考方向相反，对账前必须取相反数**（其余同号）；
   ngspice 一节**必须用容差**（~1e-10 噪声），其余精确 `==`。

## 往返契约（渲染器 ↔ 解析器）
- 回绘 SVG 带 `data-ca-*`（`ref/kind/a/b/p1/p2/value`，受控源另有 `ctrl-mode/p1/p2/ref/sense/expr`），
  **几何仍是权威**，标记只提供身份与取值。**手绘画布绝不写 `data-ca-a/b`**（那是节点名强提示，
  写上去等于把名字硬编码、用户画的接法作废）。
- 装饰必须显式声明 `class="ca-bg"`/`ca-deco`/`ca-node-mark"`；`_is_decoration` 也认 `stroke="none"`，
  **但 circle/ellipse 豁免**。**画在受控源上的控制线必须是 `ca-deco`**，否则重导入会多出导线。
- 符号本体画到「导线连到**本体边缘**」（`BODY_HALF` 与 `_glyph_*` 严格对齐）；图形元素**不许伸到
  离端子 6px 以内**，否则粘住导线端点。`render_svg(..., diagnostics={})` 回填
  `{"layout","overlaps","roundtrip_safe"}`。
- ★★ **已知未修**：自动版式把**并联两支路排在同一轴线** → `roundtrip_safe=False` → 回导时两符号叠加、
  两端判成同一结点 → 一起记为 `short_circuit` 消失（实测 5 元件电路只剩 3 个，`auto/radial/grid`
  都复现）。**三法互校、跨实现对账、功率守恒全拦不住**；能抓住它的只有版式自检 +
  `skipped_components` 结构化报出，外加纪律 **`roundtrip_safe=False` 的插图不许拿去几何回导**。
  修法在 `ir/render.py` 的布局算法（另案）。

## ★ 手绘面板（第三类输入）与静态自检
`data-p="draw"` 面板 + `/api/symbols`（吃 `render.GLYPHS`）、`/api/layout`（`geom/radial/grid/auto`）、
`/api/from-svg`（**走与导入 SVG 完全相同的 `svg_to_ir`**）。细节见 SKILL §11，四条最易错：
① 画布绝不写 `data-ca-a/b`，只写 `ref/kind/value/p1/p2`（受控源另写 `ctrl-*`，也是**坐标/位号**）；
② **T 接切开（`dNormalize`）每次几何一变就做**（`dPlace`/`dAddWire`/`onCanvasUp`/`loadToDraw`）；
③ 画布规则必须带 `#dCanvas` 前缀（它要内联后端 `.circuit-svg` 规则），`.dwrap` 定高 440px；
④ **拖动期间冻结视口**，**`pointermove/up` 挂 `window`**（重画会换掉 `#dCanvas`）。
`/api/from-svg` 会话：**沿用同一个 sid**、**绝不覆盖 `source.*`**（手绘版另存 `hand.svg`）、
换通道清 `vision*` 并换 `viewbox`/`image_size`、`original_source`/`origin_channel` **从上一份报告继承**。
`runs/_probe_draw_static.py`：语法、标签↔面板↔重画函数 1:1、DOM id、`d*` 函数、class 都有样式、
`S.pen` 三桶齐、每条改几何路径都归一化、撤销接线、提交路径不含 `data-ca-a/b`、
**卡片引用的字段后端都真的给了**。**它覆盖不了手感** —— 那一步只能人看。

## ★★ ngspice 必须串行（`_NG_LOCK`）
PySpice 有两处**进程级全局状态**：① `ffi = FFI()` 是**模块级单例**而 `_load_library()` 每次无脑
`ffi.cdef(api.h 全文)` → **第二次 cdef 必抛** `CDefError: duplicate declaration of struct ngcomplex`；
② `_instances` 缓存"查→构造→写回"**不原子**。
叠加 → **两线程同时首次构造必崩一个**（实测 8 线程：**1 成功/7 失败**）。FastAPI 同步端点跑在
**线程池** → 表现为「同一道题偶尔报求解失败、重试又好」，**单线程复现不出来**。
修法：`_NG_LOCK`，**探针与真解共用同一把锁**；`probe_availability()` 双检锁；实体在 `_solve_locked`
（**只持锁时调用**）。回归 `tests/test_ngspice_threadsafe.py` **必须单独起进程**（`ffi` 一旦被 cdef 过
竞态再也复现不出来；该文件第 ⓪ 节先自检，不满足就如实报"跳过"）。

## ★ 打包单文件 exe（`dist/circuit_agent.exe`）
onefile / ngspice 打进包 / runs+config 放 exe 同级 / 保留控制台｜`tools/build_exe.py` +
`tools/verify_exe.py`。**完整参考见 `2026-09-23.md` 附录 A/B/C**，四条最深：
① **资源根 ≠ 数据根**（`_MEIPASS` vs exe 同级），不分家则 `runs/` 落临时目录、退出就没且不报错 → `paths.data_root()`；
② **PySpice 的 `api.h`/`logging.yml` 是运行时读的** → 必须 `collect_data_files`，否则不打包正常、打包后第三法全失败；
③ **onefile 是父+子进程** → 必须 `taskkill /T /F` 且**父进程还活着时**执行；**验证跑完会留下父+子两个进程占住 exe**，
   下次打包前先确认没有 `circuit_agent.exe` 在跑（否则 PyInstaller 报"返回 1"而真实原因是文件被占）。
   ★ 杀进程要走 **PowerShell 工具**，别在 Git Bash 里写 `taskkill /PID`（`/PID` 被当路径转换，报"无效参数"）；
④ **强杀一次漏 ~250 MB** → `launcher.sweep_stale_mei()` 三道闸门（只认顶层 `_MEI*`、2 小时时效、改名试锁）。

## 日志索引
- `2026-09-20.md` — 视觉层/手绘面板/化简层事故全记录（最详）
- `2026-09-22.md` — 来源链、撤销重做、静态自检
- `2026-09-23.md` — 打包 exe、并发竞态、临时目录清理（附录承载打包全录）
- `2026-09-24.md` — 受控源 + 参数体系转正式测试，测试揪出三处"指错不改错"的缺陷；重打包 exe 验证 0 失败项
