"""第三方法：ngspice（第三方独立实现）。

**这不是"再算一遍"，是换代码路径**：前两法是本仓库自己的精确有理数实现，
这一条是 ngspice 的浮点稀疏求解器。两者一致到 1e-10 量级才算过。
共性错误（比如同一处符号约定写反）能被这条路抓出来，而"同一份代码跑两遍"抓不出来。

依赖：**真 ngspice**，通过 PySpice 调用共享库。
本机 ngspice.dll 来自 KiCad 10.0（`...\\KiCad\\10.0\\bin\\ngspice.dll`），
实测可用：`V1=10V / 100Ω + 200Ω∥300Ω` → `v(2) = 5.454545455`，
理论值 `10·120/220 = 5.4545454545`，对得上。

两条实测踩出来的坑，写在这里免得下次再踩：
1. ``NgSpiceShared.LIBRARY_PATH`` 必须**直接指向 dll 文件本身**。PySpice 默认
   找的是它自己 site-packages 下的 ``Spice64_dll/dll-vs/ngspice{}.dll``，不存在。
2. PySpice 1.5 对 ngspice 46 会报 ``Unsupported Ngspice version 46`` ——
   这只是它认不出新版枚举、退回 'last' 模式，**不影响 .op 取值**，别被吓到。
   （用户要是看到这条 warning，可以在报告里注明它是无害的。）
"""

from __future__ import annotations

import os
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..ir.model import Circuit, CircuitError
from ..paths import bundled_ngspice
from .base import Solution, declared_direction


# ---------------------------------------------------------------- 定位 DLL

DLL_ENV = "CIRCUIT_AGENT_NGSPICE_DLL"

_CANDIDATES = [
    r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\7.0\bin\ngspice.dll",
]


def find_ngspice_dll() -> Path | None:
    """按 环境变量 → **包内自带** → KiCad 常见安装位 → PySpice 自带 的次序找 ngspice.dll。

    ★ "包内自带"排在 KiCad 之前：打包成 exe 分发时，我们**主动带了一份**
      ngspice.dll 进去。它是这份 exe 自带的依赖，比用户机器上碰巧装的那个
      KiCad 更可预期 —— 用户可能装的是 7.0、可能是别人机器上残留的旧版、
      也可能压根没装。**先用自己的那一份，版本才是我测过的那个。**
      （但环境变量仍然排第一：那是用户明确的、逐次生效的覆盖意愿。）
    """
    env = os.environ.get(DLL_ENV)
    if env and Path(env).is_file():
        return Path(env)

    #: 包内自带的那一份（开发时是仓库里的 vendor/，打包后是 _MEIPASS 根）
    ours = bundled_ngspice()
    if ours is not None:
        return ours

    for p in _CANDIDATES:
        if Path(p).is_file():
            return Path(p)

    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for ver in ("10.0", "9.0", "8.0", "7.0"):
        p = local / "Programs" / "KiCad" / ver / "bin" / "ngspice.dll"
        if p.is_file():
            return p

    try:
        import PySpice  # noqa: F401
        base = Path(__import__("PySpice").__file__).parent / "Spice" / "NgSpice"
        for cand in base.rglob("ngspice*.dll"):
            return cand
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 互斥
#
# ★ 这把锁不是"保险起见加的"，是**实测出来的必需项**。
#
# PySpice 的 NgSpice 绑定有**两处进程级全局状态**，都不受我们控制：
#
# 1. ``PySpice/Spice/NgSpice/Shared.py:110`` 的 ``ffi = FFI()`` 是**模块级单例**；
#    ``_load_library()`` 每次构造实例都无脑 ``ffi.cdef(api.h 全文)``。
#    cffi 不许同一个 FFI 重复声明同一个 struct —— 于是**第二次 cdef 必然抛**
#    ``CDefError: duplicate declaration of struct ngcomplex``。
# 2. ``NgSpiceShared._instances`` 按 ``ngspice_id`` 缓存实例，但"查缓存 → 构造 →
#    写回缓存"这三步**不是原子的**。
#
# 两者叠加的结果：两个线程**同时**第一次构造时必崩一个。
# 实测（全新进程，8 线程同时首调 ``probe_availability``）：**1 成功 / 7 失败**，
# 失败者全是同一条 CDefError。而它是**偶发**的 —— 谁先谁后看调度，
# 于是表现为"同一道题，偶尔报求解失败，重试又好了"这种最难查的形态。
#
# 谁会并发进来？FastAPI 的同步端点跑在**线程池**里：
#   ``/api/health``（探针 + ``/api/solve`` 各自会调它）与 ``/api/solve``
#   都可能同时在飞。启动时尤其明显 —— 启动器自己的就绪轮询与外部检查
#   会**同时**打 ``/api/health``，两个请求都看见空缓存、都去构造 → 崩一个。
#   （打包成 exe 后第一次暴露出来，就是这条：探针报"不可用"，而真解却过了。）
#
# 所以：**ngspice 的共享库这一整块必须串行**。探针与真解共用同一把锁 ——
# 它们抢的是同一份全局状态，分成两把锁等于没加。
# 这个程序的形态是"本机单人算题"，串行化的代价可以忽略；
# 而换来的是"结果确定"。
_NG_LOCK = threading.RLock()


# ---------------------------------------------------------------- 探针


def _dll_origin(dll: Path) -> str:
    """这个 dll 是哪来的 —— 排查"为什么它用的是旧版 ngspice"时全靠它。"""
    if os.environ.get(DLL_ENV) and Path(os.environ[DLL_ENV]) == dll:
        return f"环境变量 {DLL_ENV} 指定"
    ours = bundled_ngspice()
    if ours is not None and ours == dll:
        return "随程序自带（推荐：版本是我们测过的那个）"
    for p in _CANDIDATES:
        if Path(p) == dll:
            return "系统安装的 KiCad"
    if "KiCad" in str(dll):
        return "系统安装的 KiCad"
    return "PySpice 自带"


def _probe_with_dll(dll: Path) -> tuple[bool, str]:
    """真跑一次最小仿真，确认这一路的 DLL **端到端**能用。

    ★ 为什么必须真跑，而不是「import 一下就算可用」：
      这是"不许静默降级"这条纪律在**探针本身**上的落点。
      旧实现只做 ``import PySpice`` + 设 ``LIBRARY_PATH`` 就宣布可用。
      打包成 exe 之后发生的事是：启动横幅写着「● ngspice（第三方独立实现）」
      可用、报告页也显示"ngspice 可用"，而每一道题的对账表里都留着一行
      「ngspice 求解失败：找不到 api.h」—— PySpice 运行时要读的一个**数据文件**
      没被打进包。于是用户看到一个"三法齐全"的界面，第三法其实是空的，
      而且**没有任何地方告诉他**。
      判据改成"跑完一次 .op 并且拿到的数是对的"之后，这种包在启动横幅上
      就会直接显示成「不可用 + 原因」，而不是等用户算完题自己发现。
      （一次最小 .op 只要 ~0.02 s，随便调用。）

    探测电路：1 V 源 + 1 Ω 电阻并联 → 节点电压必须是 1 V、源支路电流 ±1 A。
    两个数都要对，才算这一路真的通。
    """
    try:
        from PySpice.Spice.Netlist import Circuit as PCircuit
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.LIBRARY_PATH = str(dll)

        pc = PCircuit("probe")
        pc.V("probe_v", "1", "0", 1.0)
        pc.R("probe_r", "1", "0", 1.0)
        an = pc.simulator(simulator="ngspice-shared").operating_point()

        v = float(an["1"][0])
        if abs(v - 1.0) > 1e-6:
            return False, f"探测电路解出 V(1) = {v!r}，应当是 1.0"

        br = an.branches
        key = next((k for k in br.keys() if "probe_v" in k), None)
        if key is None:
            return False, f"拿不到电压源的支路电流（branches = {list(br.keys())}）"
        i = float(br[key][0])
        if abs(abs(i) - 1.0) > 1e-6:
            return False, f"探测电路解出 i = {i!r}，应当是 ±1.0"
        return True, ""
    except Exception as e:                           # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


#: 探测结果缓存。可用性是**环境属性**，进程活着的期间不会变；
#: 而 /api/health 与启动横幅都会调它，没必要每次都起一次仿真。
_PROBE_CACHE: dict[str, Any] | None = None


def probe_availability() -> dict[str, Any]:
    """报告第三方法当前是否可用，供 WebUI 与报告显示。

    **不允许静默降级**：不可用时要把原因、探测过的路径都写出来，
    报告里"三法对账表"必须显式显示这一路缺失，而不是少一行让人以为通过了。

    ★ 整个过程持 ``_NG_LOCK``（见上文"互斥"一节）。缓存判断做**双重检查**：
    先无锁快路径，抢到锁之后再确认一次 —— 否则"两个请求同时看见空缓存、
    都去构造"就会撞上 cffi 的重复声明。这个函数被 FastAPI 的线程池调用，
    **并发是常态，不是意外**。
    """
    global _PROBE_CACHE
    if _PROBE_CACHE is not None:                 # 快路径：探过就直接给
        return dict(_PROBE_CACHE)

    with _NG_LOCK:
        if _PROBE_CACHE is not None:             # 等锁期间别人可能已经探完了
            return dict(_PROBE_CACHE)

        dll = find_ngspice_dll()
        info: dict[str, Any] = {"available": False,
                                "dll": str(dll) if dll else None}
        if dll is None:
            info["reason"] = "未找到 ngspice.dll"
            info["hint"] = (
                f"安装 KiCad（自带 ngspice）后重试，或设置环境变量 "
                f"{DLL_ENV} 指向 ngspice.dll"
            )
        else:
            # ★ 来源也要报：同一台机器上可能同时存在"包内自带"与"KiCad 装的"
            #   两份，版本还不一样。不说清用的是哪一份，就没人能解释
            #   "为什么换台机器结果微调了"。
            info["origin"] = _dll_origin(dll)

            ok, why = _probe_with_dll(dll)
            if ok:
                info["available"] = True
                info["pyspice"] = True
                # 把"怎么判的"也写出来：这个字段存在的意义就是让人知道
                # 我们没偷懒成"import 成功就算可用"。
                info["verified"] = "已实跑一次最小 .op（1V/1Ω）并核对结果"
            else:
                info["reason"] = f"找到了 ngspice.dll，但它跑不起来：{why}"
                info["hint"] = (
                    "若是打包版，多半是 PySpice 运行时要读的数据文件没被打进包"
                    "（api.h / logging.yml，配包里用 collect_data_files('PySpice')）；"
                    "若是源码版，检查 ngspice.dll 与 PySpice 的版本是否匹配。"
                )

        _PROBE_CACHE = dict(info)
        return info


# ---------------------------------------------------------------- 主解


def ngspice_method(circuit: Circuit) -> Solution:
    """用 ngspice 求直流工作点。

    ★ 整段持 ``_NG_LOCK``。看起来"只是搭个网表、跑个 .op，没什么共享状态"，
      但底层那份 ``NgSpiceShared`` 单例**是有状态的**：``LIBRARY_PATH`` 是类属性、
      实例按 id 缓存、``.op`` 命令与结果都存在同一个实例上。
      两个请求同时进来 → 一边在改 ``LIBRARY_PATH``、一边在读上一题的结果，
      再加上 cffi 那个重复声明，就成了"偶发求解失败 / 读数串题"。
      串行化是这里唯一诚实的做法。
    """
    circuit.validate()
    with _NG_LOCK:
        return _solve_locked(circuit)


def _solve_locked(circuit: Circuit) -> Solution:
    """``ngspice_method`` 的实体。**只在持锁时调用**。"""
    dll = find_ngspice_dll()
    if dll is None:
        raise CircuitError(
            "ngspice 第三方法不可用：未找到 ngspice.dll。"
            f"装 KiCad 或设 {DLL_ENV} 环境变量。"
        )

    try:
        from PySpice.Spice.Netlist import Circuit as PCircuit
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    except ImportError as e:                     # pragma: no cover
        raise CircuitError(f"ngspice 第三方法不可用：PySpice 未安装（{e}）") from None

    NgSpiceShared.LIBRARY_PATH = str(dll)

    warnings: list[str] = []
    pc = PCircuit(circuit.name or "circuit")

    # ---- 节点名映射：ngspice 的地只能是 0，其余保持原样
    node_map: dict[str, str] = {circuit.ref_node: "0"}
    for n in circuit.hot_nodes:
        if n in ("0", "gnd", "GND", "ground"):
            raise CircuitError(
                f"节点名 {n!r} 与 ngspice 的地节点冲突，请改名后重试"
            )
        node_map[n] = n

    def nm(x: str) -> str:
        return "0" if x == circuit.ref_node else x

    def dev(c) -> str:
        """PySpice 会自动在名字前补器件字母。位号若已带该字母要先剥掉，
        否则 R1 会变成 RR1、V2 变成 VV2 —— 网表能跑但名字全错，
        对账时按位号取电流就会取空。"""
        body = c.ref
        if body[:1].upper() == c.kind:
            body = body[1:]
        return body or c.ref

    for c in circuit.components:
        a, b = nm(c.nodes[0]), nm(c.nodes[1])
        v = float(c.value)
        if c.kind == "R":
            if v <= 0:
                raise CircuitError(f"{c.ref}: ngspice 不接受非正电阻 {v}")
            pc.R(dev(c), a, b, v)
        elif c.kind == "V":
            pc.V(dev(c), a, b, v)
        elif c.kind == "I":
            pc.I(dev(c), a, b, v)
        else:
            raise CircuitError(f"{c.ref}: ngspice 路径暂不支持 {c.kind}")

    try:
        sim = pc.simulator(simulator="ngspice-shared")
        op = sim.operating_point()
    except Exception as e:
        raise CircuitError(
            f"ngspice 求解失败：{type(e).__name__}: {e}。"
            "常见原因：网表里有 ngspice 不接受的写法、或电路本身无直流工作点。"
        ) from None

    def read(name: str) -> float | None:
        """从 .op 结果里取一个标量。

        ★ PySpice 的坑：.op 结果里节点值是 **1 元素数组**，必须
        ``float(x.as_ndarray()[0])``，直接 float() 会拿到数组。
        """
        try:
            item = op[name]
        except Exception:
            return None
        arr = item.as_ndarray() if hasattr(item, "as_ndarray") else item
        try:
            return float(arr[0]) if hasattr(arr, "__len__") else float(arr)
        except Exception:
            return None

    # ---- 节点电压。参考点（内部叫 "0"）不在节点表里，直接取 0 V
    node_v: dict[str, float] = {circuit.ref_node: 0.0}
    for n in circuit.hot_nodes:
        val = read(n)
        if val is None:
            warnings.append(
                f"ngspice 结果里没有节点 {n}，该节点可能被简化掉了（例如只串在电流源上）"
            )
            val = float("nan")
        node_v[n] = val

    # ---- 支路电流：ngspice 的 .op 只给节点电压和电压源电流。电阻电流由
    # node voltage 反算 —— 节点电压仍来自 ngspice 的独立求解器，所以代码路径没变味。
    currents: dict[str, float] = {}
    drops: dict[str, float] = {}
    for c in circuit.components:
        f, t = declared_direction(c)
        d = node_v.get(f, float("nan")) - node_v.get(t, float("nan"))
        drops[c.ref] = d
        if c.kind == "V":
            # 实测：PySpice 把电压源支路电流挂在 op['v1'] 这种**器件名小写**键上
            # （`op.branches` 里就是它），不是 'i(v1)'。两种写法都试一遍更稳。
            iv = read(f"i({c.ref})")
            if iv is None:
                iv = read(c.ref.lower())
            if iv is None:
                # 电压源电流也能由该源 + 端所在节点的 KCL 反推，但那样就不独立了，
                # 这里宁可留 NaN 让对账表显式暴露缺失
                warnings.append(f"ngspice 未返回 {c.ref} 的支路电流，对账表该项将缺失")
                currents[c.ref] = float("nan")
            else:
                # SPICE 的 i(Vx) 定义是"流入 + 端的电流"（+ -> − 穿过源内部），
                # 而 IR 的电压源参考方向是"内部 − -> +"，两者符号相反。
                currents[c.ref] = -float(iv)
        elif c.kind == "R":
            currents[c.ref] = d / float(c.value)
        else:
            currents[c.ref] = float(c.value)

    return Solution(
        method="ngspice(第三方共享库浮点求解)",
        node_voltages=node_v,
        branch_currents=currents,
        branch_drops=drops,
        exact=False,
        detail={
            "dll": str(dll),
            "note": "PySpice 可能提示 'Unsupported Ngspice version 46'，"
                    "那只是版本枚举认不出，不影响 .op 取值",
        },
        warnings=warnings,
    )
