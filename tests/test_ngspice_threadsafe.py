"""ngspice 并发安全自检。

★★ 为什么必须有这个文件 —— 这是一次**偶发**故障，只靠"跑一遍看看"抓不到：

PySpice 的 NgSpice 绑定有两处**进程级全局状态**（都不在我们手里）：

1. ``PySpice/Spice/NgSpice/Shared.py:110`` 的 ``ffi = FFI()`` 是模块级单例，
   而 ``_load_library()`` 每次构造实例都无脑 ``ffi.cdef(api.h 全文)``。
   cffi **不允许同一个 FFI 重复声明同一个 struct** → 第二次 cdef 必抛
   ``CDefError: duplicate declaration of struct ngcomplex``。
2. ``NgSpiceShared._instances`` 按 id 缓存实例，但"查缓存 → 构造 → 写回"
   **不是原子的**。

叠加结果：**两个线程同时第一次构造，必崩一个**。而 FastAPI 的同步端点跑在
线程池里 —— 启动那一刻启动器自己的就绪轮询与外部检查会同时打 ``/api/health``，
``/api/solve`` 之间也会并发。于是表现为"同一道题偶尔报求解失败、重试又好了"，
这种形态**单线程怎么跑都复现不出来**。

实测（本文件第 ① 节，未加锁时）：8 线程 → **1 成功 / 7 失败**。
修法是 ``app/solver/ngspice.py`` 里的 ``_NG_LOCK``：探针与真解**共用同一把锁**
（它们抢的是同一份全局状态，分成两把等于没加）。

★ **必须单独起进程跑**：PySpice 的 ``ffi`` 一旦被 cdef 过就再也复现不出竞态，
  所以这个文件不能和其它 ngspice 用例合并到同一个进程里。
  第 ⓪ 节会先确认"本进程还没碰过 ngspice"，否则如实报"跳过"而不是假装通过。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_ngspice_threadsafe.py
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ir.model import Circuit, Component, Evidence                    # noqa: E402
from app.solver import ngspice as NG                                     # noqa: E402

FAILS = 0
SKIPPED = False


def check(name: str, ok: bool, detail: object = "") -> None:
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'通过' if ok else '失败'}] {name}"
          + (f"  —— {detail}" if detail != "" else ""))


def skip(name: str, why: str) -> None:
    global SKIPPED
    SKIPPED = True
    print(f"  [跳过] {name}  —— {why}")


def C(ref, kind, a, b, value):
    return Component(ref=ref, kind=kind, nodes=(a, b), value=value,
                     evidence=Evidence(source="manual", confidence=1.0))


def fan_out(n: int, fn) -> list:
    """n 个线程用 barrier 对齐后同时执行 fn(i)，收集结果（含异常）。"""
    bar = threading.Barrier(n)
    out: list = [None] * n

    def work(i: int) -> None:
        try:
            bar.wait()
            out[i] = fn(i)
        except BaseException as e:                       # noqa: BLE001
            out[i] = e

    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def main() -> int:
    print("=" * 72)
    print("  ngspice 并发安全自检（必须单独进程跑）")
    print("=" * 72)

    # ------------------------------------------------------------ ⓪ 前提
    print("\n" + "-" * 72)
    print("  ⓪ 前提：本进程尚未触碰过 ngspice")
    print("-" * 72)
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    primed = bool(getattr(NgSpiceShared, "_instances", {}))
    cache_free = NG._PROBE_CACHE is None
    if primed or not cache_free:
        skip("并发竞态复现",
             f"本进程已经构造过 ngspice 实例（_instances={list(getattr(NgSpiceShared, '_instances', {}))}）"
             f"或探针缓存已填（{NG._PROBE_CACHE is not None}）——"
             "cffi 的 api.h 已被 cdef 过，竞态无法复现。请单独起进程跑本文件。")
        print("\n" + "=" * 72)
        print("  总计失败项：0（有跳过项，见上）")
        print("=" * 72)
        return 0
    check("本进程是干净的（能真实复现竞态）", True)

    # ------------------------------------------------------------ ① 并发探针
    print("\n" + "-" * 72)
    print("  ① 8 个线程同时首调 probe_availability（未加锁时这里必崩 7 个）")
    print("-" * 72)
    N = 8
    res = fan_out(N, lambda i: NG.probe_availability())
    errs = [r for r in res if isinstance(r, BaseException)]
    check("没有线程抛异常", not errs,
          "".join(traceback.format_exception_only(type(errs[0]), errs[0]))
          if errs else "")
    dicts = [r for r in res if isinstance(r, dict)]
    check(f"每个线程都拿到了可用结论（{N}/{N}）",
          len(dicts) == N and all(d.get("available") for d in dicts),
          [d.get("reason") for d in dicts if not d.get("available")][:2])
    bad = [d for d in dicts if "CDefError" in str(d.get("reason", ""))]
    check("★ 一个都没撞上「duplicate declaration of struct ngcomplex」",
          not bad, bad[:1])
    check("结论一致（不会有人看到可用、有人看到不可用）",
          len({bool(d.get("available")) for d in dicts}) == 1,
          {bool(d.get("available")) for d in dicts})

    # ------------------------------------------------------------ ② 并发求解
    print("\n" + "-" * 72)
    print("  ② 8 个线程同时解同一道题（手算 v2 = 7.5V、i = 2.5mA）")
    print("-" * 72)

    def build() -> Circuit:
        return Circuit(name="ts", components=[
            C("V1", "V", "1", "0", 10.0),
            C("R1", "R", "1", "2", 1000.0),
            C("R2", "R", "2", "0", 3000.0),
        ])

    sols = fan_out(N, lambda i: NG.ngspice_method(build()))
    errs2 = [s for s in sols if isinstance(s, BaseException)]
    check("没有线程抛异常", not errs2,
          "".join(traceback.format_exception_only(type(errs2[0]), errs2[0]))
          if errs2 else "")
    ok2 = [s for s in sols if not isinstance(s, BaseException)]
    check(f"每个线程都解出了结果（{len(ok2)}/{N}）", len(ok2) == N)
    # ★ 逐项比对：并发如果串了状态，读数会互相污染
    vs = {round(s.node_voltages.get("2", float("nan")), 9) for s in ok2}
    check("★ 所有线程的节点电压完全一致（没有互相串读数）",
          vs == {7.5}, vs)
    i1 = {round(abs(s.branch_currents.get("R1", float("nan"))), 12) for s in ok2}
    check("★ 所有线程的支路电流完全一致（2.5mA = 1/400）",
          i1 == {0.0025}, i1)

    # ------------------------------------------------------------ ③ 实例复用
    print("\n" + "-" * 72)
    print("  ③ 共享库实例没有被反复构造（cdef 只该发生一次）")
    print("-" * 72)
    ids = sorted(getattr(NgSpiceShared, "_instances", {}))
    check("NgSpiceShared 只缓存了 0 号实例（说明 cdef 只跑了一次）",
          ids == [0], ids)

    print("\n" + "=" * 72)
    print(f"  总计失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
