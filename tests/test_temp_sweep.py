"""残留解压目录清理的自检。

单文件 exe 每次启动都把 ~250 MB 解压到 ``%TEMP%\\_MEI*``，正常退出由引导器删掉；
**只要上次是被强杀的**（任务管理器、强关控制台、机器崩），目录就留下没人管。
实测在本机被测了一下午就攒了 **19 个 / 3.0 GB**。

清理代码是在"删别人的东西"，所以每一道闸门都必须有测试盯着：

| 闸门 | 判据 | 对应小节 |
|---|---|---|
| 只是 ``_MEI*`` 且顶层 | 别的文件/目录一律不碰 | ①③ |
| 时效门槛 | 太新的不动（可能是刚启动的实例正在解压） | ④ |
| **改名试锁** | Windows 上**在用**的目录改不了名；改名成功才敢删 | ② |
| 自己那个 | ``sys._MEIPASS`` 永远跳过 | ③ |

★ 闸门 3 的实测依据（本文件探测出来的，不是猜的）：
  目录里**有打开的文件句柄**时 ``rename`` 报「拒绝访问」；句柄一关就能改名。
  所以"改名成功"等价于"没有任何进程还在用它"。
  这条必须真测 —— 它是"删错东西"与"清干净"之间的唯一分界线。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_temp_sweep.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import launcher as L                                                  # noqa: E402

FAILS = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'通过' if ok else '失败'}] {name}"
          + (f"  —— {detail}" if detail != "" else ""))


def make(base: Path, name: str, nbytes: int = 1000) -> Path:
    """造一个像解压目录的目录（名字不重要，大小要能核对）。"""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dummy.dat").write_bytes(b"x" * nbytes)
    return d


def main() -> int:
    print("=" * 72)
    print("  残留解压目录清理自检（全程在临时沙箱里，不碰真实 %TEMP%）")
    print("=" * 72)

    sandbox = Path(tempfile.mkdtemp(prefix="ca_sweep_test_"))
    print(f"\n  沙箱：{sandbox}")
    try:
        # ---------------------------------------------------------- ① 基本
        print("\n" + "-" * 72)
        print("  ① 够旧的残留目录会被删掉，大小如实统计")
        print("-" * 72)
        old = make(sandbox, "_MEI_old", 4096)
        st = L.sweep_stale_mei(base=sandbox, age=0)
        check("被删掉了", not old.exists(), old)
        check("统计里记了 1 个", st["removed"] == 1, st["removed"])
        check("回收字节数对得上（4096）", st["freed"] == 4096, st["freed"])

        # ---------------------------------------------------------- ② 在用
        print("\n" + "-" * 72)
        print("  ② ★ 正被使用的目录**不能**删（持有文件句柄 = 改名会失败）")
        print("-" * 72)
        busy = make(sandbox, "_MEI_busy", 8192)
        fh = open(busy / "dummy.dat", "r+b")          # 制造"正在使用"
        try:
            st2 = L.sweep_stale_mei(base=sandbox, age=0)
            check("★ 目录还在（没被误删）", busy.exists(), busy)
            check("★ 被记成 busy（而不是 failed / 装作没看见）",
                  st2["busy"] == 1, st2["busy"])
            check("没被算进回收数", st2["removed"] == 0, st2["removed"])
        finally:
            fh.close()
        st3 = L.sweep_stale_mei(base=sandbox, age=0)   # 句柄一关就能删
        check("句柄关掉之后就能删了（对照，证明上一步确实是句柄挡住的）",
              not busy.exists() and st3["removed"] == 1, st3["removed"])

        # ---------------------------------------------------------- ③ 边界
        print("\n" + "-" * 72)
        print("  ③ 边界：自己那个不能删；不叫 _MEI* 的不能碰")
        print("-" * 72)
        own = make(sandbox, "_MEI_myself", 512)
        other = make(sandbox, "some_other_app", 512)
        st4 = L.sweep_stale_mei(base=sandbox, age=0, keep=own)
        check("★ 自己正在用的那个（_MEIPASS）原封不动", own.exists(), own)
        check("★ 不叫 _MEI* 的目录没被碰", other.exists(), other)
        check("清理数只算了该算的那个", st4["removed"] == 0, st4["removed"])

        # ---------------------------------------------------------- ④ 时效
        print("\n" + "-" * 72)
        print("  ④ 时效门槛：太新的不动（可能是刚启动的实例正在往里面解压）")
        print("-" * 72)
        fresh = make(sandbox, "_MEI_fresh", 512)
        st5 = L.sweep_stale_mei(base=sandbox, age=10 ** 9)
        check("★ 门槛很大时什么都不删", fresh.exists() and st5["removed"] == 0,
              st5)
        check("默认门槛是 2 小时（够解压几秒用的，又不会把昨天的留着）",
              L.STALE_MEI_AGE == 2 * 3600, L.STALE_MEI_AGE)

        # ---------------------------------------------------------- ⑤ 开关
        print("\n" + "-" * 72)
        print("  ⑤ 关得掉（排查问题时不想让它动任何东西）")
        print("-" * 72)
        # 先清掉前几节留下的 _MEI*，让这一节的计数是**确定**的。
        # （第一版没清，`removed` 撞上前面剩下的 3 个，断言就假失败了 ——
        #   这类"测试自己没收拾干净"的错误最容易看错方向。）
        for p in [q for q in sandbox.glob("_MEI*") if q.is_dir()]:
            shutil.rmtree(p, ignore_errors=True)
        keepme = make(sandbox, "_MEI_keepme", 512)
        os.environ[L.NO_SWEEP_ENV] = "1"
        try:
            st6 = L.sweep_stale_mei(base=sandbox, age=0)
            check("设了开关就一个都不删", keepme.exists() and st6["removed"] == 0,
                  st6.get("skipped", ""))
        finally:
            os.environ.pop(L.NO_SWEEP_ENV, None)
        st7 = L.sweep_stale_mei(base=sandbox, age=0)
        check("开关摘掉后恢复工作", not keepme.exists() and st7["removed"] == 1,
              st7["removed"])

        # ---------------------------------------------------------- ⑥ 不留怪名
        print("\n" + "-" * 72)
        print("  ⑥ 不留怪名字（删不掉要改回原名，不能留一堆 .stale）")
        print("-" * 72)
        left = [p.name for p in sandbox.iterdir() if p.name.endswith(".stale")]
        check("★ 没有 .stale 残留", not left, left)
        check("沙箱里剩下的都是「不该被删」的那几个",
              sorted(p.name for p in sandbox.iterdir())
              == ["some_other_app"],                      # 只有它该留下
              sorted(p.name for p in sandbox.iterdir()))

    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        print(f"\n  已清理沙箱 {sandbox}")

    print("\n" + "=" * 72)
    print(f"  总计失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
