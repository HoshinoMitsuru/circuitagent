"""在**干净目录**里跑一遍打好的 exe，确认它自给自足。

用法::

    python tools/verify_exe.py            # 默认验 dist/circuit_agent.exe
    python tools/verify_exe.py --keep     # 保留临时目录，便于进去翻看

★ 为什么必须换个空目录、单独验一遍：

1. 单文件模式的**资源根**是临时解压目录（``_MEIPASS``），**数据根**是 exe 同级。
   在项目目录里跑根本测不出这件事 —— 开发模式下两者恰好是同一个目录，
   于是"runs/ 落到临时目录、退出即失"这种事故会被完全掩盖。
2. **缺 ngspice 不会报任何错**，只会让三法悄悄变成两法。所以这里必须主动断言
   「ngspice 可用」**且**「用的是包内自带那一份」（而不是本机 KiCad 那个）——
   否则把 exe 发到没装 KiCad 的机器上才发现，就已经晚了。
3. 同理必须断言 OCR 可用：``winrt`` 是命名空间包 + 原生扩展，
   少收一个子模块的表现是"OCR 不可用"这个**看起来完全正常**的降级分支。
"""

from __future__ import annotations

import argparse
import json
import locale
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "dist" / "circuit_agent.exe"

FAILS = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'通过' if ok else '失败'}] {name}"
          + (f"  —— {detail}" if detail != "" else ""))


def banner(t: str) -> None:
    print("\n" + "#" * 72)
    print("# " + t)
    print("#" * 72)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_alive(port: int) -> bool:
    """端口上还有人在服务吗。用它判断"服务真的没了"，比信进程对象可靠。"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.8):
            return True
    except OSError:
        return False


def stop_tree(proc: subprocess.Popen) -> None:
    """结束 exe **及其全部子进程**。

    ★★ 只 ``terminate()`` 是不够的，这是实测踩出来的：

      PyInstaller 的 onefile 形态是「**父进程**把包解压到临时目录 → 再
      ``CreateProcess`` 一个**子进程**去跑真正的应用，父进程只是等着」。
      于是 ``subprocess.Popen`` 拿到的是**父**进程 ——
      杀掉它，那个正在跑服务、占着端口与 ``%TEMP%\\_MEI*`` 的子进程就变成
      **孤儿**继续活着。我第一次跑就漏下来两个，后果是：
      下次启动报"端口被占"、临时目录删不掉（于是又悄悄攒了 8 个）。

      所以必须先 ``taskkill /T`` **在父进程还活着的时候**连树一起收
      （父进程一死就找不到子进程了）。``/T`` = 连同子进程树。
    """
    if proc.poll() is None and os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        except Exception:                                  # noqa: BLE001
            pass
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:                                      # noqa: BLE001
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:                                  # noqa: BLE001
            pass


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


#: 横幅的首行。用它当"日志解码对不对"的判据 ——
#: 只要解出来能看见这一行，就说明编码猜对了。
BANNER_MARK = "电路读图"


def read_log(p: Path) -> tuple[str, str]:
    """读启动日志，返回 ``(文本, 采用的编码)``。

    ★ 打包后的 exe 在**管道/文件**里输出时用的是系统 ANSI 代码页
      （中文 Windows 上是 cp936），不是 UTF-8 —— 那是**故意的**：
      它是给 Windows 控制台/记事本看的，硬写 UTF-8 反而会让这两者满屏乱码。
      （写到真实控制台时 Python 走 WriteConsoleW，不受此影响，所以**双击看到的是正常的**。）

    ★★ 不能只靠"decode 抛不抛异常"来选编码：不同的多字节编码之间**存在重叠**，
      某些 cp936 字节序列恰好也是合法 UTF-8，先试的那个会**静默解成乱码**。
      所以这里加一道**判据**：解出来必须能看见横幅首行；
      能看见才算解对，看不到就继续换下一个编码。
      换完都不行就如实标出来（由调用方断言），而不是把乱码当正常输出打印出去。
    """
    raw = p.read_bytes()
    seen: list[str] = []
    fallback: list[tuple[str, str]] = []
    for enc in (locale.getpreferredencoding(False), "cp936", "gbk",
                "utf-8", "cp1252", "latin-1"):
        if enc in seen:
            continue
        seen.append(enc)
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if BANNER_MARK in text:
            return text, enc
        fallback.append((enc, text))
    if fallback:
        return fallback[0][1], fallback[0][0]
    return raw.decode("utf-8", "replace"), "utf-8(replace)"


def post(url: str, payload: dict, timeout: float = 180.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ★ 格式是**标准 SPICE**：``R1 1 2 1000``（名字在前）。
#   注意别写成 ``R R1 1 2 1000``（"字母 + 位号"两段式）——
#   本解析器会把首 token 当位号，于是三个元件位号都成了 "R"，
#   报「位号重复：['R']」。我第一次就是这么写错的。
# 纯 R/V：把三法互校整条路径走通
NET_RV = """\
V1 1 0 10
R1 1 2 1000
R2 2 0 3000
"""
# 含 C/L：直流稳态要先化简（C 开路、L 短路）才能解 —— 那是本项目历史上
# 出过最严重缺陷的地方（化简层静默降级），打包后必须确认这条路仍然通。
NET_CL = """\
V1 1 0 10
R1 1 2 1000
R2 2 0 1000
C1 2 0 0.000001
L1 2 3 0.001
R3 3 0 1000
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(EXE))
    ap.add_argument("--keep", action="store_true",
                    help="保留临时目录并在结尾打印它的路径")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="等待服务就绪的秒数上限")
    a = ap.parse_args()

    exe = Path(a.exe)
    print("=" * 72)
    print("  在一个空目录里验打好的 exe")
    print("=" * 72)
    if not exe.is_file():
        print(f"[错误] 找不到 {exe}。先跑 python tools/build_exe.py。")
        return 2
    print(f"  exe      {exe}")
    print(f"  大小     {exe.stat().st_size / 1024 / 1024:.1f} MB")

    work = Path(tempfile.mkdtemp(prefix="ca_verify_"))
    target = work / exe.name
    shutil.copy2(exe, target)
    port = free_port()
    log = work / "launch.log"

    print(f"  临时目录 {work}")
    print(f"  端口     {port}")

    env = dict(os.environ)
    env["CIRCUIT_AGENT_PORT"] = str(port)
    env["CIRCUIT_AGENT_NO_BROWSER"] = "1"
    # ★ 故意清掉这一条：要证明用的是**包内自带**的 ngspice，
    #   而不是环境变量硬指过来的本机那一份。
    env.pop("CIRCUIT_AGENT_NGSPICE_DLL", None)

    banner("启动（冷启动包含一次解压，慢是正常的）")
    t0 = time.time()
    with log.open("wb") as fh:
        proc = subprocess.Popen([str(target)], cwd=str(work), env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=fh, stderr=subprocess.STDOUT,
                                creationflags=getattr(
                                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    live = None
    try:
        deadline = time.time() + a.timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                live = get(f"http://127.0.0.1:{port}/api/health", timeout=2)
                break
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(0.4)
        boot = time.time() - t0
        check("服务起来了（健康检查 200）", live is not None,
              f"冷启动 {boot:.1f} 秒" if live else
              f"{a.timeout:.0f} 秒内没起来，见 {log}")

        if live is None:
            print("\n---- 启动日志 ----")
            print(read_log(log)[0][-3000:])
            return 1

        # ---------------------------------------------------------- 路径
        banner("① 资源根 / 数据根：数据必须落在 exe 同级，不能落进临时解压目录")
        paths = live.get("paths") or {}
        check("运行形态如实标为 packaged(exe)",
              paths.get("mode") == "packaged(exe)", paths.get("mode"))
        check("★ 数据根 = exe 所在目录（便携模式）",
              Path(paths.get("data_root", "")).resolve()
              == work.resolve(), paths.get("data_root"))
        check("★ 数据根**不是**临时解压目录（那里面退出即失）",
              "_MEI" not in str(paths.get("data_root", "")),
              paths.get("data_root"))
        rr = Path(paths.get("resource_root", ""))
        check("资源根取自包内解压目录（web/ 就在那儿）",
              "_MEI" in str(rr) or rr.name.lower().startswith("_"),
              str(rr))
        check("数据目录给出「为什么落在这儿」的说明",
              bool(paths.get("data_root_reason")), paths.get("data_root_reason"))

        # ---------------------------------------------------------- 三法
        banner("② 三法：第三法必须可用，且用的是包内自带那一份")
        ng = live.get("ngspice") or {}
        check("★ ngspice 可用（否则三法悄悄变成两法）",
              ng.get("available") is True,
              ng.get("reason") or ng.get("dll"))
        check("★ 用的是**随包自带**的 ngspice，而不是本机 KiCad 那个",
              "自带" in str(ng.get("origin", "")), ng.get("origin"))
        dl = str(ng.get("dll", ""))
        check("★ 探针是真跑了一次最小 .op 才判定的（不是只看 import）",
              bool(ng.get("verified")), ng.get("verified"))
        check("该 dll 就在包内解压目录里（说明 datas 打进去了）",
              "_MEI" in dl or Path(dl).parent.name.lower().startswith("_"), dl)

        # ---------------------------------------------------------- 视觉层
        banner("③ 视觉层：winrt 原生扩展有没有被收全（少一个只会静默降级）")
        vis = live.get("vision") or {}
        loc = vis.get("local") or {}
        ocr = loc.get("ocr") or {}
        wires = loc.get("wires") or {}
        check("位图通道已开启", vis.get("enabled") is True, vis.get("mode"))
        check("★ 系统 OCR 可用（不可用多半是 winrt 子模块没收全）",
              ocr.get("available") is True,
              ocr.get("reason") or ocr.get("hint") or "ok")
        if ocr.get("available"):
            check("  └ 识别语言与源码环境一致",
                  bool(ocr.get("engine_language")), ocr.get("engine_language"))
        check("本地几何层就绪（opencv / scipy 都在）",
              wires.get("ready") is True,
              f"opencv {wires.get('opencv')} / scipy {wires.get('scipy')}")

        # ---------------------------------------------------------- 算题
        banner("④ 真算一道题：三法互校 + 功率守恒（纯 R/V）")
        b = post(f"http://127.0.0.1:{port}/api/from-spice",
                 {"text": NET_RV, "name": "验证用 R/V"})
        check("from-spice ok", b.get("ok") is True, str(b.get("error"))[:80])
        sid = (b.get("session") or {}).get("sid")
        check("拿到会话 id", bool(sid))
        if sid:
            sl = post(f"http://127.0.0.1:{port}/api/solve", {"sid": sid})
            check("solve ok", sl.get("ok") is True, str(sl.get("error"))[:90])
            pack = sl.get("pack") or {}
            check("★ 三法互校通过（overall_pass）",
                  pack.get("overall_pass") is True, pack.get("failures"))
            bt = {e["ref"]: e for e in (pack.get("branch_table") or [])}
            # 手算：10V 经 1k 串 3k → i = 2.5mA，R2 上 7.5V
            check("支路电流是手算的 2.5mA（三法各自都给这个值）",
                  all(str(bt.get(k, {}).get("exact", "")).startswith("1/400")
                      for k in ("V1", "R1", "R2")),
                  {k: v.get("exact") for k, v in bt.items()})
            # ★ 显式确认三路都在，而不是"少一行也看不出"
            check("★ 对账表里 ngspice 那一路有值（不是空的）",
                  all(str(bt.get(k, {}).get("ngspice") or "").strip()
                      for k in ("V1", "R1", "R2")),
                  {k: bt.get(k, {}).get("ngspice") for k in ("V1", "R1", "R2")})

        banner("⑤ 含 C/L 的电路：化简层（C→开路、L→短路）打包后仍然通")
        b2 = post(f"http://127.0.0.1:{port}/api/from-spice",
                  {"text": NET_CL, "name": "验证用 含C/L"})
        check("from-spice ok（含 C/L 不许 500）", b2.get("ok") is True,
              str(b2.get("error"))[:80])
        sid2 = (b2.get("session") or {}).get("sid")
        if sid2:
            s2 = post(f"http://127.0.0.1:{port}/api/solve", {"sid": sid2})
            check("solve 状态码是 200 那一类（不是 5xx）",
                  s2.get("ok") is not None, str(s2.get("error"))[:90])
            p2 = s2.get("pack") or {}
            check("★ 含 C/L 也通过三法互校",
                  p2.get("overall_pass") is True, p2.get("failures"))
            # 直流：C 开路、L 短路 → R2(1k) 与 R3(1k) 并联再串 R1(1k)
            #   → 总 1.5k，i = 10/1.5k = 20/3 mA，节点 2 = 10 × 500/1500 = 10/3 V
            nt = {e["node"]: e for e in (p2.get("node_table") or [])}
            two = next((v for k, v in nt.items() if abs((v.get("exact_float") or 0)
                                                        - 10 / 3) < 1e-9), None)
            check("节点电位 = 10/3 V（C 开路 + L 短路后 1k∥1k 分压）",
                  two is not None,
                  {k: v.get("exact") for k, v in nt.items()})

        # ---------------------------------------------------------- 落盘
        banner("⑥ 落盘：runs/ 与 config/ 真的建在 exe 同级了")
        runs = work / "runs"
        cfg = work / "config"
        check("runs/ 建在 exe 同级", runs.is_dir(), str(runs))
        check("会话目录真的写进去了（不是只建了空壳）",
              runs.is_dir() and any(runs.iterdir()),
              f"{len(list(runs.iterdir())) if runs.is_dir() else 0} 项")
        check("config/ 建在 exe 同级", cfg.is_dir(), str(cfg))
        # ★ 真正的判据：包内解压目录里**不该**有 runs/。
        #   有就说明路径没分家，数据会随进程退出一起没。
        rr2 = Path(paths.get("resource_root", ""))
        check("★ 包内解压目录里没有 runs/（有就说明路径没分家）",
              not (rr2 / "runs").exists(), str(rr2 / "runs"))
        # 模板是代码生成的，不依赖包里有没有那份 json
        check("视觉配置模板已铺好（缺 key 也能跑，key 要用户自己填）",
              (cfg / "secrets.example.json").is_file(),
              str(cfg / "secrets.example.json"))

        banner("启动横幅（用户双击后看到的第一屏）")
        text, log_enc = read_log(log)
        ok_dec = BANNER_MARK in text
        check("★ 启动日志能正确解码（不是乱码）", ok_dec,
              f"编码 {log_enc}" if ok_dec
              else f"用 {log_enc} 解出来看不到横幅首行")
        i = text.find(BANNER_MARK)
        print(text[i - 4:i + 1800] if i >= 0 else text[-1800:])

        # ---------------------------------------------------------- 横幅说实话
        banner("⑦ 横幅不许自相矛盾（状态显示说谎 = 用户按它判断，然后踩空）")
        key_ok = bool((vis.get("vlm") or {}).get("api_key_present"))
        for ln in text.splitlines():
            if "视觉大模型" in ln:
                if not key_ok:
                    # 本机没填 key，于是这一行**必须**如实说缺 key，
                    # 不能同时出现"已配置"这种绿点话术。
                    check("★ 缺 key 时横幅不说「已配置」", "已配置" not in ln,
                          ln.strip())
                    check("  └ 并且明确说清是缺 key", "key" in ln, ln.strip())
                break
        else:
            check("横幅里有「视觉大模型」这一行", False, "没找到")

    finally:
        banner("⑧ 收尾：必须把 exe **连同它的子进程**收干净")
        stop_tree(proc)
        # ★ 端口重新空出来 = 服务真的没了。比"进程对象说它退了"可信。
        port_gone = not port_alive(port)
        check("★ 端口已释放（没有孤儿进程还在服务）", port_gone,
              "连不上了" if port_gone else f"127.0.0.1:{port} 仍能连上")
        print()
        if a.keep:
            print(f"  临时目录保留在：{work}")
        else:
            # ★ 不许 ignore_errors。第一次跑就把 8 个临时目录悄悄漏在了 %TEMP%
            #   —— 因为孤儿子进程还占着里面的文件，rmtree 失败得无声无息。
            #   清理失败必须能被看见，不然"跑一次验一遍"就会慢慢攒一地垃圾。
            try:
                shutil.rmtree(work)
                cleaned, why = not work.exists(), ""
            except OSError as e:
                cleaned, why = False, f"{type(e).__name__}: {e}"
            check("★ 临时目录真的删掉了（不是「看着删了」）", cleaned,
                  why or str(work))
            print(f"  临时目录 {work} —— "
                  + ("已清理" if cleaned else "**没删掉**（见上）"))

        # ★ 我们自己产生的垃圾自己收：上面是**硬杀**（taskkill /F），引导器
        #   没机会清它自己的解压目录，于是每跑一次验证就在 %TEMP% 漏一个
        #   ~250 MB 的 _MEI*。启动时那个 sweep_stale_mei() 能兜住（2 小时门槛），
        #   但兜底不该拿来当日常 —— 验证跑得勤，两小时里能攒好几个。
        #   这个路径是从 /api/health 的 paths.resource_root 拿的，精确。
        rr_exe = ((live or {}).get("paths") or {}).get("resource_root")
        if rr_exe:
            p = Path(rr_exe)
            if p.is_dir() and p.name.startswith("_MEI"):
                try:
                    shutil.rmtree(p)
                    print(f"  已清理 exe 的解压目录 {p}")
                except OSError as e:
                    print(f"  解压目录没删掉 {p} —— {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    print(f"  exe 验证失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
