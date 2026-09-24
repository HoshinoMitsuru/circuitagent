"""双击 exe 的入口：起本地服务 + 自动开浏览器。

为什么不直接让 exe 跑 ``app/api/server.py``：双击之后用户看到的**第一屏**
就是这个控制台窗口，它必须把三件事说清楚，否则用户只能靠猜 ——

1. **服务在哪**（地址与端口，端口被占时会自动换一个）；
2. **我的文件存在哪**（会话、上传的原件、密钥文件；打包后数据目录可能
   因为 exe 同级不可写而落到用户目录，这件事必须写在脸上）；
3. **三法与 OCR 到底能不能用**（缺了哪一路、从哪来的 —— 这是"不许静默降级"
   在启动期的落点：第三法没了就得当场说，不能等用户算完题才发现对账表少一行）。

另外两个容易忽略的细节：

- **服务就绪之后再开浏览器**。先开浏览器只会让用户看到"无法访问此网站"，
  然后自己按 F5 —— 看起来像程序坏了。
- **``multiprocessing.freeze_support()``**。冻结后的 exe 只要被"以子进程方式
  重新执行"（某些库会这么干），就会再跑一遍 ``main()`` ——
  表现是**浏览器被反复打开 / 服务互相抢端口**。这一行是防它的。
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

#: 优先用的端口。被占用时依次往后试。
PREFERRED_PORT = 8765
PORT_TRIES = 40
HOST = "127.0.0.1"

RULE = "=" * 68

#: 残留解压目录的年龄门槛（秒）。见 ``sweep_stale_mei()``。
STALE_MEI_AGE = 2 * 3600
#: 关掉清理（排查问题时用）
NO_SWEEP_ENV = "CIRCUIT_AGENT_NO_TEMP_SWEEP"
#: 覆盖年龄门槛（测试用）
SWEEP_AGE_ENV = "CIRCUIT_AGENT_TEMP_SWEEP_AGE"


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def sweep_stale_mei(base: Path | None = None, age: float | None = None,
                    keep: Path | None = None) -> dict[str, Any]:
    """清掉**上次被强杀**留下的 ``_MEI*`` 解压目录。

    ★ 为什么这件事得我们自己做：单文件 exe 每次启动都要把 ~250 MB 解压到
      临时目录，正常退出时由引导器自己删掉。但**只要上次是被强杀的**
      （任务管理器结束进程、控制台被强行关掉、机器崩），那个目录就留在
      ``%TEMP%`` 里没人管。实测本机被测了一下午就攒了 **19 个 / 3.0 GB**。

    ★★ 两道闸门，缺一不可 —— 这是"删别人的东西"的代码，必须保守：

      1. **名字必须以 ``_MEI`` 开头**，且在临时目录的**顶层**。其它一律不碰。
      2. **时效门槛**（默认 2 小时）：太新的目录不动。刚启动的实例正在往
         自己那个目录里解压，此时目录可能还没有任何打开的文件句柄 ——
         光靠第 3 条判不出来。而解压只要几秒，2 小时远超这个窗口。
      3. **改名试锁**：Windows 上**正在被使用的目录改不了名**（实测
         「拒绝访问」，因为引导器持有目录句柄且没开 FILE_SHARE_DELETE）。
         改名成功的，就证明没有任何进程还在用它 —— 这时才敢删。
         删完如果失败，把名字改回去，不留怪名。

      自己正在用的那个（``sys._MEIPASS``）永远跳过。

    返回一份统计，交给横幅如实显示 —— 悄悄删东西也不行。
    """
    if os.environ.get(NO_SWEEP_ENV):
        return {"skipped": f"环境变量 {NO_SWEEP_ENV} 已设", "removed": 0,
                "freed": 0, "busy": 0, "failed": 0}

    root = base if base is not None else Path(tempfile.gettempdir())
    if age is None:
        try:
            age = float(os.environ.get(SWEEP_AGE_ENV, STALE_MEI_AGE))
        except ValueError:
            age = STALE_MEI_AGE

    mine = keep if keep is not None else getattr(sys, "_MEIPASS", None)
    try:
        mine_p = Path(mine).resolve() if mine else None
    except OSError:
        mine_p = None

    now = time.time()
    removed = freed = busy = failed = 0
    try:
        cands = sorted(p for p in root.glob("_MEI*") if p.is_dir())
    except OSError:
        cands = []

    for d in cands:
        try:
            if mine_p is not None and d.resolve() == mine_p:
                continue                       # 自己，永远不动
            if now - d.stat().st_ctime < age:  # Windows 上 st_ctime 是创建时间
                continue                       # 太新：可能是刚启动的实例
        except OSError:
            continue
        probe = d.with_name(d.name + ".stale")
        try:
            d.rename(probe)                    # ★ 闸门 3：改得了名 = 没人用它
        except OSError:
            busy += 1
            continue
        size = _dir_size(probe)
        try:
            shutil.rmtree(probe)
            removed += 1
            freed += size
        except OSError:
            failed += 1
            try:
                probe.rename(d)                # 删不掉就放回去
            except OSError:
                pass
    return {"skipped": "", "removed": removed, "freed": freed,
            "busy": busy, "failed": failed, "age": age, "root": str(root)}


def _dw(s: str) -> int:
    """字符串在等宽终端里的**显示宽度**：全角字符算 2。

    ★ 不能用 ``len()``：``len("位图通道")`` 是 4，但它在控制台里占 8 列。
      用 len 去补空格，横幅的列就会歪 —— 而这一屏是用户对程序的第一印象。
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _dw(s))


def _line(mark: str, label: str, value: str = "", note: str = "") -> None:
    """横幅里的一行：``● 标签  值  （备注）``，标签列对齐。"""
    if not label:
        print(f"    {mark} {value}" + (f"  {note}" if note else ""))
        return
    print(f"    {mark} {_pad(label, 24)}{value}" + (f"  （{note}）" if note else ""))


def _port_free(port: int, host: str = HOST) -> bool:
    """真去 bind 一下 —— 与"能不能连上"是两回事，占用检测必须用 bind。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port() -> tuple[int, str]:
    """返回 ``(端口, 说明)``。说明用来告诉用户"为什么不是 8765"。"""
    for i in range(PORT_TRIES):
        p = PREFERRED_PORT + i
        if _port_free(p):
            if i == 0:
                return p, f"用默认端口 {p}"
            return p, (f"{PREFERRED_PORT} 起已被占用（上次没退干净？别的程序占着？），"
                       f"改用 {p}")
    raise RuntimeError(
        f"从 {PREFERRED_PORT} 往后 {PORT_TRIES} 个端口都被占用了。"
        f"请关掉占用端口的程序，或设环境变量 CIRCUIT_AGENT_PORT 指定一个空闲端口。")


def _health(port: int, timeout: float = 1.5) -> dict | None:
    """探一次 /api/health，成功返回解析后的 JSON，失败返回 None。"""
    import json
    url = f"http://{HOST}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _wait_ready(port: int, deadline_s: float = 40.0) -> dict | None:
    end = time.time() + deadline_s
    while time.time() < end:
        h = _health(port)
        if h:
            return h
        time.sleep(0.15)
    return None


def _print_banner(port: int, port_note: str, live: dict,
                  sweep: dict | None = None) -> None:
    from app.solver.ngspice import probe_availability

    ng = live.get("ngspice") or probe_availability()
    vis = live.get("vision") or {}
    # 字段名以 VP.probe() 的真实返回为准（别凭印象写）：
    #   local.ocr.{available,engine_language,installed_languages,reason,hint}
    #   vlm.{configured,api_key_present,model}、will_call_vlm、problems、hint
    loc = vis.get("local") or {}
    ocr = loc.get("ocr") or {}
    wires = loc.get("wires") or {}
    vlm = vis.get("vlm") or {}
    paths = live.get("paths") or {}

    print()
    print(RULE)
    print("  电路读图 · 三法对账  —  本地服务已启动")
    print(RULE)
    print(f"  界面地址   http://{HOST}:{port}/")
    print(f"             {port_note}")
    print()
    print(f"  运行形态   {paths.get('mode', '?')}")
    print(f"  数据目录   {paths.get('data_root', '?')}")
    print(f"             └ {paths.get('data_root_reason', '')}")
    print(f"     会话/原件  {paths.get('runs_dir', '?')}")
    print(f"     配置/密钥  {paths.get('config_dir', '?')}")
    if sweep and sweep.get("freed"):
        _line("", "临时目录",
              f"回收了 {sweep['removed']} 个上次残留的解压目录",
              f"{sweep['freed'] / 1024 / 1024:.0f} MB")
    print()

    # ---- 三法：缺哪一路必须当场说 ----
    print("  求解三方（本程序的可信度全部来自这里）")
    _line("●", "节点电压法", "(精确有理数 MNA)")
    _line("●", "支路电流法", "(生成树基本回路)")
    if ng.get("available"):
        _line("●", "ngspice", "(第三方独立实现)")
        _line("", "来源", str(ng.get("origin", "?")))
        _line("", "文件", str(ng.get("dll", "?")))
        if ng.get("verified"):
            # ★ 把"我们是怎么判的"也写出来。这个字段存在的唯一理由就是
            #   证明可用性**不是**靠「import 一下没报错」得出的 ——
            #   打包版曾经就是在这个地方撒过谎：横幅说三法齐全，
            #   而每一道题的对账表里第三法都是空的（PySpice 运行时要读的
            #   api.h 没被打进包），且没有任何地方告诉用户。
            _line("", "判定依据", str(ng["verified"]))
    else:
        _line("○", "ngspice", "** 不可用 —— 三法只剩两法! **")
        _line("", "原因", str(ng.get("reason", "?")))
        _line("", "提示", str(ng.get("hint", "")))
        print("      ★ 两法互校拦不住「网表抄错 / 参考方向定反 / 参考节点选错」")
        print("        这类错 —— 它们共用同一份 IR，会一致地给出同一个错答案。")

    # ---- 视觉层 ----
    print()
    print("  视觉层（照片通道；不使用照片的话，下面都不影响）")
    _line("●" if vis.get("enabled") else "○", "位图通道",
          "已开启" if vis.get("enabled") else "已关闭")
    if ocr:
        ok = bool(ocr.get("available"))
        _line("●" if ok else "○", "系统 OCR", "可用" if ok else "不可用")
        if ok:
            libs = ocr.get("installed_languages") or []
            _line("", "识别语言", "、".join(map(str, libs)) or "?")
            missing = ocr.get("missing_languages") or []
            if missing:
                # ★ 缺语言包必须说：中文电路图是中西混排，缺中文包正是
                #   "整图文字全空"最常见的原因。
                _line("", "未安装语言包", "、".join(map(str, missing)))
        else:
            _line("", "原因", str(ocr.get("reason") or "未知"))
            if ocr.get("hint"):
                _line("", "提示", str(ocr["hint"]))
    else:
        _line("○", "系统 OCR", "没有探测结果")
    if wires:
        _line("●" if wires.get("ready") else "○", "本地几何层",
              "就绪" if wires.get("ready") else "不可用",
              f"opencv {wires.get('opencv', '?')} / scipy {wires.get('scipy', '?')}")
    # ★ 这里不能只看 will_call_vlm。"升级策略不是 never + 通道开着"并不等于
    #   "真能调起来" —— 还要有 key。少了这个判断，横幅会打出
    #   「● 视觉大模型 已配置」，而后面紧跟一句「缺 key」——
    #   一个绿点加一句自相矛盾的话，用户只会看绿点。
    #   这和"横幅说三法齐全、第三法其实是空的"是同一类错：**别让状态显示说谎。**
    key_ok = bool(vlm.get("api_key_present"))
    if vis.get("will_call_vlm") and key_ok:
        _line("●", "视觉大模型", "已配置，结构识别不出来时会升级",
              f"模型 {vlm.get('model', '?')}")
    elif vis.get("will_call_vlm"):
        _line("○", "视觉大模型", "缺 key —— 需要升级时调不起来",
              f"模型 {vlm.get('model', '?')}")
        if vis.get("hint"):
            print(f"        {vis['hint']}")
    else:
        _line("○", "视觉大模型", "不会调用（没配 key —— 这是正常状态）")
        if vis.get("hint"):
            print(f"        {vis['hint']}")
    for p in (vis.get("problems") or []):
        print(f"    ! {p}")

    print()
    print(RULE)
    print("  浏览器会自动打开。**关掉这个窗口就是退出程序**，")
    print("  或在窗口里按 Ctrl+C。")
    print(RULE)
    print()
    # 这一屏是排错时唯一要看的东西，强制刷一次，别留在缓冲区里。
    sys.stdout.flush()


def main() -> int:
    # ★ 必须在最前面：冻结后的 exe 一旦被以子进程方式重新执行，
    #   不加这一行就会把 main() 再跑一遍（反复开浏览器 / 抢端口）。
    multiprocessing.freeze_support()

    # ★ 这个窗口的全部意义就是"把情况说清楚"，所以输出**必须立刻可见**。
    #   默认情况下 Python 只在 stdout 接着终端时按行刷新；一旦输出被重定向
    #   （用户存日志、我们做无头验证），它就变成整块缓冲 —— 表现是
    #   "窗口里什么都没有，直到进程结束才一股脑刷出来"，
    #   而进程通常是被关窗口/Ctrl+C 结束的，于是那点输出**全丢了**。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)   # type: ignore[union-attr]
        except Exception:                             # noqa: BLE001
            pass

    #: 先把可写目录铺出来。横幅马上要告诉用户"配置在哪"，
    #: 而指向一个还不存在的目录等于让人白找一趟。
    from app.paths import ensure_data_dirs
    _boot = ensure_data_dirs()
    if _boot.get("warning"):
        print(f"[警告] {_boot['warning']}")
    if _boot.get("created"):
        print(f"[首次运行] 已铺好配置模板：{_boot['created']}")
        print("           （要接视觉大模型就往 api_key 里填；不填也能用，本地 OCR 层不要 key）")

    # 顺手清掉上次被强杀留下的解压目录（单文件模式一次 ~250 MB）。
    # 受控：只认 _MEI*、只动够旧的、改名成功才删、自己那个永远跳过。
    _sweep = sweep_stale_mei()
    if _sweep.get("freed"):
        print(f"[清理] 回收了上次残留的解压目录 {_sweep['removed']} 个，"
              f"释放 {_sweep['freed'] / 1024 / 1024:.0f} MB")
    if _sweep.get("failed"):
        print(f"[清理] 有 {_sweep['failed']} 个残留目录没删掉（可能正被别的实例占用）")

    port_env = os.environ.get("CIRCUIT_AGENT_PORT")
    if port_env and port_env.isdigit():
        port = int(port_env)
        if not _port_free(port):
            print(f"[错误] 环境变量指定的端口 {port} 已被占用。")
            return 2
        port_note = "来自环境变量 CIRCUIT_AGENT_PORT"
    else:
        try:
            port, port_note = _pick_port()
        except RuntimeError as e:
            print(f"[错误] {e}")
            return 2

    # 先把"服务就绪"的判断条件准备好，再起服务
    try:
        import uvicorn

        from app.api.server import app
    except Exception:
        print("[错误] 载入程序模块失败 —— 这个包可能不完整，或依赖缺失。")
        print("       完整报错如下：")
        traceback.print_exc()
        return 3

    config = uvicorn.Config(app, host=HOST, port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)

    # ★ 服务跑在后台线程：主线程要留着接 Ctrl+C ——
    #   uvicorn 只在**主线程**里装信号处理器，放进线程就等于放弃了 Ctrl+C。
    t = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    t.start()

    live = _wait_ready(port)
    if live is None:
        print(f"[错误] 服务在 40 秒内没有就绪（端口 {port}）。")
        if t.is_alive():
            print("       进程还在跑但没响应，请把本窗口的内容反馈给开发者。")
        else:
            print("       服务线程已经退出，多半是启动时就抛异常了。")
        return 4

    _print_banner(port, port_note, live, _sweep)

    url = f"http://{HOST}:{port}/"
    #: 不开浏览器的两种情形：显式要求（脚本化调用 / 无头验证），
    #: 或者跑在无桌面会话里。都不是错误，只是少做一步。
    want_browser = "--no-browser" not in sys.argv \
        and not os.environ.get("CIRCUIT_AGENT_NO_BROWSER")
    if want_browser:
        try:
            webbrowser.open(url)
        except Exception:
            print(f"（自动打开浏览器失败，请手动访问 {url}）")
    else:
        print(f"（已按要求不自动打开浏览器，请手动访问 {url}）")

    try:
        while t.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止服务…")
        server.should_exit = True
        t.join(timeout=8)
        print("已停止。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:                                # noqa: BLE001
        # ★ 双击运行时控制台会**一闪而过**，用户只看到"闪了一下就没了"。
        #   所以任何意外都要先把栈打全，再等一次回车 —— 让人有机会把这段话
        #   截图发回来。这比"干净地退出"重要得多。
        print()
        print(RULE)
        print("  程序异常退出。请把下面这段完整内容反馈给开发者：")
        print(RULE)
        traceback.print_exc()
        print(RULE)
        try:
            input("按回车键关闭窗口…")
        except EOFError:
            pass
        sys.exit(9)
