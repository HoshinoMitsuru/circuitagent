"""把项目打包成单文件 exe。

用法::

    python tools/build_exe.py              # 完整构建
    python tools/build_exe.py --sync-only  # 只把 ngspice 归置好，不打包

它做四件事，顺序不能换：

1. **归置内置 ngspice**（``vendor/ngspice/``）。源可以是环境变量
   ``CIRCUIT_AGENT_NGSPICE_DLL``、仓库里已有的一份、或本机 KiCad 的安装位置。
   复制过来之后**算一遍 SHA256 并写进许可证声明** —— 因为它是随 exe 一起
   分发出去的第三方二进制，"这个文件到底是哪一个"必须可追溯，
   否则下次换台机器构建出行为不同的包就没人说得清了。
2. **生成图标**（``build/app.ico``）。没有图标的 exe 在资源管理器里是一个
   通用白框，双击之前分不清是什么东西。
3. **跑 PyInstaller**（配置在 ``circuit_agent.spec``）。
4. **报告产物**：路径、大小、以及"包里到底带了没有 ngspice"的自检。

★ 为什么要有第 1 步而不是直接引用 KiCad 目录里的 dll：
  那个路径**只在这台机器上成立**。构建机上碰巧装了 KiCad，不等于用户机器上
  也有 —— 而缺了它第三法就没了，只剩两法互校（拦不住共同 IR 里的错）。
  所以必须把它**复制进仓库**，让包自给自足、构建可复现。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VENDOR_DIR = ROOT / "vendor" / "ngspice"
VENDOR_DLL = VENDOR_DIR / "ngspice.dll"
LICENSE_FILE = VENDOR_DIR / "LICENSE.ngspice.txt"
ICON_FILE = ROOT / "build" / "app.ico"
SPEC = ROOT / "circuit_agent.spec"
DIST = ROOT / "dist"
EXE = DIST / "circuit_agent.exe"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. ngspice


def _find_source_dll() -> tuple[Path, str]:
    """返回 ``(路径, 来源说明)``。次序与运行时一致，免得构建/运行两边不一致。"""
    env = os.environ.get("CIRCUIT_AGENT_NGSPICE_DLL")
    if env and Path(env).is_file():
        return Path(env), "环境变量 CIRCUIT_AGENT_NGSPICE_DLL"

    # 已经在仓库里的那份：**优先于**本机 KiCad。
    # 否则构建机升级 KiCad 之后，同一份源码会悄悄构建出不同的包。
    if VENDOR_DLL.is_file():
        return VENDOR_DLL, "仓库内已有的 vendor/ngspice/ngspice.dll"

    from app.solver.ngspice import find_ngspice_dll
    p = find_ngspice_dll()
    if p is not None:
        return p, "本机 KiCad 安装位置"
    raise SystemExit(
        "[错误] 找不到 ngspice.dll。请任选一种方式提供：\n"
        "        1) 安装 KiCad（自带 ngspice），或\n"
        "        2) 设 CIRCUIT_AGENT_NGSPICE_DLL 指向 ngspice.dll，或\n"
        f"        3) 手工把 ngspice.dll 放到 {VENDOR_DLL}\n"
        "      第三法（ngspice）是三法互校里唯一的外部实现，缺了它只剩两法。")


_LICENSE_TEMPLATE = """\
ngspice —— 随本程序一起分发的第三方二进制
================================================================================

本文件说明 vendor/ngspice/ngspice.dll 的来源与许可，供再分发时保留声明之用。


一、来源（本次构建）
--------------------------------------------------------------------------------
文件      {name}
大小      {size} 字节
SHA256    {sha}
ngspice   版本 {version}
取自      {source}
取得时间  {when}
构建产物  circuit_agent.exe（PyInstaller 单文件）

★ 保留 SHA256 的用意：这个 dll 是**随 exe 一起发出去的**。日后有人报告
  "换台机器结果对不上"，第一件要确认的事就是两边带的 ngspice 是不是同一个
  文件 —— 有哈希才比得出来。


二、本程序怎么用它
--------------------------------------------------------------------------------
- **未做任何修改**（原样复制）。
- 以**共享库**方式动态调用（ctypes / PySpice 的 ``NgSpiceShared``），
  没有静态链接、没有把它链接进本程序自己的可执行代码。
- 只调用其对外公开的仿真接口：把网表交给它跑 ``.op``（直流工作点），
  取回节点电压与支路电流，用于三法互校里的第三方对账。
- 它是**可选**依赖：找不到时程序照常运行，但报告会明确标出
  "第三法缺失、只剩两法"，绝不静默降级。


三、许可状况（★ 重要：ngspice 整体**不适用单一许可证**）
--------------------------------------------------------------------------------
ngspice 官方 COPYING 的开头写明："The Ngspice branch as a whole will not be
covered by a specific license." 它由多个来源的代码组成，各自保留原许可：

  组件        作者                         许可
  ----------  ---------------------------  ------------------------------
  spice3f5    The Regents of the            New BSD（2007-07-17 由原许可
              University of California       改为 New BSD）
  xspice      Georgia Tech Research Corp.  Public Domain（公共领域）
  cider       University of California     Old BSD（Research Software
                                           Agreement）
  numparam    Georg Post                   LGPL
  adms        Laurent Lemaitre             LGPL
  tclspice    Stefan Jones                 LGPL

对本程序的意义：LGPL 组件（numparam / adms / tclspice）以**动态库**形式提供、
且**未被修改**，本程序通过共享库接口调用、未静态链接 —— 属于 LGPL 允许的
再分发方式。故可随本程序一同分发，但**必须保留下述声明**。

完整的、逐文件的许可清单请见 ngspice 源码包内的 COPYING 文件：
  https://ngspice.sourceforge.io/


四、spice3f5（模拟仿真核心，本程序实际用到的部分）的 New BSD 声明
--------------------------------------------------------------------------------
Copyright (c) 1985-1991 The Regents of the University of California.
All rights reserved.

Permission is hereby granted, without written agreement and without license
or royalty fees, to use, copy, modify, and distribute this software and its
documentation for any purpose, provided that the above copyright notice and
the following two paragraphs appear in all copies of this software.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
"AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


五、xspice（混合信号仿真）的公共领域声明
--------------------------------------------------------------------------------
THE SOFTWARE PROGRAMS BELOW ARE IN THE PUBLIC DOMAIN AND ARE PROVIDED FREE OF
ANY CHARGE. THE GEORGIA TECH RESEARCH CORPORATION, THE GEORGIA INSTITUTE OF
TECHNOLOGY, AND/OR OTHER PARTIES PROVIDE THIS SOFTWARE "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH THE USER.
SHOULD THE PROGRAM PROVE DEFECTIVE, THE USER ASSUMES THE ENTIRE COST OF ALL
NECESSARY SERVICING, REPAIR OR CORRECTION. IN NO EVENT WILL THE GEORGIA TECH
RESEARCH CORPORATION, THE GEORGIA INSTITUTE OF TECHNOLOGY, AND/OR OTHER PARTIES
PROVIDING THE PROGRAMS BELOW BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR
INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA
BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS).


六、ngspice 自身的版权声明（dll 内嵌的那两行）
--------------------------------------------------------------------------------
** Copyright 1985-1994, Regents of the University of California.
** Copyright 2001-2025, The ngspice team.
"""


def _detect_version(dll: Path) -> str:
    """问 ngspice 自己要版本号；拿不到就写 unknown（不猜）。"""
    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.LIBRARY_PATH = str(dll)
        out = NgSpiceShared.new_instance().exec_command("version") or ""
        for line in out.splitlines():
            if "ngspice-" in line:
                return line.strip("* \t")
    except Exception:
        pass
    return "unknown（探测失败）"


def sync_ngspice(*, force: bool) -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    src, origin = _find_source_dll()

    if src.resolve() != VENDOR_DLL.resolve():
        shutil.copy2(src, VENDOR_DLL)
        print(f"  已归置 ngspice.dll ← {origin}")
    else:
        print(f"  沿用已有的 ngspice.dll（{origin}）")

    # 源就是仓库里那份、且许可证已存在时，不重写 —— 保留**最初取得时间**，
    # 否则每次构建都会把 provenance 里的时间刷成今天，追溯就失去意义了。
    if src.resolve() == VENDOR_DLL.resolve() and LICENSE_FILE.is_file() and not force:
        print("  许可证声明已存在，保留原样（含最初的取得时间与哈希）")
        return

    size = VENDOR_DLL.stat().st_size
    sha = _sha256(VENDOR_DLL)
    ver = _detect_version(VENDOR_DLL)
    LICENSE_FILE.write_text(
        _LICENSE_TEMPLATE.format(
            name=VENDOR_DLL.name, size=size, sha=sha, version=ver,
            source=origin,
            when=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
        ),
        encoding="utf-8",
    )
    print(f"  ngspice {ver}，{size / 1024 / 1024:.2f} MB，sha256 {sha[:16]}…")
    print(f"  已写出许可声明 {LICENSE_FILE.relative_to(ROOT)}")


# ---------------------------------------------------------------- 2. 图标


def make_icon(*, force: bool = False) -> Path:
    """画一个认得出来的图标：深色圆角底 + 白色电阻折线。

    为什么是电阻折线：它是电路图里唯一"一眼就是电路"的符号，
    而且到 16×16 缩略图大小仍然读得出形状（比画三极管、电容都清楚）。
    """
    if ICON_FILE.is_file() and not force:
        print(f"  图标已存在，跳过（{ICON_FILE.relative_to(ROOT)}）")
        return ICON_FILE

    from PIL import Image, ImageDraw

    ICON_FILE.parent.mkdir(parents=True, exist_ok=True)
    S = 1024                      # 先画大图再逐档降采样，边缘才干净
    bg = (22, 32, 43, 255)        # 与界面同一套墨色 #16202b
    fg = (255, 255, 255, 255)

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.2), fill=bg)

    # 折线：横向 5 段，竖直幅度 0.16S，线宽 0.10S
    w = int(S * 0.10)
    amp = S * 0.16
    cy = S / 2
    xs = [S * 0.14, S * 0.26, S * 0.38, S * 0.50, S * 0.62, S * 0.74, S * 0.86]
    ys = [cy, cy - amp, cy + amp, cy - amp, cy + amp, cy - amp, cy]
    d.line(list(zip(xs, ys)), fill=fg, width=w, joint="curve")
    # 两端各补一个小圆点，让线头是圆的（成对出现的"接线端"）
    r = w / 2
    for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fg)

    img.save(ICON_FILE, format="ICO",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                    (64, 64), (128, 128), (256, 256)])
    print(f"  已生成图标 {ICON_FILE.relative_to(ROOT)}")
    return ICON_FILE


# ---------------------------------------------------------------- 3. 打包


def _exe_locked(path: Path) -> bool:
    """上一版 exe 是不是正被占用（有人还开着它）。

    ★ 这是重建时**最常撞**的失败，而 PyInstaller 只会给一句
      ``returned non-zero exit status 1``（我自己就撞过，一秒就退出、
      看不出任何原因）。判出来之后就能把话说到点子上。

    ★★ 判据是**实测选的，不是猜的**。在运行中的 exe 上实测三种做法：

      ================  ====================  ==========================
      做法              结果                  能否当判据
      ================  ====================  ==========================
      ``rename()``      **成功**              不能 —— Windows 允许给
                                             运行中的镜像改名（映射的是
                                             内容不是目录项），会漏判
      ``open('r+b')``   Permission denied     **可以**，且非破坏性
      ``unlink()``      拒绝访问              可以，但会真把文件删掉
      ================  ====================  ==========================

      我第一版写的就是 rename —— 在运行中的 exe 上返回 False，
      等于这个提示永远不会出现，白写。所以这里留一行实测结论。
    """
    if not path.is_file():
        return False
    try:
        with open(path, "r+b"):
            return False
    except OSError:
        return True


def run_pyinstaller() -> None:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           str(SPEC)]
    print("  $ " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        hint = ""
        if _exe_locked(DIST / "circuit_agent.exe"):
            hint = (
                "\n  ★ dist\\circuit_agent.exe 正被占用 —— 多半是上一次"
                "验证/试跑的那个还在运行。\n"
                "    关掉它的控制台窗口（或在任务管理器里结束 "
                "circuit_agent.exe）之后重跑本脚本即可。"
            )
        raise SystemExit(f"[错误] PyInstaller 返回 {r.returncode}{hint}")


def emit_third_party_notices() -> Path | None:
    """把第三方声明放到 ``dist/`` 里，**与 exe 并排**。

    ★ 为什么不能只塞进包里：ngspice 的许可（BSD 系）要求再分发时**保留声明**。
      声明确实被打进包内了（``vendor/ngspice/LICENSE.ngspice.txt``，可用
      ``pyi-archive_viewer`` 取出来），但对拿到单个 .exe 的用户来说那等于没有 ——
      他看不到。所以发版时**必须把这份声明和 exe 一起给出去**，
      文件名取成一看就知道该留着的样子。
    """
    if not LICENSE_FILE.is_file():
        return None
    dst = DIST / "THIRD-PARTY-NOTICES.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LICENSE_FILE, dst)
    return dst


# ---------------------------------------------------------------- 4. 报告


def report() -> None:
    print()
    print("=" * 72)
    if not EXE.is_file():
        raise SystemExit(f"[错误] 没看到产物 {EXE}")
    mb = EXE.stat().st_size / 1024 / 1024
    print(f"  产物：{EXE}")
    print(f"  大小：{mb:.1f} MB")

    # ★ 自检：包里到底有没有 ngspice。缺了它**不会报任何错**，只会少一法 ——
    #   所以必须在这里主动确认一次，而不是等用户算完题才发现对账表少一行。
    if VENDOR_DLL.is_file():
        dll_mb = VENDOR_DLL.stat().st_size / 1024 / 1024
        print(f"  内置 ngspice：有（{VENDOR_DLL.relative_to(ROOT)}，{dll_mb:.1f} MB）")
        print(f"  许可声明：{'有' if LICENSE_FILE.is_file() else '**缺失**'}"
              f"（{LICENSE_FILE.relative_to(ROOT)}）")
    else:
        print("  内置 ngspice：**没有** —— 打出来的包在没装 KiCad 的机器上只剩两法！")
    print("=" * 72)
    print()
    print("  下一步：把 dist/circuit_agent.exe 拷到一个**空目录**里双击运行。")
    print("  首次启动要把包解压到临时目录，慢一些（十几秒）属正常。")
    print("  运行时产生的 runs/ 与 config/ 会落在 exe 同级目录。")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="打包为单文件 exe")
    ap.add_argument("--sync-only", action="store_true",
                    help="只归置 ngspice 与图标，不调用 PyInstaller")
    ap.add_argument("--force-license", action="store_true",
                    help="强制重写 ngspice 许可证声明（会刷新取得时间与哈希）")
    ap.add_argument("--force-icon", action="store_true", help="强制重画图标")
    args = ap.parse_args()

    print("=" * 72)
    print("  1/4  归置内置 ngspice（第三法的唯一依赖）")
    print("=" * 72)
    sync_ngspice(force=args.force_license)

    print()
    print("=" * 72)
    print("  2/4  生成图标")
    print("=" * 72)
    make_icon(force=args.force_icon)

    if args.sync_only:
        print("\n  （--sync-only：到此为止）")
        return 0

    print()
    print("=" * 72)
    print("  3/4  打包（单文件，约需 1~3 分钟）")
    print("=" * 72)
    run_pyinstaller()

    print()
    print("=" * 72)
    print("  4/4  结果")
    print("=" * 72)
    report()
    notices = emit_third_party_notices()
    if notices is not None:
        print(f"  ★ 发版时请把 {notices.relative_to(ROOT)} 与 exe **一起**给出：")
        print("    它含 ngspice 的许可声明，再分发时按许可要求必须随附。")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
