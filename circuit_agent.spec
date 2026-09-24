# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（单文件、带控制台）。

★ 三处不是"顺手写的"，每一处都对应一个坑：

1. **``vendor/ngspice`` 必须进包**。``ngspice.dll`` 是第三法唯一依赖，
   而 PySpice **并不自带**它 —— 目标机没装 KiCad 就只剩两法。
   打进包里之后 ``app/paths.bundled_ngspice()`` 会在 ``sys._MEIPASS`` 下
   找到它，查找次序还排在"用户机器上碰巧装的那个 KiCad"**之前** ——
   这样用的才是我们测过的那个版本。许可证一起带上（BSD 系，要求保留声明）。

2. **``winrt`` 必须整体收集**。OCR 那几个 ``from winrt....`` 写在函数体里，
   而 ``winrt`` 是个**命名空间包**、真正的实现是 ``winrt/_winrt_windows_*.pyd``
   这些原生扩展。少收一个，表现是"OCR 不可用"这个**看起来完全正常**的
   降级分支 —— 而不是报错。所以这里用 ``collect_submodules`` 收全。

3. **``uvicorn`` 必须整体收集**。它的协议/事件循环实现是**运行时按名字挑**的
   （``uvicorn.protocols.http.auto`` 之类），静态分析看到的是一个变量而不是
   导入语句。漏了它，exe 起来之后一收到请求就 500。

排除项都是**确认过没被 app/ 引用**的：``sympy`` / ``lxml`` 装了但一行都没用，
``tkinter`` / ``matplotlib`` / ``pandas`` / Jupyter 全家都不在依赖里。
排除它们纯粹是给产物瘦身，不影响功能。
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

#: spec 所在目录 = 项目根（本文件放在项目根下）
ROOT = Path(SPECPATH).resolve()

# ---------------------------------------------------------------- 资源

datas = [
    # 前端（唯一需要随包走的静态资源，132KB）
    (str(ROOT / "web" / "index.html"), "web"),
    # 内置 ngspice：dll + 许可证声明。落点是 ``<资源根>/vendor/ngspice/``，
    # 与 ``app/paths.bundled_ngspice()`` 找的位置**必须一致**。
    (str(ROOT / "vendor" / "ngspice"), "vendor/ngspice"),
]

# ★ PySpice 的**数据文件**必须单独收，PyInstaller 默认只收 .py 与二进制。
#   它只要两个：``PySpice/Spice/NgSpice/api.h``（运行时读，缺了直接
#   FileNotFoundError）与 ``PySpice/Config/logging.yml``。
#   这个坑的表现极其隐蔽：**不打包时一切正常**，打出来之后
#   ngspice 每一道题都失败，而探针（当时只 import 一下）还报"可用" ——
#   界面显示三法齐全，第三法其实是空的。现在探针也改成真跑一次仿真了，
#   两边一起把这类"看起来正常的缺件"堵住。
datas += collect_data_files("PySpice", include_py_files=False)

# ---------------------------------------------------------------- 隐式导入

hiddenimports: list[str] = []
hiddenimports += collect_submodules("winrt")      # 见文件头第 2 条
hiddenimports += collect_submodules("uvicorn")    # 见文件头第 3 条
hiddenimports += collect_submodules("PySpice")    # 它内部也有按名挑模块的地方
hiddenimports += [
    "multipart",              # starlette 解析 multipart 表单（上传）时要用
    "python_multipart",
    "anyio",
    "anyio._backends._asyncio",
]

# ---------------------------------------------------------------- 排除

excludes = [
    # 装了但**一行都没被引用**的（grep 过 app/ 全目录）
    "sympy", "lxml",
    # 图形界面工具链（这是个 WebUI，不需要任何桌面 GUI 库）
    "tkinter", "turtle", "idlelib",
    "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
    # 数据分析与笔记本全家桶
    "pandas", "IPython", "jupyter", "notebook", "nbformat", "nbconvert",
    # 测试与打包工具自己
    "pytest", "_pytest", "nose", "hypothesis",
    "setuptools", "pkg_resources", "pip", "wheel",
    # numpy 自带的编译/测试脚手架
    "numpy.f2py", "numpy.distutils",
]

# ---------------------------------------------------------------- 图标

_icon = ROOT / "build" / "app.ico"
icon = str(_icon) if _icon.is_file() else None

# ---------------------------------------------------------------- 构建

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="circuit_agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # ★ UPX 保持关闭：高压壳会把 cv2/numpy 的 DLL 压坏（杀软也更容易误报），
    #   而产物本来就要靠 zip 分发，省那几十 MB 不值当。
    upx=False,
    runtime_tmpdir=None,
    console=True,          # 用户选了"保留控制台"：出问题能看见原因
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
