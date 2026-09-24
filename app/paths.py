"""运行时路径：**只读资源**与**可写数据**必须分开。

★ 为什么必须分开：PyInstaller 的单文件模式把整个包解压到一个临时目录
（``sys._MEIPASS``），那是**每次运行都不一样、进程退出即删**的地方。
项目原来的写法 ``ROOT = Path(__file__).resolve().parents[2]``
在开发时完全正确，冻结之后却会把 ``runs/`` 落进那个临时目录 ——
**用户上传的原件、每次求解的会话，退出即失**，而界面看起来一切正常。
这是"静默降级"在打包层面的一种变体：不是少算了什么，是**少存了**，
而且没有任何地方会报错。

于是定两条口径：

- ``resource_root()``：**只读**。打进包里的东西（``web/index.html``、
  内置的 ``ngspice.dll``）。冻结后就是 ``sys._MEIPASS``。
- ``data_root()``：**可写**。跑起来才产生的（``runs/``、``config/``）。
  冻结后优先放 **exe 同级**（便携：整个文件夹搬 U 盘就能走）；
  同级不可写（装在 ``Program Files``、只读盘、网络盘）才回退到
  ``%LOCALAPPDATA%\\circuit_agent``。

**回退这件事必须留痕**：``data_root_reason()`` 会给出"为什么落在这儿"，
启动横幅与 ``/api/health`` 都把它显示出来 —— 否则用户会遇到
"我明明放在 D 盘，配置却写到了 C 盘"，而没有任何地方告诉他原因。
"""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

#: 数据目录名（回退到用户目录时用）
APP_NAME = "circuit_agent"

#: 内置 ngspice 在这两个位置找（前者=打包后的布局，后者=开发时的仓库布局）
_NGSPICE_RELATIVE = (
    ("ngspice.dll",),
    ("vendor", "ngspice", "ngspice.dll"),
)


def is_frozen() -> bool:
    """是不是 PyInstaller 打出来的 exe。``sys.frozen`` 只在冻结后存在。"""
    return bool(getattr(sys, "frozen", False))


@lru_cache(maxsize=1)
def resource_root() -> Path:
    """**只读**资源根：``web/``、内置 ``ngspice.dll`` 都在它下面。

    冻结后是 ``sys._MEIPASS``（单文件=解压出的临时目录，单目录=``_internal``）。
    开发时是项目根。**永远不要往这里面写东西。**
    """
    if is_frozen():
        # 单文件模式一定有 _MEIPASS；单目录模式也有（指向 _internal）。
        # 万一将来某种打包方式不设它，退到 exe 所在目录也比抛异常好。
        return Path(getattr(sys, "_MEIPASS", None) or Path(sys.executable).parent)
    return Path(__file__).resolve().parent.parent


def exe_dir() -> Path:
    """程序所在目录：冻结后是 exe 同级，开发时是项目根。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _writable(d: Path) -> bool:
    """真去写一个文件试试，而不是看 ``os.access``。

    ★ 只看 ``os.access`` 在 Windows 上会骗人：受 UAC 虚拟化、只读属性、
      网络盘与同步盘（OneDrive 占位文件）影响，它可能报"可写"而实际写失败。
      这里要判错的代价是"用户的数据悄悄写到别处去了"，所以宁可实测。
    """
    try:
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".ca_wtest_",
                                         suffix=".tmp"):
            pass
        return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def _resolve_data_root() -> tuple[Path, str]:
    """返回 ``(目录, 为什么是它)``。只算一次并缓存。"""
    if not is_frozen():
        return exe_dir(), "开发模式：直接用项目根目录"

    cand = exe_dir()
    if _writable(cand):
        return cand, "便携模式：exe 同级目录（整个文件夹可以搬走）"

    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / APP_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 连用户目录都建不出来（极端情况）——退回临时目录，
        # 至少让程序能跑，并把这件事明说出去。
        import tempfile as _tf
        d = Path(_tf.gettempdir()) / APP_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d, f"exe 同级与用户目录都不可写，临时落在 {d}（本次数据不保证留存）"
    return d, f"exe 同级不可写（装在只读位置？），回退到用户目录 {d}"


def data_root() -> Path:
    """**可写**数据根：``runs/`` 与 ``config/`` 都挂在它下面。"""
    return _resolve_data_root()[0]


def data_root_reason() -> str:
    """数据目录为什么在这儿 —— 给人看的一句话，启动横幅与 health 都用它。"""
    return _resolve_data_root()[1]


def runs_dir() -> Path:
    return data_root() / "runs"


def config_dir() -> Path:
    return data_root() / "config"


def bundled_ngspice() -> Path | None:
    """**随包自带**的 ``ngspice.dll``（与"用户机器上装没装 KiCad"无关）。

    这是"三法互校"能不能成立的前提：缺了它，第三法（ngspice）就没了，
    只剩两法 —— 而两法互校拦不住它们共同那份 IR 里的错。
    """
    for rel in _NGSPICE_RELATIVE:
        p = resource_root().joinpath(*rel)
        if p.is_file():
            return p
    return None


def ensure_data_dirs() -> dict[str, str]:
    """把可写目录铺出来（``runs/`` 与 ``config/``），并回填配置模板。

    ★ 为什么**启动时**就要铺，而不是用到才建：
      启动横幅会告诉用户"配置/密钥在哪个目录"。如果那时目录还不存在，
      用户照着路径去找会找不到 —— 这不是致命错，但会让人怀疑程序
      有没有真的在工作。铺好之后他能直接看到 ``secrets.example.json``
      并照着填 key（没 key 本来就是正常状态，本地 OCR 层不需要它）。

    返回一份"做了什么"的摘要，供横幅/日志使用；**任何一步失败都不抛**：
    没配 key 不是错误，目录建不出来也只是少一个可写位置，
    绝不该拦住程序启动。
    """
    out: dict[str, str] = {"runs": "", "config": "", "created": "", "warning": ""}
    try:
        (data_root() / "runs").mkdir(parents=True, exist_ok=True)
        out["runs"] = str(runs_dir())
    except OSError as e:
        out["warning"] = f"建 runs/ 失败：{e}"
    try:
        (data_root() / "config").mkdir(parents=True, exist_ok=True)
        out["config"] = str(config_dir())
    except OSError as e:
        out["warning"] = (out["warning"] + "；" if out["warning"] else "") + \
            f"建 config/ 失败：{e}"
    try:
        # 延迟导入：``app.vision.config`` 会反过来 import 本模块，
        # 放在函数体里就不会形成循环。
        from .vision.config import ensure_example, touch_local_template
        ensure_example()                       # 可提交的模板（永远是没有 key 的样子）
        made = touch_local_template()           # 只有"完全不存在"时才铺，绝不覆盖
        if made is not None:
            out["created"] = str(made)
    except Exception as e:                       # noqa: BLE001
        out["warning"] = (out["warning"] + "；" if out["warning"] else "") + \
            f"铺配置模板失败：{type(e).__name__}: {e}"
    return out


def describe() -> dict[str, str]:
    """给启动横幅与 ``/api/health`` 用的一份人话摘要。"""
    d = data_root()
    return {
        "mode": "packaged(exe)" if is_frozen() else "source(开发)",
        "exe_dir": str(exe_dir()),
        "resource_root": str(resource_root()),
        "data_root": str(d),
        "data_root_reason": data_root_reason(),
        "runs_dir": str(runs_dir()),
        "config_dir": str(config_dir()),
    }
