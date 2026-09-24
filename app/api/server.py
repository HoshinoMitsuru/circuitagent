"""后端：导入 -> 解析 -> 人工确认 -> 求解 -> 出报告。

设计取向和这个项目的其它部分一致：**宁可明说"需要人工确认"，不许静默猜**。
所以接口不是"给我一张图，还你一个答案"，而是三段式：

1. ``/api/import``  不管成不成，都把**读图依据**（几何量、跨线、悬空端点、
   被当成本体剔除的线段、跳过的元件）和**每一条待人工确认的判断**一起返回；
2. 前端把这些摆在"结构确认"面板上，人确认/改数值/改类型；
3. ``/api/solve`` 才去算，而且算的时候**三条独立代码路径都跑**，
   任何一条缺失都在报告里显式占一行（不允许把"少跑了一路"当作通过）。

会话只为本机单用户使用，文件落在 ``runs/<sid>/``，IR 保持在内存里；
进程重启会丢会话但不丢原始文件（可以重新导入）。
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from ..ingest.kicad_in import kicad_netlist_text_to_ir, kicad_sch_to_ir
from ..ingest.svg_in import svg_to_ir
from ..ir.model import ALLOWED_KINDS, CONFIDENCE_GATE, Circuit, CircuitError
from ..ir.render import (BODY_HALF, GLYPHS, KIND_LABEL, KIND_UNIT, layout_auto,
                         layout_geom, layout_grid, layout_overlaps,
                         layout_radial, render_overlay_payload, render_svg,
                         style_block)
from ..ir.spice import canonical_netlist_view, from_spice, to_spice
from ..paths import data_root, describe as describe_paths, resource_root
from ..solver.reconcile import format_text_report, run_all

# ★ 资源与数据必须分家（详见 app/paths.py 的模块注释）：
#   - ``WEB_DIR`` 吃的是**打进包里**的只读资源（冻结后是 sys._MEIPASS）；
#   - ``RUNS_DIR`` 是**跑起来才产生**的可写数据（会话、用户上传的原件），
#     冻结后落在 exe 同级，绝不能落进那个"退出即删"的临时解压目录。
#   ``RUNS_DIR`` 保持为**模块级全局**：tests/test_api.py 与若干探针会直接
#   替换它来重定向会话目录，改成函数调用就会把那些重定向悄悄失效。
ROOT = resource_root()
WEB_DIR = ROOT / "web"
DATA_ROOT = data_root()
RUNS_DIR = DATA_ROOT / "runs"

#: 各通道能吃的扩展名
SVG_EXT = {".svg", ".svgz"}
KICAD_SCH_EXT = {".kicad_sch", ".sch"}
NETLIST_EXT = {".net", ".cir", ".sp", ".ckt", ".txt"}
BITMAP_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


# ---------------------------------------------------------------- 会话


class BitmapUnreadable(CircuitError):
    """位图通道没能建出网表。

    ★ 单独一个类型，是因为它和"文件格式不对"完全不是一回事：
    图可能是好的、只是这一层读不出来。而且它**带着诊断包** ——
    导入接口要把两级各自跑到哪、哪些地方没把握一并交回前端，
    不能只说一句"失败了"。这正是本项目「不静默降级」的要求。
    """

    def __init__(self, message: str, *, report: dict[str, Any] | None = None,
                 issues: list[dict[str, Any]] | None = None,
                 outcome: Any = None) -> None:
        super().__init__(message)
        self.report = report or {}
        self.issues = issues or []
        #: 真对象，给叠图用（读不出来的时候**最**需要这张叠图：
        #: 人要靠它看出"到底是哪一步没读出来"）
        self.outcome = outcome


@dataclass
class Session:
    sid: str
    filename: str
    ext: str
    channel: str
    source_path: Path | None = None
    image_size: tuple[int, int] | None = None
    viewbox: tuple[float, float, float, float] | None = None
    ir: Circuit | None = None
    report: dict[str, Any] = field(default_factory=dict)
    #: 位图通道那一路的完整结论（两级各自跑到哪、为什么升级、模型原始返回）
    vision: dict[str, Any] | None = None
    #: ★ 上面那个是**给 JSON 用的字典**；叠图要的是真对象（线段、结点、
    #:   符号框都是 dataclass，不是 dict），所以两个都得留。
    vision_obj: Any = None
    #: ★ 用户在校对表里改过的文字。存**编辑清单**而不是"改完的结果"：
    #:   每次重算都从原始 OCR 重放这份清单，所以改错了再改回去一定回得到原样，
    #:   也不需要撤销栈。见 ``ocr.apply_ocr_edits``。
    vision_ocr_edits: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid, "filename": self.filename, "ext": self.ext,
            "channel": self.channel,
            "image_size": ({"w": self.image_size[0], "h": self.image_size[1]}
                           if self.image_size else None),
            "created": self.created,
        }


_SESSIONS: dict[str, Session] = {}


def _get(sid: str) -> Session:
    s = _SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, f"会话 {sid} 不存在（服务重启会清空会话，请重新导入）")
    return s


# ---------------------------------------------------------------- 导入


def _svg_image_size(path: Path) -> tuple[int, int] | None:
    """只读 SVG 的声明尺寸（不解析几何），给叠图定盒子用。"""
    import re
    head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', head)
    if m:
        try:
            _, _, w, h = (float(v) for v in re.split(r"[\s,]+", m.group(1).strip()))
            return (max(1, int(round(w))), max(1, int(round(h))))
        except ValueError:
            pass
    w = re.search(r'width\s*=\s*"([\d.]+)', head)
    h = re.search(r'height\s*=\s*"([\d.]+)', head)
    if w and h:
        return (max(1, int(float(w.group(1)))), max(1, int(float(h.group(1)))))
    return None


def _bitmap_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return tuple(im.size)  # type: ignore[return-value]
    except Exception:
        return None


def _import_file(sess: Session, *, tol: float, ref_node: str | None,
                 vision_cfg: Any = None) -> None:
    """按通道解析，结果写进会话。**失败也把原因留在 report 里**，不吞异常。"""
    assert sess.source_path is not None
    p = sess.source_path
    if sess.channel == "svg":
        text = p.read_text(encoding="utf-8", errors="replace")
        circ, report = svg_to_ir(text, name=p.stem, tol=tol, ref_node=ref_node)
        sess.image_size = _svg_image_size(p)
    elif sess.channel == "kicad":
        circ = kicad_sch_to_ir(p)
        report = {"component_source": "kicad_netlist",
                  "note": "KiCad 原理图经 kicad-cli 导出网表后解析；"
                          "网表没有极性语义，电压源两端次序需人工确认。"}
    elif sess.channel == "netlist":
        text = p.read_text(encoding="utf-8", errors="replace")
        circ, report = kicad_netlist_text_to_ir(text, name=p.stem), \
            {"component_source": "kicad_netlist"}
    elif sess.channel == "bitmap":
        circ, report = _import_bitmap(sess, cfg=vision_cfg)
    else:
        raise CircuitError(f"不认识的通道 {sess.channel!r}。")
    sess.ir = circ
    sess.report = report
    vb = (report or {}).get("viewbox")
    if vb:
        sess.viewbox = tuple(float(v) for v in vb)  # type: ignore[assignment]


def _import_bitmap(sess: Session, *, cfg: Any = None
                   ) -> tuple[Circuit | None, dict[str, Any]]:
    """位图通道：两级视觉（本地几何+OCR 优先，结构缺失才调模型）。

    ★ **这个函数是阻塞的**（内含阻塞式 OCR 与阻塞式 HTTP），
    调用点必须在线程池里跑，否则会卡住事件循环 —— 这一点在 ``ocr`` /
    ``vlm`` / ``pipeline`` 三个模块的文档里都写着，这里不能再忘。

    返回 ``(IR 或 None, report)``。**IR 为 None 不是异常** ——
    那是"这一层诚实地承认读不出来"，原因全在 report 里。
    """
    from ..vision import pipeline as VP
    from ..vision.config import load_config

    assert sess.source_path is not None
    loaded = load_config()
    vcfg = cfg or loaded.config
    out = VP.run(sess.source_path, cfg=vcfg, name=sess.filename)
    sess.vision_obj = out
    sess.vision = out.to_dict()

    # 叠图要用原图的像素尺寸；位图本来就该在这里量一次
    sess.image_size = _bitmap_image_size(sess.source_path)

    issues = [i.to_dict() for i in out.issues]
    report: dict[str, Any] = {
        "component_source": f"bitmap-{out.tier}",
        "vision": out.to_dict(),
        "note": _bitmap_note(out),
        "config_warnings": loaded.warnings,
        "api_key_present": vcfg.api_key_present,
    }
    if not out.ok:
        # 没建出网表：把"为什么"提炼成一句人话放在 error 位置，
        # 详细原因照旧整包交回去（issues / warnings / 两级各自的结论）。
        why = [i["text"] for i in issues if i.get("severity") == "structural"]
        raise BitmapUnreadable(
            why[0] if why else "本地几何层与视觉模型都没能给出网表。",
            report=report, issues=issues, outcome=out)

    if out.tier == "vlm" and out.vlm is not None:
        report["vlm_note"] = out.vlm.get("note")
    return out.circuit, report


def _bitmap_note(out: Any) -> str:
    """一句话说清"这一遍是谁给的结论"。

    ★ 四种情况必须分开说。早先只分"是不是 vlm"，于是**模型调用失败**时
    报告会写"本地层结构完整、没有调模型" —— 而实际上试过了、失败了，
    用户会以为自己不需要模型、或者以为模型看过了，两种误解都很难纠正。
    """
    vlm = getattr(out, "vlm", None) or {}
    tried = bool(vlm)
    failed = tried and not vlm.get("ok")

    if out.tier == "vlm":
        why = getattr(out, "escalation", "") or ""
        lead = ("按你在界面上的要求，把整张图交给了视觉模型"
                if "手动" in why else
                "本地几何层报出结构性缺口，按「结构缺失才调模型」把整张图交给了视觉模型")
        return (f"位图通道：{lead}，采信其结论（source=vlm，原始返回留档、"
                "叠图核对照跑）。")

    if out.tier == "local":
        soft = len([i for i in out.issues if i.severity == "soft"])
        if failed:
            return ("位图通道：**这次没能调通视觉模型**"
                    f"（{vlm.get('kind') or '未知原因'}），当前结论来自本地几何层"
                    "（端点+圆点定连接、洞+模板定类型、Win 内置 OCR 读位号数值）。"
                    f"本地另有 {soft} 处待人工过目。模型没看成这张图，"
                    "要交模型请先按上面的提示把配置补好。")
        return (f"位图通道：本地几何层（端点+圆点定连接、洞+模板定类型、"
                f"Win 内置 OCR 读位号数值）结构完整，直接出网表，**没有调模型**"
                + (f"；另有 {soft} 处需人工过目的软问题。" if soft else "。"))

    return "位图通道：没有产出。"


def _vision_ocr_edits(sess: Session) -> dict[str, Any]:
    """把用户的文字修改从清单重放一遍，产出"当前生效的那份 OCR 结果"。

    ★ **永远从 ``loc.ocr``（机器原始读数）重放**，绝不把校过的结果写回 ``loc.ocr``。
    这一步是踩过才写死的：早先顺手写了 ``loc.ocr = res``，于是
    "清空编辑清单"时基线已经是被改过的那份，**改错了就再也改不回原样**
    （实测现象：清空 edits 后 22k 没有还原成 10k）。
    现在 ``loc.ocr`` 恒为原始读数、``loc.ocr_effective`` 存生效的那份，
    界面的行号也才能永远指向同一个词。
    """
    from ..vision import ocr as OCR
    from ..vision import pipeline as VP

    out = sess.vision_obj
    loc = getattr(out, "local", None) if out is not None else None
    if loc is None:
        return {"ok": False, "error": "这个会话没有本地层的结果，无从校对。",
                "notes": []}

    res, notes = OCR.apply_ocr_edits(loc.ocr, sess.vision_ocr_edits)
    # 没人改动时留 None，界面据此判断"这几格你是不是动过"
    loc.ocr_effective = res if (notes or sess.vision_ocr_edits) else None

    circ, issues = VP.rebuild_local_ir(loc, res, name=sess.filename)
    loc.circuit = circ
    loc.issues = issues
    # 采信的一层如果是本地，跟着换；如果已经是模型的结论，**不动它** ——
    # 用户改文字是为了把本地那一遍修准，不该顺手把已采信的模型结论换掉。
    if getattr(out, "tier", None) in (None, "local", "none"):
        out.circuit = circ
        out.issues = list(issues)
        out.tier = "local" if circ is not None else "none"
    sess.vision = out.to_dict()
    return {"ok": True, "notes": notes,
            "ocr": res.to_dict() if res else None,
            "effective": loc.ocr_effective is not None,
            "rebuild_issues": [i.to_dict() for i in issues]}


def _vision_apply_to_session(sess: Session) -> None:
    """重算之后，把新网表装回会话（和 ``_import_file`` 装的方式保持一致）。"""
    out = sess.vision_obj
    circ = getattr(out, "circuit", None) if out is not None else None
    sess.ir = circ
    if circ is not None:
        circ.origin = {
            "channel": f"bitmap-{getattr(out, 'tier', 'local')}",
            "vision_tier": getattr(out, "tier", "local"),
            "escalation": getattr(out, "escalation", ""),
            "note": _bitmap_note(out),
        }
    report = dict(sess.report or {})
    report["component_source"] = f"bitmap-{getattr(out, 'tier', 'none')}"
    report["vision"] = out.to_dict() if out is not None else None
    report["note"] = _bitmap_note(out)
    report["ocr_edits"] = list(sess.vision_ocr_edits)
    sess.report = report


def _bitmap_preview(sess: Session) -> dict[str, Any] | None:
    """位图通道的叠图。

    ★ 用**视觉层自己的**叠图，不走 ``render_overlay_payload``：
    后者是照 IR 的**节点坐标**画的，而位图 IR 里没有节点坐标 ——
    连接是几何层从像素里算出来的，节点名只是给结点编的号。
    拿位图 IR 去问它，只会得到一句"没有元件带视觉坐标"。
    位图这一路要核对的本来就是几何量：抽出的线段、判定的结点、检出的圆点、
    符号框、OCR 读到的词，以及**有字却没读出来的块**（读失败时最需要看的就是它）。
    """
    if sess.source_path is None:
        return None
    from ..vision import pipeline as VP

    loc = sess.vision_obj.local if sess.vision_obj is not None else None
    if loc is None or not loc.ran:
        return {"kind": "raster",
                "source_url": f"/api/session/{sess.sid}/source",
                "error": "这一遍没有几何结论可画（本地层没跑起来）。"}
    try:
        pv = VP.overlay_payload(loc, sess.image_size,
                                source_url=f"/api/session/{sess.sid}/source")
    except Exception as e:                         # noqa: BLE001
        return {"kind": "raster",
                "source_url": f"/api/session/{sess.sid}/source",
                "error": f"叠图生成失败：{type(e).__name__}: {e}"}
    pv["kind"] = "raster"
    return pv


def _ir_payload(sess: Session) -> dict[str, Any]:
    """把一个会话整理成前端能直接渲染的一整包。"""
    assert sess.ir is not None
    circ = sess.ir
    warnings: list[str] = []
    try:
        warnings = circ.validate(allow_incomplete=True)
    except CircuitError as e:
        warnings = [f"结构校验未通过：{e}"]

    out: dict[str, Any] = {
        "session": sess.to_dict(),
        "ir": circ.to_dict(),
        "needs_human": circ.unmet_needs(),
        "confidence_gate": CONFIDENCE_GATE,
        "report": sess.report,
        "warnings": warnings,
        "origin_warnings": (circ.origin or {}).get("warnings", []),
        "vision": sess.vision or (sess.report or {}).get("vision"),
        "netlist": None,
        "netlist_view": None,
        "preview": None,
        "figure": None,
    }
    try:
        out["netlist"] = to_spice(circ, title=circ.name)
        out["netlist_view"] = canonical_netlist_view(circ)
    except CircuitError as e:
        out["netlist_error"] = str(e)

    # ---- 叠图：回绘层必须与原图共用视口，否则叠不准
    if sess.channel == "bitmap":
        out["preview"] = _bitmap_preview(sess)
    elif sess.channel == "svg" and sess.source_path is not None:
        size = sess.image_size or (800, 600)
        try:
            out["preview"] = render_overlay_payload(
                circ, size, viewbox=sess.viewbox)
            out["preview"]["source_url"] = f"/api/session/{sess.sid}/source"
            out["preview"]["kind"] = "svg"
        except CircuitError as e:
            out["preview"] = {"error": str(e)}
    elif sess.image_size:
        try:
            out["preview"] = render_overlay_payload(circ, sess.image_size)
            out["preview"]["source_url"] = f"/api/session/{sess.sid}/source"
            out["preview"]["kind"] = "raster"
        except CircuitError as e:
            out["preview"] = {"error": str(e)}

    # ---- 报告插图（自动版式 + 版面自检）
    try:
        dg: dict = {}
        svg = render_svg(circ, mode="auto", diagnostics=dg)
        out["figure"] = {"svg": svg, "layout": dg["layout"],
                         "overlaps": dg["overlaps"],
                         "roundtrip_safe": dg["roundtrip_safe"]}
    except CircuitError as e:
        out["figure"] = {"error": str(e)}
    return out


# ---------------------------------------------------------------- 应用


app = FastAPI(title="电路读图与三法对账", version="0.1.0")


@app.get("/")
def index() -> FileResponse:
    idx = WEB_DIR / "index.html"
    if not idx.is_file():
        raise HTTPException(500, f"前端文件缺失：{idx}")
    return FileResponse(idx, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    from ..solver.ngspice import probe_availability
    from ..vision import pipeline as VP

    vis = VP.probe()
    return {
        "ok": True,
        "ngspice": probe_availability(),
        "confidence_gate": CONFIDENCE_GATE,
        # 位图通道的可用性由视觉层自己答（OCR 装没装、模型配没配），
        # 不再是一个写死的 False。
        "bitmap_channel": bool(vis["enabled"]),
        "vision": vis,
        "web_dir": str(WEB_DIR),
        # ★ 路径信息一起给出去，界面上要能看见"我的文件到底存在哪"。
        #   打包成 exe 之后，数据目录**可能**因为 exe 同级不可写而落到用户目录 ——
        #   这件事必须能被用户与截图诊断看出，不然就是"配置写在哪没人知道"。
        "paths": describe_paths(),
    }


# ---------------------------------------------------------------- 视觉设置


@app.get("/api/vision/config")
def api_vision_config() -> dict[str, Any]:
    """读视觉通道配置。**只回掩码，绝不明文回 key。**

    顺带把 ``probe()`` 一起给出：前端拿一个响应就能把"当前能不能用、
    走哪一路、还缺什么"整屏画出来，不用再打第二个请求。
    """
    from ..vision import ocr as OCR
    from ..vision import pipeline as VP
    from ..vision.config import ensure_example, load_config

    ensure_example()
    loaded = load_config()
    try:
        langs = OCR.available_languages()
    except Exception:                          # noqa: BLE001
        langs = []                             # OCR 组件没装就当空表（已是预期分支）
    return {
        "ok": True,
        "config": loaded.config.to_dict(mask=True),
        "warnings": loaded.warnings,
        "sources": loaded.sources,
        "probe": VP.probe(loaded.config),
        "ocr_langs_installed": langs,
    }


@app.post("/api/vision/config")
def api_vision_config_save(payload: dict[str, Any]) -> JSONResponse:
    """写入视觉设置，**写盘后重新读盘再回** —— 前端看到的必须是盘上真生效的值。

    明文 key 的写入受 ``save_config`` 保护：空串 / 掩码 / 纯星号都表示
    "不动原值"，防止前端把掩码回填把 key 悄悄清掉。
    """
    from ..vision import pipeline as VP
    from ..vision.config import save_config

    if not isinstance(payload, dict):
        return JSONResponse(status_code=200, content={
            "ok": False, "error": "请求体应该是一个 JSON 对象。"})
    try:
        loaded = save_config(payload.get("config") or payload)
    except Exception as e:                     # noqa: BLE001
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"写配置失败：{type(e).__name__}: {e}"})
    return JSONResponse(content={
        "ok": True,
        "config": loaded.config.to_dict(mask=True),
        "warnings": loaded.warnings,
        "sources": loaded.sources,
        "probe": VP.probe(loaded.config),
    })


@app.post("/api/vision/ocr")
async def api_vision_ocr(payload: dict[str, Any]) -> JSONResponse:
    """★ 校对表提交：把人改过的文字重放一遍，重算网表。

    这是"**先审后算**"的落点 —— 之前 OCR 一出结果就直接进网表，
    用户看见了 ``RI`` 也只能去结构确认表里改元件位号（而"有字没读出来"的块
    根本不在元件表里，压根没地方填）。现在文字层本身可改，改完点一下重算。

    请求体：``{"sid": ..., "edits": [{"op":"set","i":3,"text":"R1"}, ...]}``
    ``edits`` 是**完整清单**，不是增量 —— 前端每次都把当前表格整份交上来，
    服务端从原始 OCR 重放。这样"改错了再改回去"是真的回得去。

    这个端点**不调视觉模型**（毫秒级，可以随便点）。要不要交给模型是另一个按钮的事。
    """
    sid = str(payload.get("sid") or "").strip()
    sess = _get(sid)
    if sess.channel != "bitmap":
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"这个通道（{sess.channel}）没有 OCR 校对表。"})
    if sess.vision_obj is None:
        return JSONResponse(status_code=200, content={
            "ok": False, "error": "这个会话没有视觉层的结果（服务可能重启过），请重新导入。"})

    raw = payload.get("edits")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        return JSONResponse(status_code=200, content={
            "ok": False, "error": "edits 必须是数组。"})
    sess.vision_ocr_edits = [e for e in raw if isinstance(e, dict)]

    try:
        res = await run_in_threadpool(_vision_ocr_edits, sess)
    except Exception as e:                         # noqa: BLE001
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"重算网表时出错：{type(e).__name__}: {e}"})
    if not res.get("ok"):
        return JSONResponse(status_code=200, content={
            "ok": False, "error": res.get("error"), "notes": res.get("notes")})

    _vision_apply_to_session(sess)
    out = sess.vision_obj
    body: dict[str, Any] = {
        "ok": True,
        "channel": sess.channel,
        "session": sess.to_dict(),
        "vision": sess.vision,
        "notes": res.get("notes") or [],
        "rebuild_issues": res.get("rebuild_issues") or [],
        "escalation": getattr(out, "escalation", ""),
        "tier": getattr(out, "tier", "none"),
        "preview": _bitmap_preview(sess),
    }
    if sess.ir is None:
        # 改完之后反而建不出网表了（比如把唯一能定拓扑的文字改没了）。
        # **不删会话**：让用户看着自己的改动继续修。
        body["ok"] = False
        body["error"] = ("按你改过的文字重算之后，本地层建不出网表了。"
                         "上面的文字校对表可以继续改，或交给视觉模型。")
        return JSONResponse(status_code=200, content=body)
    body.update(_ir_payload(sess))
    body["ok"] = True
    return JSONResponse(content=body)


@app.post("/api/vision/escalate")
async def api_vision_escalate(payload: dict[str, Any]) -> JSONResponse:
    """★ 「交给视觉模型识别」按钮：**由用户决定**要不要花这次调用。

    默认档位 ``escalate_on="manual"`` 下，导入时本地跑完就停在这里，
    把"为什么建议/不建议升级"连同本地读到的文字一起摆给用户看。
    这个端点就是那个按钮 —— 按下去才真的调模型。

    ★ 本地那一遍**不重跑**（``escalate_now``）：用户很可能刚在校对表里改过文字，
    重跑会把他改的东西冲掉。就地补上第二级。
    """
    sid = str(payload.get("sid") or "").strip()
    sess = _get(sid)
    if sess.channel != "bitmap":
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"这个通道（{sess.channel}）没有视觉模型可选。"})
    if sess.source_path is None or sess.vision_obj is None:
        return JSONResponse(status_code=200, content={
            "ok": False, "error": "这个会话没有可升级的本地结果（服务可能重启过），请重新导入。"})

    from ..vision import pipeline as VP
    from ..vision.config import load_config

    loaded = load_config()
    out = sess.vision_obj

    def _run() -> None:
        VP.escalate_now(out, sess.source_path, cfg=loaded.config,
                        name=sess.filename)

    try:
        await run_in_threadpool(_run)          # 阻塞式 HTTP，必须丢线程池
    except Exception as e:                     # noqa: BLE001
        return JSONResponse(status_code=200, content={
            "ok": False,
            "error": f"调用视觉模型时出错：{type(e).__name__}: {e}",
            "vision": sess.vision,
            "escalation": getattr(out, "escalation", ""),
        })

    sess.vision = out.to_dict()
    _vision_apply_to_session(sess)
    vlm = out.vlm or {}
    body: dict[str, Any] = {
        "ok": sess.ir is not None,
        "channel": sess.channel,
        "session": sess.to_dict(),
        "vision": sess.vision,
        "tier": getattr(out, "tier", "none"),
        "escalation": getattr(out, "escalation", ""),
        # ★ 光看 ok 分不清"模型到底调通没有"：本地网表在模型失败时也还在，
        #   ok 会是 true。用户按的是"交给模型"，就必须明确回答这一件事。
        "vlm_ok": bool(vlm.get("ok")),
        "vlm_reason": (vlm.get("message") or "") if vlm and not vlm.get("ok") else "",
        "vlm_elapsed_s": vlm.get("elapsed_s"),
    }
    if sess.ir is None:
        why = [i.text for i in out.issues if i.severity == "structural"]
        body["error"] = why[0] if why else "交给模型之后仍然没能拿到网表。"
        body["issues"] = [i.to_dict() for i in out.issues]
        return JSONResponse(status_code=200, content=body)
    body.update(_ir_payload(sess))
    body["ok"] = True
    return JSONResponse(content=body)


@app.post("/api/import")
async def api_import(
    file: UploadFile = File(...),
    tol: float = Form(6.0),
    ref_node: str | None = Form(None),
) -> JSONResponse:
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext in SVG_EXT:
        channel = "svg"
    elif ext in KICAD_SCH_EXT:
        channel = "kicad"
    elif ext in NETLIST_EXT:
        channel = "netlist"
    elif ext in BITMAP_EXT:
        channel = "bitmap"
    else:
        raise HTTPException(400, f"不认识的扩展名 {ext!r}。"
                                 "支持 SVG / .kicad_sch / 网表(.net .cir .txt) / 位图(暂未实现)")

    sid = uuid.uuid4().hex[:12]
    d = RUNS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"source{ext}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    sess = Session(sid=sid, filename=name, ext=ext, channel=channel,
                   source_path=dest)
    if channel == "bitmap":
        sess.image_size = _bitmap_image_size(dest)
    _SESSIONS[sid] = sess

    try:
        if channel == "bitmap":
            # ★ 必须丢线程池：OCR 与 VLM 都是阻塞调用，直接 await 会把
            #   事件循环按住，整个界面在这一次导入期间失去响应。
            await run_in_threadpool(_import_file, sess, tol=tol,
                                    ref_node=(ref_node or None))
        else:
            _import_file(sess, tol=tol, ref_node=(ref_node or None))
    except BitmapUnreadable as e:
        # ★ 位图读不出来**保留会话**：文件已落盘、诊断包有内容，
        #   前端要靠它把"哪个环节没读出什么"摆出来给人看。
        sess.report = dict(e.report)
        sess.vision = (e.report or {}).get("vision")
        sess.vision_obj = e.outcome
        return JSONResponse(status_code=200, content={
            "ok": False, "error": str(e), "issues": e.issues,
            "vision": sess.vision, "report": sess.report,
            "session": sess.to_dict(), "channel": channel,
            "preview": _bitmap_preview(sess),
        })
    except CircuitError as e:
        # 解析失败不是 500：把原因和已落盘的文件一起交回去，前端好显示
        _SESSIONS.pop(sid, None)
        return JSONResponse(status_code=200, content={
            "ok": False, "error": str(e), "session": sess.to_dict(),
            "channel": channel,
        })
    except Exception as e:                     # pragma: no cover
        _SESSIONS.pop(sid, None)
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"{type(e).__name__}: {e}",
            "session": sess.to_dict(), "channel": channel,
        })

    payload = _ir_payload(sess)
    payload["ok"] = True
    return JSONResponse(content=payload)


@app.get("/api/session/{sid}")
def api_session(sid: str) -> dict[str, Any]:
    sess = _get(sid)
    if sess.ir is None:
        raise HTTPException(400, "这个会话还没有解析出电路")
    return _ir_payload(sess)


@app.get("/api/session/{sid}/source")
def api_source(sid: str) -> FileResponse:
    sess = _get(sid)
    assert sess.source_path is not None
    media = {
        ".svg": "image/svg+xml", ".svgz": "image/svg+xml",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".bmp": "image/bmp", ".webp": "image/webp", ".gif": "image/gif",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }.get(sess.ext, "application/octet-stream")
    return FileResponse(sess.source_path, media_type=media)


@app.post("/api/ir")
async def api_put_ir(payload: dict[str, Any]) -> dict[str, Any]:
    """接收前端改过的 IR（人工确认闸门 / IR 编辑器的结果），存回会话。

    这里**不校验就拒绝**：缺数值、类型不对都可能只是编辑中的中间状态，
    前端要靠 ``warnings`` 提示。真正拦死是在 ``/api/solve``。
    """
    sid = payload.get("sid") or ""
    sess = _get(sid)
    try:
        sess.ir = Circuit.from_dict(payload["ir"])
    except Exception as e:
        raise HTTPException(400, f"IR 无法解析：{type(e).__name__}: {e}") from None
    body = _ir_payload(sess)
    body["ok"] = True
    return body


@app.post("/api/solve")
async def api_solve(payload: dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid") or ""
    sess = _get(sid)
    if "ir" in payload:
        try:
            sess.ir = Circuit.from_dict(payload["ir"])
        except Exception as e:
            return JSONResponse(status_code=200, content={
                "ok": False, "error": f"IR 无法解析：{type(e).__name__}: {e}"})
    assert sess.ir is not None
    try:
        pack = run_all(sess.ir, reduce_dc=bool(payload.get("reduce_dc", True)))
    except CircuitError as e:
        # ★ 约束矛盾（CircuitUnsatisfiable）也走这里：它是 CircuitError 的子类。
        #   这不是 500 —— 输入没问题，只是这张理想化电路本身无解，
        #   要把原因原样交给用户，而不是抛一个栈。
        return JSONResponse(status_code=200, content={
            "ok": False, "stage": "solve", "error": str(e)})
    except Exception as e:                     # pragma: no cover
        return JSONResponse(status_code=200, content={
            "ok": False, "stage": "solve",
            "error": f"{type(e).__name__}: {e}"})

    # ---- 文本报告是**派生视图**，不是解本身。
    # 它曾经因为用原电路的节点/位号去查化简后的对账表而抛 StopIteration，
    # 而这一句当时不在任何 try 里 → 整个 /api/solve 变 HTTP 500，
    # 连已经算好的 pack 都一并丢掉。现在：结构化结果照常返回，
    # 渲染失败**显式留痕**（不静默、也不是"没有报告就是没问题"）。
    text_report: str | None = None
    text_report_error: str | None = None
    try:
        text_report = format_text_report(sess.ir, pack)
    except Exception as e:                     # noqa: BLE001
        text_report_error = f"{type(e).__name__}: {e}"
        print(f"[circuit_agent] 文本报告渲染失败（结构化结果不受影响）：{text_report_error}",
              file=sys.stderr)

    return JSONResponse(content={
        "ok": True,
        "pack": pack,
        "text_report": text_report,
        "text_report_error": text_report_error,
        "ir": sess.ir.to_dict(),
        "unmet": sess.ir.unmet_needs(),
    })


@app.post("/api/render")
def api_render(payload: dict[str, Any]) -> dict[str, Any]:
    sid = payload.get("sid") or ""
    sess = _get(sid)
    circ = sess.ir
    if "ir" in payload:
        try:
            circ = Circuit.from_dict(payload["ir"])
        except Exception as e:
            raise HTTPException(400, f"IR 无法解析：{e}") from None
    if circ is None:
        raise HTTPException(400, "这个会话还没有电路")
    mode = str(payload.get("mode", "auto"))
    want_viewbox = bool(payload.get("match_source", False))
    vb = sess.viewbox if (want_viewbox and sess.viewbox) else None
    dg: dict = {}
    try:
        svg = render_svg(circ, mode=mode, viewbox=vb, dark=bool(payload.get("dark")),
                         diagnostics=dg)
    except CircuitError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "svg": svg, **dg}


@app.post("/api/from-spice")
def api_from_spice(payload: dict[str, Any]) -> JSONResponse:
    """手工贴一段 SPICE 网表也能直接进流程 —— 这条路径最省事也最不容易出错。"""
    text = str(payload.get("text") or "")
    if not text.strip():
        return JSONResponse(status_code=200, content={"ok": False, "error": "网表是空的"})
    try:
        circ = from_spice(text, name=str(payload.get("name") or "手工网表"))
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"{type(e).__name__}: {e}"})
    sid = uuid.uuid4().hex[:12]
    d = RUNS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.cir").write_text(text, encoding="utf-8")
    sess = Session(sid=sid, filename="手工网表", ext=".cir", channel="netlist",
                   source_path=d / "source.cir")
    sess.ir = circ
    sess.report = {"component_source": "spice_text",
                   "note": "来自手工粘贴的 SPICE 网表，拓扑与取值按字面采信（source=exact）。"}
    _SESSIONS[sid] = sess
    body = _ir_payload(sess)
    body["ok"] = True
    return JSONResponse(content=body)


@app.post("/api/layout")
def api_layout(payload: dict[str, Any]) -> JSONResponse:
    """给出**结点坐标**，供手绘面板把已有电路打回画布继续编辑。

    ★ 为什么不把回绘 SVG 解析回画布、而是直接要坐标：
    IR 里**根本没有"导线"这个对象** —— 同名节点就代表它们连在一起，
    回绘时一个节点只对应一个点。所以把 IR 放回画布，
    只要知道"每个节点画在哪儿"就够了：两个元件共用节点 ``1``，
    它们的端点会落在**同一个点**上，天然重合，不需要画任何导线。
    绕一圈去解析 SVG 反而要猜"这条线是元件的引线还是用户画的导线"。

    也能反过来说明手绘面板里"画导线"到底在干什么：把两个**不同的点**
    连起来，让解析器把它们归成同一个结点。
    """
    sess = _get(str(payload.get("sid") or ""))
    if sess.ir is None:
        return JSONResponse(content={"ok": False, "error": "这个会话还没有电路。"})
    circ = sess.ir
    mode = str(payload.get("mode") or "auto")
    has_geom = bool(circ.components) and all(
        c.geom.get("p1") and c.geom.get("p2") for c in circ.components)
    try:
        if mode == "geom":
            if not has_geom:
                return JSONResponse(content={
                    "ok": False,
                    "error": "这份电路没有视觉坐标（来自网表/KiCad 通道），"
                             "无法按原图版式回画布。请改用 grid 或 radial 版式。"})
            pos, resolved = layout_geom(circ), "geom"
        elif mode == "radial":
            pos, resolved = layout_radial(circ), "radial"
        elif mode == "grid":
            pos, resolved = layout_grid(circ), "grid"
        elif mode == "auto":
            if has_geom:
                pos, resolved = layout_geom(circ), "geom"
            else:
                pos, resolved = layout_auto(circ)
        else:
            return JSONResponse(content={"ok": False,
                                         "error": f"不认识的版式 {mode!r}。"})
    except CircuitError as e:
        return JSONResponse(content={"ok": False, "error": str(e)})
    if not pos:
        return JSONResponse(content={"ok": False, "error": "算不出任何结点坐标。"})
    overlaps = layout_overlaps(circ, pos)
    return JSONResponse(content={
        "ok": True, "layout": resolved,
        "node_pos": {n: [round(float(x), 3), round(float(y), 3)]
                     for n, (x, y) in pos.items()},
        "overlaps": overlaps,
        "roundtrip_safe": not overlaps,
        # ★ 自动版式会重叠这件事必须一起说。重叠意味着"按这个坐标画出来的图
        #   几何回导会判错"（长支路导线穿过无关结点），此时**不能**拿它去改。
        "hint": ("这个版式有重叠，只能当参考图看、不要直接拿去几何回导"
                 if overlaps else ""),
    })


@app.get("/api/symbols")
def api_symbols() -> JSONResponse:
    """手绘面板要用的**元件定义**：图形 + 本体半长 + 中文名 + 单位 + 共用 CSS。

    ★ 为什么让后端给，而不是在 JS 里照着 ``render.py`` 重抄一份：
    重抄必然漂移。符号一改，画布上画出来的就和渲染器画出来的不是一个东西 ——
    而手绘面板的全部意义就是「画出来的 = 能算的」。
    这里直接吃 ``render.GLYPHS`` / ``render.BODY_HALF``：
    **后端加一种元件，前端的调色板自动多一个，一行前端代码都不用改。**
    这正是"对元件具有高度兼容性"的落地方式。

    ``body_half`` 必须一起给：画布要按它把导线停在**本体边缘**，
    不能画到中心 —— 画到中心就等于导线钻进符号里，回导时几何会判错。
    """
    kinds = [k for k in ALLOWED_KINDS if k in GLYPHS]
    symbols = [{
        "kind": k,
        "label": KIND_LABEL.get(k, k),
        "unit": KIND_UNIT.get(k, ""),
        "glyph": GLYPHS[k](),
        "body_half": BODY_HALF.get(k, 15.0),
    } for k in sorted(kinds)]
    return JSONResponse(content={
        "ok": True,
        "style": style_block(False),      # 与渲染器**同一份** CSS
        "symbols": symbols,
        #: 画布栅格步长（px）。取 20 是因为元件本体半长在 12~19 之间，
        #: 太小拖不准、太大摆不下邻近元件。前端可以覆盖。
        "grid": 20,
        "ref_node": "0",
    })


@app.post("/api/from-svg")
def api_from_svg(payload: dict[str, Any]) -> JSONResponse:
    """手绘面板提交的 SVG 进流程。**走的是与导入 SVG 文件完全相同的解析器。**

    ★ 这是本面板「几何即权威」的落点：画布只负责产出图形，网表由
    ``svg_to_ir`` 用几何反推 —— 和把一张照片/示意图喂进来是同一套判定，
    不另立一套规矩。所以"在画布上看着连上了、网表里却没连上"这种事，
    一旦发生就是**解析器的问题**，两侧不会各有一套说法。

    画布产出的 SVG 带 ``data-ca-*`` 语义标记（与 ``render_svg`` 同构），
    解析器优先用标记定身份与取值、用几何定位 —— 于是往返无损。

    ``sid`` 给了就**接着那个会话改**（保留它的来源与报告，另存一份"手绘"痕迹）；
    不给就新建一个会话。
    """
    svg = str(payload.get("svg") or "")
    if not svg.strip():
        return JSONResponse(status_code=200,
                            content={"ok": False, "error": "画布是空的，没有可解析的内容。"})
    if "<svg" not in svg:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "error": "提交的内容里没有 <svg> 根元素 —— 这不该发生，"
                     "说明画布的导出坏了，请把这段反馈给开发者。"})
    tol = float(payload.get("tol") or 6.0)
    ref_node = payload.get("ref_node") or None
    name = str(payload.get("name") or "手绘电路图")

    old_sid = str(payload.get("sid") or "").strip()
    sess = _SESSIONS.get(old_sid) if old_sid else None
    if old_sid and sess is None:
        # ★ 不静默新建：用户以为在改原来那张图，结果拿到一个新会话，
        #   而界面看起来"改成功了" —— 那是比报错坏得多的情况。
        raise HTTPException(404, f"会话 {old_sid} 不存在（服务重启会清空会话），"
                                 "请重新导入，或取消勾选「接着当前会话改」。")

    try:
        circ, report = svg_to_ir(svg, name=name, tol=tol, ref_node=ref_node)
    except CircuitError as e:
        # 解析失败不是 500：画布还在用户手里，他改完能立刻重交
        return JSONResponse(status_code=200, content={
            "ok": False, "error": str(e), "channel": "svg",
            "issues": [{"code": "svg_parse_failed", "severity": "structural",
                        "text": str(e)}],
        })
    except Exception as e:                          # noqa: BLE001
        return JSONResponse(status_code=200, content={
            "ok": False, "error": f"解析手绘 SVG 时出错：{type(e).__name__}: {e}",
            "channel": "svg"})

    report = dict(report)
    # 手绘的语义要说清楚：**人画的线就是这个电路的权威**，不是"机器读出来的"。
    # 但仍照跑三法互校与功率守恒 —— 画错了（元件被短接、图不闭合）照样能被算出来。
    report["component_source"] = "hand_drawn"
    report["note"] = (
        "来自界面上的手绘画布。连接由你画出的几何反推（与导入 SVG 走同一个解析器），"
        "元件身份与取值取自画布写入的语义标记（data-ca-*）。"
        "这是人给的，不是机器读出来的 —— 三法互校与功率守恒照跑，"
        "画错了照样会算出来，所以结论仍需你自己核一眼。")

    if sess is not None:
        # ---- 接着当前会话改：**沿用同一个 sid**
        # ★ 为什么不新开一个 sid：用户的心智是"我在改这张图"。换 id 会留下
        #   指向同一个 Session 对象的旧键（两个 sid 描述同一份状态，越用越乱），
        #   前端还得跟着换 id，任何一处漏了就是"改完看不到变化"。
        d = (sess.source_path.parent if sess.source_path
             else (RUNS_DIR / sess.sid))
        d.mkdir(parents=True, exist_ok=True)
        # ★ 绝不覆盖 ``source.*`` —— 那是用户上传的原件（照片 / 交上来的 SVG）。
        #   手绘版另存一份，并在报告里写清原件在哪，两边都留得住。
        hand = d / "hand.svg"
        hand.write_text(svg, encoding="utf-8")
        # 视口与像素尺寸都要跟着换成手绘这份的 —— 留着位图那份尺寸的话，
        # 叠图会按原照片的坐标系去摆手绘图的坐标，画出来是错位的。
        sess.viewbox = tuple(float(v) for v in report["viewbox"]) \
            if report.get("viewbox") else None
        sess.image_size = _svg_image_size(hand)
        prev = (sess.report or {}).get("note") if isinstance(sess.report, dict) else None
        prev_he = (sess.report or {}).get("hand_edited") \
            if isinstance(sess.report, dict) else None
        # ★ 在手绘面板里改第二次时，"原件"不能被上一次的手绘版顶掉。
        #   这时的 ``sess.source_path`` 已经指向 ``hand.svg`` 了 —— 直接取它会得到
        #   "这份图的原件是 hand.svg 自己"，来源链就断在第一环上。
        #   所以：已经手绘过就从上一份报告里**继承**最初的原件名与最初通道。
        orig_name = (prev_he or {}).get("original_source") if prev_he else \
            (sess.source_path.name if sess.source_path else None)
        orig_ch = (prev_he or {}).get("origin_channel") if prev_he else sess.channel
        report["hand_edited"] = {
            "name": name,
            "original_source": orig_name,
            "origin_channel": orig_ch,
            "prev_channel": sess.channel,
            "prev_note": prev,
            "hand_svg": "hand.svg",
            "edit_count": int((prev_he or {}).get("edit_count") or 0) + 1,
        }
        sess.ir = circ
        sess.report = report
        sess.source_path = hand
        sess.channel = "svg"
        sess.ext = ".svg"
        # 会话换成 SVG 通道了：位图那一路的视觉结论/校对清单不再对应这份 IR，
        # 必须清掉。留着的话前端会拿旧的 OCR 校对表去改新网表 —— 改的是别的东西。
        sess.vision = None
        sess.vision_obj = None
        sess.vision_ocr_edits = []
        body = _ir_payload(sess)
        body["ok"] = True
        body["hand_edited"] = True
        return JSONResponse(content=body)

    sid = uuid.uuid4().hex[:12]
    d = RUNS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.svg").write_text(svg, encoding="utf-8")
    sess = Session(sid=sid, filename=name, ext=".svg", channel="svg",
                   source_path=d / "source.svg")
    sess.ir = circ
    sess.report = report
    if report.get("viewbox"):
        sess.viewbox = tuple(float(v) for v in report["viewbox"])
    sess.image_size = _svg_image_size(d / "source.svg")
    _SESSIONS[sid] = sess
    body = _ir_payload(sess)
    body["ok"] = True
    return JSONResponse(content=body)


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    import uvicorn
    uvicorn.run("app.api.server:app" if reload else app,
                host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":                     # pragma: no cover
    h = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    print(f"[circuit_agent] WebUI: http://{h}:{p}/")
    serve(h, p)
