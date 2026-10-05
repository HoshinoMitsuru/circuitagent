"""API 层回归测试：7 个端点里最容易出事的那两个（/api/from-spice、/api/solve）。

★ **为什么必须有这个文件**：在这之前 API 层**零自动化测试** ——
只有 09-20 那一次人工 curl，而且当时用的都是纯 R/V 电路。
于是漏掉了一整类：**含 C/L 的电路**。实测症状是

    POST /api/solve  （含电感）→ HTTP 500 Internal Server Error
    POST /api/solve  （纯 R/V）→ HTTP 200

根因是 `format_text_report` 用**原电路**的节点名/位号去查**化简后**电路生成的
对账表，被电感合并掉的节点、被电容移除的支路在表里都不存在 → `StopIteration`；
而那一句当时不在任何 `try` 里，于是整个解连同已经算好的 pack 一起被丢掉。

这个文件就是钉住那两件事：
1. 含 C/L 的电路求解**不得**是 5xx；
2. 无解电路要返回**可读的原因**（`ok=false` + 说明），而不是 5xx 或一个"通过"。

运行：
    C:\\Users\\Psyche\\.workbuddy\\binaries\\python\\envs\\circuit_agent\\Scripts\\python.exe tests\\test_api.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient                                 # noqa: E402

from app.api import server as srv                                         # noqa: E402
from app.ir.model import ALLOWED_KINDS, BITMAP_KINDS, CONTROLLED_KINDS    # noqa: E402
from app.ir.render import BODY_HALF, GLYPHS                               # noqa: E402
from app.vision import config as vconf                                    # noqa: E402

FAILS = 0

NET_PURE_RV = """纯阻网络
V1 1 0 DC 12
R1 1 2 100
R2 2 0 200
.end
"""

NET_WITH_L = """电感串联
V1 1 0 DC 10
R1 1 2 5
L1 2 3 1m
R2 3 0 5
.end
"""

NET_UNSOLVABLE = """电压源被电感短路
V1 1 0 DC 10
L1 1 0 1m
R1 1 2 5
R2 2 0 5
.end
"""

# 电流控制型受控源 + 系统插入的 0V 探针 + 一行语义标记。
# ★ 这一份是本项目**自己导出**的网表该有的样子（to_spice 的输出格式）：
#   探针不是题目元件，题目里说的是 R1；只有那行 `* ca-ctrl` 能把两者分开。
NET_CTRL = """受控源（含 0V 探针与语义标记）
V1 1 0 DC 10
R1 1 ns_R1 1000
R2 2 0 1000
* ca-ctrl H1 mode=I ref=R1 sense=Vsense_R1
H1 3 0 Vsense_R1 -2000
R3 3 0 1000
Vsense_R1 2 ns_R1 DC 0
.op
.end
"""

NET_VCVS = """压控压源
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 2k
E1 3 0 2 0 3
R3 3 0 1k
.end
"""


def check(label: str, cond: bool, detail: str = "") -> None:
    global FAILS
    if not cond:
        FAILS += 1
    print(f"  [{'通过' if cond else '失败'}] {label}" + (f"  —— {detail}" if detail else ""))


def eq(label, got, want) -> None:
    global FAILS
    ok = got == want
    if not ok:
        FAILS += 1
    print(f"  [{'通过' if ok else '失败'}] {label}: 期望 {want}  实得 {got}")


def banner(t: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {t}")
    print("#" * 72)


def _font(size: int):
    from PIL import ImageFont
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:                          # noqa: BLE001
            continue
    return ImageFont.load_default()


def _tiny_png(with_short_text: bool = False):
    """一张闭合串联回路的位图，带一行标注。

    两个变体，各测一条路：

    - ``with_short_text=False``（默认）：标注写成 ``"R1 10k"``。
      ★ 为什么位号和数值要写在**同一行**：Windows 内置 OCR 对**短文本行**会整条跳过。
    - ``with_short_text=True``：只写位号 ``"R1"``，不写数值。
      ★ 这个形状是**实测挑出来的**，用来测"有字但没读出来"的补填入口：
      ``runs/_probe_refdes_scan.py`` 把 ``R1..R9``／``C1``／``V1`` 等逐个单独渲染后送进
      OCR，结果是 ``R1 C1 V1`` 在 16/20/26/32/40px **每一档都被整条吞掉**，
      ``R1`` 尤其稳定（五个字号全丢），而 ``R3 C3 R5`` 之类却认得到 ——
      所以这一档必须**钉死在 R1**，不能随手换个位号就完事。
      附带结论（重要）：教科书里最常见的 ``R1``／``C1``／``V1`` 恰好都在丢失集里，
      所以"位号读不出来"在真实照片上是**常态**，不是边缘情况。

    标注位置 ``(230,150)`` 也是量出来的：元件槽的搜索半径是
    ``max(20, max(w,h)*1.6)``，这张图的电阻槽 60×40 → 半径 96px，
    槽心在 ``(300,120)``。只写 ``"R1"`` 时字比 ``"R1 10k"`` 窄，
    画在 ``(200,145)`` 中心离槽心约 95.6px，**正好卡在 96px 的边界上**；
    挪到 ``(230,150)`` 后约 70px，留足余量。
    """
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (620, 480), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([270, 100, 330, 140], outline="black", width=4)
    d.line([120, 120, 270, 120], fill="black", width=4)
    d.line([330, 120, 480, 120], fill="black", width=4)
    d.line([480, 120, 480, 360], fill="black", width=4)
    d.line([480, 360, 120, 360], fill="black", width=4)
    d.ellipse([95, 215, 145, 265], outline="black", width=4)
    d.line([120, 120, 120, 215], fill="black", width=4)
    d.line([120, 265, 120, 360], fill="black", width=4)
    if with_short_text:
        d.text((230, 150), "R1", fill="black", font=_font(20))
    else:
        d.text((200, 145), "R1 10k", fill="black", font=_font(20))
    return im


def _blank_png():
    """一张白纸：本地层必然一条线都抽不出来 —— 用来测"读不出来"这条路。"""
    from PIL import Image
    return Image.new("RGB", (420, 320), "white")


def _canvas_svg(comps, wires, box=(0, 0, 600, 500)):
    """按手绘面板 `dSubmitSvg()` 的格式造一份 SVG。

    ★ 为什么要在这里等价实现一遍：画布产出的 SVG 是**浏览器里拼出来的**，
      而本机没有可用的无头浏览器（见 MEMORY：Edge 的 `--screenshot` /
      `--dump-dom` 都取不到输出），没法自动化验收。所以把这份格式钉在这里 ——
      前端那边改坏了格式，这组断言就会红。

    ★ 刻意**不**写 ``data-ca-a`` / ``data-ca-b``：那两个属性是**节点名的强提示**，
      解析器见到就把对应结点改叫那个名字。回绘 SVG 写它们是对的（它有确切的
      节点名要还原），但手绘图没有节点名这回事 —— 名字正是要**算出来**的东西。
      写上去等于把节点名硬编码，用户画的接法反而作废。

    ``comps`` 每项：``{ref, kind, a:(x,y), b:(x,y), value?}``；``wires`` 每项一个端点对。
    """
    x, y, w, h = box
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}" '
           f'width="{w}" height="{h}" class="circuit-svg" data-ca-refnode="0">',
           f'<rect class="ca-bg" x="{x}" y="{y}" width="{w}" height="{h}" '
           'fill="#ffffff" stroke="none"/>']
    for a, b in wires:
        out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke-width="2"/>')
    for c in comps:
        a, b = c["a"], c["b"]
        bh = BODY_HALF.get(c["kind"], 15.0)
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ae = (mx - bh * ux, my - bh * uy)
        bb = (mx + bh * ux, my + bh * uy)
        out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{ae[0]:.3f}" y2="{ae[1]:.3f}" stroke-width="2"/>')
        out.append(f'<line x1="{bb[0]:.3f}" y1="{bb[1]:.3f}" x2="{b[0]}" y2="{b[1]}" stroke-width="2"/>')
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        val = f' data-ca-value="{c["value"]}"' if c.get("value") else ""
        out.append(f'<g class="glyph ca-component" '
                   f'transform="translate({mx:.3f},{my:.3f}) rotate({ang:.2f})" '
                   f'data-ca-ref="{c["ref"]}" data-ca-kind="{c["kind"]}" '
                   f'data-ca-p1="{a[0]},{a[1]}" data-ca-p2="{b[0]},{b[1]}"{val}>')
        out.append(GLYPHS[c["kind"]]())            # ★ 与画布同一个来源
        out.append(f'<g transform="rotate({-ang:.2f})">'
                   f'<text x="0" y="-16" text-anchor="middle" class="lbl">{c["ref"]}</text>'
                   f'<text x="0" y="-1" text-anchor="middle" class="val">'
                   f'{c.get("value") or "缺数值"}</text></g>')
        out.append('</g>')
    out.append('</svg>')
    return "".join(out)


def _fr(x):
    try:
        return Fraction(str(x))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _import_text(cli: TestClient, text: str) -> str:
    r = cli.post("/api/from-spice", json={"text": text})
    assert r.status_code == 200, f"from-spice 返回 {r.status_code}"
    body = r.json()
    assert body.get("ok"), f"from-spice 失败：{body.get('error')}"
    return body["session"]["sid"]


def main() -> int:
    # 会话目录改到临时目录，别把 runs/ 塞满测试产物
    with tempfile.TemporaryDirectory(prefix="ca_api_test_") as td:
        srv.RUNS_DIR = Path(td)
        # ★ 密钥配置也必须重定向。视觉设置的保存端点写的是真文件
        #   （config/secrets.local.json），测试如果往那儿写，就等于**替用户改了
        #   他的 key** —— 而用户明确说过"我稍后自己填新 key"。
        #   load_config / save_config 都在**调用时**读这个模块全局，所以改它有效。
        _secrets_backup = (vconf.SECRETS_LOCAL, vconf.SECRETS_EXAMPLE)
        vconf.SECRETS_LOCAL = Path(td) / "secrets.local.json"
        vconf.SECRETS_EXAMPLE = Path(td) / "secrets.example.json"
        # raise_server_exceptions=False：否则 TestClient 会把异常直接抛出来，
        # 我们就看不到真实的 HTTP 状态码，也就**测不出**"是不是 500"。
        cli = TestClient(srv.app, raise_server_exceptions=False)
        try:
            return _run(cli, td)
        finally:
            # 最后把每个端点的效果都还原：测试不许在用户机器上留下任何痕迹
            vconf.SECRETS_LOCAL, vconf.SECRETS_EXAMPLE = _secrets_backup


def _run(cli: TestClient, td: str) -> int:
    global FAILS

    banner("健康检查")
    r = cli.get("/api/health")
    eq("GET /api/health 状态码", r.status_code, 200)
    check("健康检查里如实报告位图通道已开启",
          r.json().get("bitmap_channel") is True)
    check("健康检查里上报 ngspice 可用性", "ngspice" in r.json())
    vis = r.json().get("vision") or {}
    check("健康检查里带上视觉层的体检（档位/升级条件/能不能调模型）",
          all(k in vis for k in ("enabled", "mode", "escalate_on",
                                 "will_call_vlm", "local", "vlm")))
    check("健康检查**不**回明文 key",
          "api_key" not in (vis.get("vlm") or {})
          or not (vis.get("vlm") or {}).get("api_key"))

    banner("纯 R/V 电路：基线（一直是对的，防止把好的改坏）")
    sid = _import_text(cli, NET_PURE_RV)
    r = cli.post("/api/solve", json={"sid": sid})
    eq("POST /api/solve 状态码", r.status_code, 200)
    body = r.json()
    check("ok = true", body.get("ok") is True)
    eq("V(2) = 12·200/300 = 8",
       next(x["exact"] for x in body["pack"]["node_table"] if x["node"] == "2"), "8")
    check("文本报告非空", bool(body.get("text_report")))
    check("reduction 为空（电路里没有 C/L）", body["pack"]["reduction"] is None)

    banner("★ 含电感的电路：曾经 HTTP 500，现在必须 200")
    sid_l = _import_text(cli, NET_WITH_L)
    r = cli.post("/api/solve", json={"sid": sid_l})
    eq("POST /api/solve 状态码（回归：曾 500）", r.status_code, 200)
    body_l = r.json()
    check("ok = true", body_l.get("ok") is True)
    check("没有 text_report_error（报告渲染成功）",
          body_l.get("text_report_error") is None,
          str(body_l.get("text_report_error")))
    check("文本报告非空且含化简一节",
          bool(body_l.get("text_report")) and "直流稳态化简" in body_l["text_report"])
    red = body_l["pack"]["reduction"]
    check("reduction 非空且 applied", bool(red) and red["applied"])
    eq("被短路移除的 L1 电流已反算 = 1",
       [(e["ref"], e.get("current")) for e in red["shorted"]], [("L1", "1")])
    eq("化简后支路表只剩 V1 / R1 / R2",
       sorted(x["ref"] for x in body_l["pack"]["branch_table"]),
       ["R1", "R2", "V1"])
    check("被移除的节点 3 不再出现在节点表里",
          "3" not in [x["node"] for x in body_l["pack"]["node_table"]])
    check("总判定通过", body_l["pack"]["overall_pass"] is True)

    banner("★ 无解电路（电压源被电感短路）：必须给出可读原因，不许 5xx、不许报通过")
    sid_u = _import_text(cli, NET_UNSOLVABLE)
    r = cli.post("/api/solve", json={"sid": sid_u})
    eq("POST /api/solve 状态码", r.status_code, 200)
    body_u = r.json()
    check("ok = false", body_u.get("ok") is False)
    eq("失败阶段标记为 solve", body_u.get("stage"), "solve")
    err = body_u.get("error") or ""
    check("错误信息说明『无解』", "无解" in err, err.splitlines()[0] if err else "")
    check("错误信息点名 V1", "V1" in err)
    check("错误信息给出怎么办", "怎么办" in err)
    check("响应里**没有** pack（不会给出一个会被误读为通过的结论）",
          "pack" not in body_u)
    # 这一条是本缺陷的核心：旧实现返回一个 overall_pass=True 的 pack
    check("错误信息明确否定『算不出来』这种误导说法",
          "不是" in err and "算不出来" in err)

    banner("关闭化简（reduce_dc=false）时含 C/L 的电路必须显式报错，不许硬解")
    r = cli.post("/api/solve", json={"sid": sid_l, "reduce_dc": False})
    eq("状态码", r.status_code, 200)
    body_nd = r.json()
    check("ok = false", body_nd.get("ok") is False)
    check("错误信息提示要先做直流稳态化简",
          "直流稳态化简" in (body_nd.get("error") or ""),
          (body_nd.get("error") or "").splitlines()[0])

    banner("视觉设置端点：只回掩码、写盘后重新读盘、空 key 不动原值")
    r = cli.get("/api/vision/config")
    eq("GET /api/vision/config 状态码", r.status_code, 200)
    body_v = r.json()
    check("ok = true", body_v.get("ok") is True)
    cf = body_v.get("config") or {}
    check("回了完整的可调项",
          all(k in cf for k in ("enabled", "mode", "escalate_on", "base_url",
                                "model", "timeout", "max_tokens", "max_edge",
                                "blank_text", "secrets_path")))
    # ★ 回归：密钥文件被指到项目之外时，**报错文案**里那句"写进 <路径>"曾经会
    #   直接抛 ValueError（relative_to 越界），把 GET/POST 打成 500。
    #   本测试就是把 SECRETS_LOCAL 重定向到临时目录跑的 —— 只要还能拿到 200，
    #   这一条就成立。它比"路径好不好看"重要得多：一条本该告诉用户去哪填 key 的
    #   提示，自己把请求打崩，比没有提示更坏。
    check("★ 配置文件在项目之外也不许把端点打崩（曾经 500）",
          r.status_code == 200 and bool(cf.get("secrets_path")),
          str(cf.get("secrets_path"))[:70])
    check("回了 probe（一条响应就能画完整个面板）",
          isinstance(body_v.get("probe"), dict))
    check("回了已装的 OCR 语言（不装语言包是预期分支，要让用户看得到）",
          isinstance(body_v.get("ocr_langs_installed"), list))
    # 这一条是硬要求：GET 永远不许回明文 key
    secret = "sk-LOCAL-SECRET-MUST-NOT-LEAK-1234"
    im_path = Path(td) / "probe.png"
    _tiny_png().save(im_path)
    r = cli.post("/api/vision/config", json={"config": {
        "api_key": secret, "mode": "local_only"}})
    eq("POST /api/vision/config 状态码", r.status_code, 200)
    check("保存成功", r.json().get("ok") is True)
    check("保存后回的仍是掩码，不是明文",
          secret not in r.text and "*" in (r.json()["config"]["api_key"] or ""))
    r = cli.get("/api/vision/config")
    check("GET 也不泄漏明文", secret not in r.text)
    check("写盘后重新读盘，值真的生效了",
          r.json()["config"]["mode"] == "local_only",
          r.json()["config"]["mode"])
    # 空 key = 不动原值（防止前端把掩码回填把 key 清掉）
    cli.post("/api/vision/config", json={"config": {"api_key": ""}})
    r = cli.get("/api/vision/config")
    check("api_key 传空串 -> 不动原值（key 还在）",
          r.json()["config"]["api_key_present"] is True)
    # 非法档位必须被挡住并留痕，不许静默接受
    r = cli.post("/api/vision/config", json={"config": {"mode": "乱填"}})
    check("非法档位不会被写进去（或至少给出警告）",
          r.json()["config"]["mode"] != "乱填" or bool(r.json().get("warnings")),
          str(r.json().get("warnings"))[:60])
    cli.post("/api/vision/config", json={"config": {"mode": "local_first"}})
    # ★ blank_text 曾经是"死开关"：表单里有勾选框，但 to_dict / to_file_dict /
    #   save_config 三处都没有这个字段 —— 勾选框永远显示未勾，取消了也白取消。
    #   所以这里必须测**往返**，光测"字段在不在"抓不到这种病。
    check("blank_text 初值是 true（涂白文字的默认行为）",
          r.json()["config"].get("blank_text") is True,
          str(r.json()["config"].get("blank_text")))
    cli.post("/api/vision/config", json={"config": {"blank_text": False}})
    r = cli.get("/api/vision/config")
    check("★ blank_text 关掉之后能读回来（曾经怎么写都还是 true）",
          r.json()["config"].get("blank_text") is False,
          str(r.json()["config"].get("blank_text")))
    cli.post("/api/vision/config", json={"config": {"blank_text": True}})
    r = cli.get("/api/vision/config")
    check("blank_text 能改回 true",
          r.json()["config"].get("blank_text") is True)

    banner("★ 位图通道：导入一张合成回路图，必须出拓扑正确的网表")
    r = cli.post("/api/import", files={"file": ("loop.png", im_path.read_bytes(),
                                                "image/png")})
    eq("POST /api/import（位图）状态码（阻塞调用不许把事件循环按住）",
       r.status_code, 200)
    body_b = r.json()
    check("ok = true（本地层结构完整，无需调模型）", body_b.get("ok") is True,
          str(body_b.get("error"))[:80])
    check("通道识别为 bitmap",
          (body_b.get("session") or {}).get("channel") == "bitmap")
    vb = body_b.get("vision") or {}
    check("如实记录这一遍是本地层给的结论", vb.get("tier") == "local", vb.get("tier"))
    comps_b = (body_b.get("ir") or {}).get("components") or []
    refs_b = {c["ref"]: c for c in comps_b}
    check("定位到 R1 与 V1", {"R1", "V1"} <= set(refs_b), sorted(refs_b))
    if {"R1", "V1"} <= set(refs_b):
        check("★ R1 两端不是同一个结点（不许把电阻短接）",
              refs_b["R1"]["nodes"][0] != refs_b["R1"]["nodes"][1],
              refs_b["R1"]["nodes"])
        check("★ R1 与 V1 接在同一对结点上（回路接法）",
              set(refs_b["R1"]["nodes"]) == set(refs_b["V1"]["nodes"]))
        check("类型来源如实标为 cv（几何层）",
              refs_b["R1"]["evidence"]["source"] == "cv")
        check("数值从图上读出来了（10k -> 10000）",
              refs_b["R1"]["value"] == 10000.0, refs_b["R1"]["value"])
    check("叠图用的是视觉层自己的那份（mode=vision，不是 IR 节点坐标那份）",
          (body_b.get("preview") or {}).get("mode") == "vision",
          str((body_b.get("preview") or {}).get("error"))[:60])
    pv = body_b.get("preview") or {}
    check("叠图里画了结点与 OCR 词（可逐项核对）",
          "ca-ov-wires" in (pv.get("svg") or "")
          and "ca-ov-text" in (pv.get("svg") or ""))
    check("叠图带图例", isinstance(pv.get("legend"), dict) and pv["legend"])
    check("read 图依据里带了视觉层的统计",
          "需人工过目" in json.dumps(body_b, ensure_ascii=False))

    banner("位图通道：读不出来时必须带着诊断包回来，而不是只报一句失败")
    blank = Path(td) / "blank.png"
    _blank_png().save(blank)
    r = cli.post("/api/import", files={"file": ("blank.png", blank.read_bytes(),
                                               "image/png")})
    eq("状态码仍然是 200（这是业务失败，不是服务端错误）", r.status_code, 200)
    body_bad = r.json()
    check("ok = false", body_bad.get("ok") is False)
    check("★ 给出了具体原因（不是一句『导入失败』）",
          bool(body_bad.get("error")) and len(body_bad["error"]) > 8,
          str(body_bad.get("error"))[:70])
    check("位图读不出来时**保留会话**（文件已落盘、诊断包要能继续看）",
          bool((body_bad.get("session") or {}).get("sid")),
          str(body_bad.get("session"))[:60])
    check("★ 带回了结构性问题清单（哪一步没读出什么）",
          isinstance(body_bad.get("issues"), list) and body_bad["issues"],
          str([i.get("code") for i in (body_bad.get("issues") or [])])[:80])
    check("带回了视觉层结论（两级各自跑到哪）",
          isinstance(body_bad.get("vision"), dict)
          and "escalation" in body_bad["vision"])
    check("带回了叠图（读不出来的时候最需要看它）",
          bool((body_bad.get("preview") or {}).get("svg")))

    # ==================================================================
    # ★ 先审后算：OCR 结果先在界面上给用户改，要不要交给模型由用户按按钮
    # ==================================================================
    banner("★ 默认不自动调模型：导入位图后停在「等你过目」这一步")
    r = cli.post("/api/import", files={"file": ("loop.png", im_path.read_bytes(),
                                                "image/png")})
    b_rev = r.json()
    eq("位图导入状态码", r.status_code, 200)
    check("本地结构完整 → 出网表", b_rev.get("ok") is True, str(b_rev.get("error"))[:60])
    vb2 = b_rev.get("vision") or {}
    check("★ tier = local（默认 escalate_on=manual，没自动调模型）",
          vb2.get("tier") == "local", vb2.get("tier"))
    check("★ vlm 那一级是空的（一次调用都没发出去）", vb2.get("vlm") is None,
          str(vb2.get("vlm"))[:60])
    sid2 = (b_rev.get("session") or {}).get("sid")
    check("★ 把「该不该交给模型」的理由摆出来了（不能只写「不升级」）",
          bool(vb2.get("escalation")), vb2.get("escalation"))
    loc2 = vb2.get("local") or {}
    ocr2 = loc2.get("ocr") or {}
    words2 = ocr2.get("words") or []
    check("★ 结论包里带回了逐词读数（校对表要靠它画）",
          bool(words2), f"{len(words2)} 个词")
    refs2 = [c["ref"] for c in (b_rev.get("ir") or {}).get("components") or []]
    check("先记下原始读数，后面改完好对照", "R1" in refs2, refs2)

    banner("★ 文字校对：改文字 -> 重算网表（毫秒级、不动几何）")
    # 找到数值那一格，把它改成 22k
    val_i = next((i for i, w in enumerate(words2)
                  if w["text"].strip().lower().endswith("k")), None)
    check("OCR 读到了一个带单位的数值", val_i is not None,
          [w["text"] for w in words2])
    if val_i is not None:
        r = cli.post("/api/vision/ocr", json={
            "sid": sid2, "edits": [{"op": "set", "i": val_i, "text": "22k"}]})
        b_e = r.json()
        eq("POST /api/vision/ocr 状态码", r.status_code, 200)
        check("ok = true", b_e.get("ok") is True, str(b_e.get("error"))[:60])
        check("★ 逐条回报改了什么（原来是什么 → 现在是什么，不静默）",
              any("22k" in n for n in (b_e.get("notes") or [])),
              str(b_e.get("notes"))[:80])
        vmap = {c["ref"]: c.get("value")
                for c in (b_e.get("ir") or {}).get("components") or []}
        check("★ 网表里的数值跟着变了（校对真的生效，不是只改了显示）",
              vmap.get("R1") == 22000.0, vmap)
        check("拓扑没被人为改动（改文字不该动几何）",
              sorted(vmap) == sorted(refs2), sorted(vmap))

        banner("★ 重放语义：清空修改 -> 必须回到机器原始读数")
        r = cli.post("/api/vision/ocr", json={"sid": sid2, "edits": []})
        b_r = r.json()
        vmap2 = {c["ref"]: c.get("value")
                 for c in (b_r.get("ir") or {}).get("components") or []}
        check("★ 改错了改回去 = 真的回得去（不是把修改累加上去）",
              vmap2.get("R1") == 10000.0, vmap2)

        banner("★ 标记噪声：丢掉一个读错的词，网表跟着变")
        r = cli.post("/api/vision/ocr", json={
            "sid": sid2, "edits": [{"op": "drop", "i": val_i}]})
        b_d = r.json()
        vmap3 = {c["ref"]: c.get("value")
                 for c in (b_d.get("ir") or {}).get("components") or []}
        check("★ 丢掉那个数值词之后，数值变成待人工填（而不是留着旧值）",
              vmap3.get("R1") is None, vmap3)
        check("★ 并如实报出「这一格读不出来了」",
              any(i.get("code") in ("value_missing", "value_split")
                  for i in (b_d.get("rebuild_issues") or [])),
              str([i.get("code") for i in (b_d.get("rebuild_issues") or [])])[:70])
        # 收尾：还原成没有人工修改的状态
        cli.post("/api/vision/ocr", json={"sid": sid2, "edits": []})

        banner("★ 补填「有字没读出来」的块（以前这里没有入口）")
        # 造一个能落进 dropped 的形状：一个只有一两个字符的短文本块。
        # 实测 Windows OCR 对短 token 会整条跳过（无解），而几何层仍然知道"这儿有字"。
        drop_png = Path(td) / "loop_shorttext.png"
        im_short = _tiny_png(with_short_text=True)
        im_short.save(drop_png)
        r = cli.post("/api/import", files={"file": ("loop_shorttext.png",
                                                    drop_png.read_bytes(),
                                                    "image/png")})
        b_s = r.json()
        loc_s = (b_s.get("vision") or {}).get("local") or {}
        ocr_s = loc_s.get("ocr") or {}
        dropped = ocr_s.get("dropped") or []
        check("这一档能测到「有字没读出来」（否则下面那条断言会静默跳过）",
              bool(dropped), f"dropped={len(dropped)} words={len(ocr_s.get('words') or [])}")
        if dropped:
            bl = dropped[0]
            sid_s = (b_s.get("session") or {}).get("sid")
            v_before = {c["ref"]: c.get("value")
                        for c in (b_s.get("ir") or {}).get("components") or []}
            r = cli.post("/api/vision/ocr", json={
                "sid": sid_s,
                "edits": [{"op": "add", "text": "47k", "x": bl["x"],
                           "y": bl["y"], "w": bl["w"], "h": bl["h"]}]})
            b_f = r.json()
            loc_f = ((b_f.get("vision") or {}).get("local") or {})
            # ★ 这里**不能**写成 `A or B`：补填对的时候 `dropped` 正好是空列表 `[]`，
            #   而 `[] or B` 会穿透到 B（原始那份、仍有 1 条），于是"修好了"被读成
            #   "没修好"。要按**字段在不在**分流，不是按真假。
            eff = loc_f.get("ocr_effective")
            check("★ 改动过就带回「生效的那份」读数（没有它，界面分不清改没改）",
                  eff is not None, f"ocr_effective={type(eff).__name__}")
            dropped_after = (eff.get("dropped") or [] if eff is not None
                             else (loc_f.get("ocr") or {}).get("dropped") or [])
            v_after = {c["ref"]: c.get("value")
                       for c in (b_f.get("ir") or {}).get("components") or []}
            check("★ 补填之后那个块不再算「没读出来」（界面不会自相矛盾）",
                  len(dropped_after) < len(dropped),
                  f"{len(dropped)} -> {len(dropped_after)}")
            check("★ 补填的文字真的进了网表（47k -> 47000）",
                  47000.0 in [v for v in v_after.values() if v is not None],
                  f"{v_before} -> {v_after}")

    banner("★ 手动交给视觉模型：由用户按按钮，不按不花钱")
    # ★ 必须把 base_url 指到一个**本机必然拒绝**的端口。
    #   不指的话，这条会拿上面那个假 key 去请求真实端点 —— 测试里出现一次真的
    #   外网调用：慢、依赖网络、还会拿废弃 key 去撞人家的鉴权。
    #   "鉴权失败"那条路已经由 test_vision.py 的假服务端（_FakeVLM 回 401）覆盖了，
    #   这里只需要一个**确定且不涉网**的失败。
    cli.post("/api/vision/config", json={"config": {
        "base_url": "http://127.0.0.1:1/v1"}})
    r = cli.post("/api/vision/escalate", json={"sid": sid2})
    eq("POST /api/vision/escalate 状态码（模型调不通也只是 200）", r.status_code, 200)
    b_up = r.json()
    check("★ 明确回答了「模型到底调通没有」（ok 会被本地网表带成 true，分不出来）",
          b_up.get("vlm_ok") is False, str(b_up.get("vlm_ok")))
    check("★ 给了一句能照做的原因（没配 key 就该说没配 key）",
          bool(b_up.get("vlm_reason")), str(b_up.get("vlm_reason"))[:80])
    check("★ 本地那份网表还在（不因为模型没调通就把结论丢掉）",
          bool(b_up.get("ir")), str(b_up.get("tier")))
    check("★ escalation 改口了（不能说「采信了模型结论」，那是报告在说谎）",
          "没成功" in (b_up.get("escalation") or ""),
          (b_up.get("escalation") or "")[-60:])

    banner("非位图会话：这两个端点必须明确拒绝，不许静默什么都不做")
    r = cli.post("/api/vision/ocr", json={"sid": sid_l, "edits": []})
    check("文本通道点「重新生成网表」 -> 明确说这个通道没有校对表",
          r.json().get("ok") is False and "OCR" in (r.json().get("error") or ""),
          str(r.json().get("error"))[:70])
    r = cli.post("/api/vision/escalate", json={"sid": sid_l})
    check("文本通道点「交给模型」 -> 明确拒绝",
          r.json().get("ok") is False, str(r.json().get("error"))[:70])
    r = cli.post("/api/vision/ocr", json={"sid": "不存在的会话", "edits": []})
    check("会话不存在 -> 404（不是 500）", r.status_code == 404, r.status_code)

    banner("视觉设置还原（别把测试写进用户的配置文件）")
    # ★ 上面为了测端点把 mode 改成过 local_first，key 也写进去过。
    #   密钥配置文件是用户的，测试**不许**留下痕迹 —— 这里把 key 清回空。
    r = cli.post("/api/vision/config", json={"config": {"api_key": ""}})
    check("测试结束时不会把假 key 留在用户配置里",
          secret not in json.dumps(r.json(), ensure_ascii=False))

    # ==================================================================
    # 手绘电路图：元件定义由后端给 + 画布产出能不能解析回来
    # ==================================================================
    banner("手绘面板：元件定义由后端给（前端一行符号都不重抄）")
    r = cli.get("/api/symbols")
    eq("GET /api/symbols 状态码", r.status_code, 200)
    sy = r.json()
    ks = {s["kind"] for s in sy.get("symbols") or []}
    # ★ 这条别写死成 {"R","V","I","C","L"} —— 加了受控源之后它就是 9 种。
    #   写死的话，每加一种元件都要来改测试，而真正该钉住的是
    #   「画布拿到的 = IR 允许的」这个**等式**，不是某一次的快照。
    check("★ 画布拿到的元件集 == IR 允许的元件集（ALLOWED_KINDS 一种不漏）",
          ks == set(ALLOWED_KINDS), sorted(ks))
    check("★ 四种受控源也在画布上（否则手绘场景画不了受控源）",
          set(CONTROLLED_KINDS) <= ks, sorted(ks))
    check("★ 位图通道那五种也在（画布与识图共用同一份符号）",
          set(BITMAP_KINDS) <= ks, sorted(ks))
    check("每个元件都带本体半长 —— 画布要靠它把导线停在本体边缘",
          all(isinstance(s.get("body_half"), (int, float)) and s["body_half"] > 0
              for s in sy["symbols"]))
    check("带中文名与单位（调色板与数值标签直接用）",
          all(s.get("label") and "unit" in s for s in sy["symbols"]))
    check("★ 带上了渲染器那份 CSS（画布与出图必须同一个样子）",
          "<style>" in (sy.get("style") or "") and ".circuit-svg" in sy["style"])
    check("栅格步长是正数", isinstance(sy.get("grid"), int) and sy["grid"] > 0)

    banner("★ 手绘产出（等价格式）经 /api/from-svg 必须出正确拓扑")
    # 串联回路：V1 在左边竖放（A→E），R1/R2 在上边并排（A→B→C），
    # 右边与下边是普通导线（C→D→E）。于是三个结点两两共用、正好成环。
    PS = {"A": (100, 100), "B": (300, 100), "C": (500, 100),
          "D": (500, 400), "E": (100, 400)}
    comps = [
        {"ref": "V1", "kind": "V", "a": PS["A"], "b": PS["E"], "value": "12"},
        {"ref": "R1", "kind": "R", "a": PS["A"], "b": PS["B"], "value": "1k"},
        {"ref": "R2", "kind": "R", "a": PS["B"], "b": PS["C"], "value": "3k"},
    ]
    wires = [(PS["C"], PS["D"]), (PS["D"], PS["E"])]
    svg_d = _canvas_svg(comps, wires)
    r = cli.post("/api/from-svg", json={"svg": svg_d, "name": "手绘串联"})
    eq("POST /api/from-svg 状态码", r.status_code, 200)
    b_d = r.json()
    check("ok = true", b_d.get("ok") is True, str(b_d.get("error"))[:90])
    if b_d.get("ok"):
        cm = {c["ref"]: c for c in b_d["ir"]["components"]}
        check("三个元件都建起来了", set(cm) == {"V1", "R1", "R2"}, sorted(cm))
        check("★ 取值走工程记法（1k→1000、3k→3000，不是被当字符串丢掉）",
              cm.get("R1", {}).get("value") == 1000.0
              and cm.get("R2", {}).get("value") == 3000.0,
              {k: v.get("value") for k, v in cm.items()})
        if set(cm) == {"V1", "R1", "R2"}:
            ns = [set(cm[k]["nodes"]) for k in ("V1", "R1", "R2")]
            sh = [ns[0] & ns[1], ns[1] & ns[2], ns[0] & ns[2]]
            check("★ 连接是**几何反推**出来的：每条支路与另外两条各共用一个结点",
                  all(len(s) == 1 for s in sh), sh)
            check("★ 而且是**串联**：三个共用结点互不相同（连成环，不是全并在一起）",
                  len({next(iter(s)) for s in sh}) == 3, sh)
            ns_all = {n for c in cm.values() for n in c["nodes"]}
            check("★ 结点名是解析器自己算出来的（画布一个都没写进去，全靠几何）",
                  len(ns_all) == 3 and "0" in ns_all, sorted(ns_all))
            check("★ 元件身份与取值按**精确**采信（画布写了语义标记，不是猜的）",
                  all(c["evidence"]["source"] == "exact" for c in cm.values()),
                  {k: v["evidence"]["source"] for k, v in cm.items()})

            banner("★ 画完就能算：手绘的图直接进三法")
            sid_d = b_d["session"]["sid"]
            r = cli.post("/api/solve", json={"sid": sid_d})
            sl = r.json()
            check("求解 ok = true", sl.get("ok") is True,
                  str(sl.get("error"))[:90])
            if sl.get("ok"):
                # ★ 期望值全部**独立手算**，不从输出里抄：
                #   12V 经 1k 串 3k 分压 → R1 上 3V、R2 上 9V；i = 3V/1k = 3mA；
                #   功率 12V×3mA = 36mW = 3m²×1k + 3m²×3k = 9mW + 27mW ✓
                bt = {e["ref"]: e for e in sl["pack"]["branch_table"]}
                nt = {e["node"]: e for e in sl["pack"]["node_table"]}
                check("★ 支路电流 = 3mA（三法各自都给这个值）",
                      all(_fr(bt[k].get("exact")) == Fraction(3, 1000)
                          and _fr(bt[k].get("mna")) == Fraction(3, 1000)
                          and _fr(bt[k].get("branch")) == Fraction(3, 1000)
                          for k in ("V1", "R1", "R2")),
                      {k: v.get("exact") for k, v in bt.items()})
                # ★ 结点电位**按物理对应**断言，绝不按"排序后的列表"硬比。
                #   结点名是解析器自己算出来的（这里是 N1/N2，不是数字），
                #   凡是按名字排序去比顺序的写法，都等于把测试绑死在命名细节上 ——
                #   名字一换（N1→N3）就会莫名其妙地红，而电路一点没变。
                #   改判据：拿 V1 **自己的两个端子**当坐标系。
                #   IR 约定 declared_direction(V) = (nodes[1], nodes[0])，
                #   符号的"+"画在 nodes[0] 一侧，所以 nodes[0] 必须比 nodes[1] 高 12V。
                vn = cm["V1"]["nodes"]
                vp = nt.get(vn[0], {}).get("exact_float")
                vm = nt.get(vn[1], {}).get("exact_float")
                nz = sorted(round(nt[n].get("exact_float") or 0.0, 9) for n in nt)
                check("★ 结点电位正好是 {0, 9, 12} 三个（分压 3:1 对上了）",
                      nz == [0.0, 9.0, 12.0], nz)
                check("★ V1 正端（nodes[0]，符号画 + 的那侧）就是 12V、另一端是参考 0",
                      vp is not None and round(vp, 9) == 12.0
                      and vm is not None and round(abs(vm), 9) == 0.0,
                      {"+": vn[0], "-": vn[1], "V+": vp, "V-": vm})
                check("★ 三法互校通过（不是只跑了一路）",
                      sl["pack"].get("overall_pass") is True,
                      sl["pack"].get("failures"))

            banner("★ 接着当前会话改：同 sid，且不毁掉原件")
            before = sorted(p.name for p in (Path(td) / sid_d).iterdir())
            r = cli.post("/api/from-svg", json={"svg": svg_d, "sid": sid_d,
                                                "name": "改过"})
            b_k = r.json()
            check("接着改 ok = true", b_k.get("ok") is True,
                  str(b_k.get("error"))[:80])
            check("★ 会话 id 不变（用户的心智是「我在改这张图」，换 id 会留一堆别名的键）",
                  b_k.get("session", {}).get("sid") == sid_d,
                  b_k.get("session", {}).get("sid"))
            check("★ 报告里说清了「这是接着改的」以及原件叫什么",
                  bool((b_k.get("report") or {}).get("hand_edited")))
            after = sorted(p.name for p in (Path(td) / sid_d).iterdir())
            check("★ 原件没被覆盖（用户上传的东西不许被我们写掉）",
                  "source.svg" in after, f"{before} -> {after}")
            check("手绘版另存了一份", "hand.svg" in after, after)

    # ------------------------------------------------------------------
    # 来源链：在画布里连着改第二次，最初那份原件不能被顶掉
    # ------------------------------------------------------------------
    # ★ 这里刻意用**真的位图会话**（sid2，原件是 source.png）来验。
    #   如果起点本来就是 svg，那么"origin_channel 有没有被继承"和
    #   "原始文件名有没有被顶掉"都看不出来 —— 继承与重算得到同一个值，
    #   测试会一直是绿的，而 bug 在照片那条路上照样存在。
    banner("★ 位图会话进画布改两次：origin 要一直是照片，不能断成 hand.svg")
    r = cli.post("/api/from-svg", json={"svg": svg_d, "sid": sid2,
                                        "name": "照片改的"})
    b_h1 = r.json()
    check("位图会话能直接进画布改（同一个 sid）", b_h1.get("ok") is True,
          str(b_h1.get("error"))[:80])
    he1 = (b_h1.get("report") or {}).get("hand_edited") or {}
    check("★ 第一次改：记下最初通道是 bitmap（不是已经手绘过的 svg）",
          he1.get("origin_channel") == "bitmap", he1.get("origin_channel"))
    check("★ 第一次改：原件是那张照片 source.png",
          he1.get("original_source") == "source.png", he1.get("original_source"))
    check("第一次改：prev_channel 是紧邻的上一个状态 bitmap",
          he1.get("prev_channel") == "bitmap", he1.get("prev_channel"))
    check("★ 改完通道就转成 svg 了（后续解析走 hand.svg 那份）",
          (b_h1.get("session") or {}).get("channel") == "svg",
          (b_h1.get("session") or {}).get("channel"))

    r = cli.post("/api/from-svg", json={"svg": svg_d, "sid": sid2, "name": "再改"})
    b_h2 = r.json()
    he2 = (b_h2.get("report") or {}).get("hand_edited") or {}
    check("第二次改也 ok", b_h2.get("ok") is True, str(b_h2.get("error"))[:80])
    check("★ 第二次改：origin_channel 仍是 bitmap（继承，不是照抄当前通道）",
          he2.get("origin_channel") == "bitmap", he2.get("origin_channel"))
    check("★ 第二次改：原件仍是 source.png —— 绝不能被 hand.svg 顶掉",
          he2.get("original_source") == "source.png",
          he2.get("original_source"))
    check("★ prev_channel 如实记 svg（紧邻的上一个状态确实已经是手绘了）",
          he2.get("prev_channel") == "svg", he2.get("prev_channel"))
    check("★ 改了两次就记两次（不是每次都从 1 重数）",
          he2.get("edit_count") == 2, he2.get("edit_count"))

    banner("★ 换成 SVG 通道之后，视觉那两个端点必须说「通道不对」")
    # 这时的会话文件已经不是照片了（hand.svg），旧照片的 OCR 结论也清空了。
    # 端点若沉默地回一句"没有视觉层的结果"，用户会以为是自己漏了一步；
    # 必须明确说是**通道不对** —— 这两件事的补救办法完全不同。
    r = cli.post("/api/vision/ocr", json={"sid": sid2, "edits": []})
    e = str(r.json().get("error") or "")
    check("手绘会话点「重新生成网表」-> 明确说这个通道没有校对表",
          r.json().get("ok") is False and "通道" in e, e[:70])
    r = cli.post("/api/vision/escalate", json={"sid": sid2})
    e = str(r.json().get("error") or "")
    check("手绘会话点「交给模型」-> 明确说这个通道没有视觉模型可选",
          r.json().get("ok") is False and "通道" in e, e[:70])

    banner("手绘面板：坏输入与危险图都要有话，不许 500")
    r = cli.post("/api/from-svg", json={"svg": "   "})
    check("空画布 -> 明确说空，不是 500",
          r.status_code == 200 and "空" in (r.json().get("error") or ""),
          r.json().get("error"))
    r = cli.post("/api/from-svg", json={"svg": "<div>没有 svg 根元素</div>"})
    check("没有 <svg> 根元素 -> 明确说这是画布的错、要找开发者",
          r.status_code == 200 and "<svg>" in (r.json().get("error") or ""),
          str(r.json().get("error"))[:60])
    r = cli.post("/api/from-svg", json={"svg": svg_d, "sid": "不存在的会话"})
    check("★ 要改一个不存在的会话 -> 404，绝不静默新建一个",
          r.status_code == 404, r.status_code)
    # 两个元件画在同一条路径上 = 把元件短路了。解析器必须**报出来**，
    # 而不是给一个看着合理的答案 —— 这正是"不静默降级"要守的东西。
    short = [{"ref": "V1", "kind": "V", "a": PS["A"], "b": PS["B"], "value": "12"},
             {"ref": "R1", "kind": "R", "a": PS["A"], "b": PS["B"], "value": "1k"}]
    r = cli.post("/api/from-svg", json={"svg": _canvas_svg(short, []), "name": "短接"})
    b_bad = r.json()
    check("★ 画出来的图把元件短路了 -> 明确报出来（不是给个看似合理的答案）",
          b_bad.get("ok") is False or bool(b_bad.get("issues")),
          f"ok={b_bad.get('ok')} err={str(b_bad.get('error'))[:70]}")
    check("★ 说了原因是「本体被当导线/元件被短接」这一类，而不是一句「解析失败」",
          "短" in json.dumps(b_bad, ensure_ascii=False)
          or "本体" in json.dumps(b_bad, ensure_ascii=False))

    banner("已有会话进画布：/api/layout 给的结点坐标必须齐全")
    r = cli.post("/api/layout", json={"sid": sid_l, "mode": "grid"})
    eq("POST /api/layout 状态码", r.status_code, 200)
    bl = r.json()
    check("ok = true", bl.get("ok") is True, str(bl.get("error"))[:70])
    if bl.get("ok"):
        want = {n for c in srv._get(sid_l).ir.components for n in c.nodes}
        got = set(bl.get("node_pos") or {})
        check("★ 每个结点都有坐标（缺一个就会有一个元件放不进画布）",
              want <= got, f"缺 {sorted(want - got)}")
        check("版式自检结果一起回来（重叠了就不能拿去回导）",
              "roundtrip_safe" in bl and isinstance(bl.get("overlaps"), list))
    r = cli.post("/api/layout", json={"sid": "不存在的会话"})
    check("不存在的会话 -> 明确报错，不是 500",
          r.status_code in (200, 404) and r.json().get("ok") is not True,
          r.status_code)
    r = cli.post("/api/layout", json={"sid": sid_l, "mode": "乱填"})
    check("不认识的版式 -> 明确拒绝", r.json().get("ok") is not True,
          str(r.json().get("error"))[:50])

    banner("★ 前端产出格式自检（无头浏览器验不了，只能从源码上钉）")
    js = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    i0 = js.index("function dSubmitSvg")
    seg = js[i0:i0 + js[i0:].index("\n}")]
    check("★ 提交的 SVG **不**写 data-ca-a / data-ca-b —— 写了就把节点名硬编码了",
          "data-ca-a=" not in seg and "data-ca-b=" not in seg)
    check("提交的 SVG 写了 data-ca-p1 / data-ca-p2 —— 缺了整条语义提示会被丢掉",
          "data-ca-p1=" in seg and "data-ca-p2=" in seg)
    check("提交的 SVG 带 ca-component 类与 ref/kind",
          "ca-component" in seg and "data-ca-ref=" in seg and "data-ca-kind=" in seg)
    check("★ 值只在非空时才写 data-ca-value（写空串会让解析器报「解析不出来」）",
          'if (val ? ` data-ca-value' in seg or "val ?" in seg)

    # ==================================================================
    # 参数体系 / 受控源：名字 ↔ 绑定 ↔ 取值 这条链必须端到端通
    # ==================================================================
    banner("★ 元件定义带上'受控 / 独立 / 控制量 / 增益单位'（画布与参数页都要用）")
    defs = {s["kind"]: s for s in sy["symbols"]}
    for k, mode, gsym, out in (("E", "V", "μ", "V"), ("G", "V", "gm", "A"),
                               ("H", "I", "rm", "V"), ("F", "I", "α", "A")):
        d = defs.get(k) or {}
        check(f"{k}：受控源，控制量 {mode}、增益符号 {gsym}、输出 {out}",
              d.get("controlled") is True and d.get("independent") is False
              and d.get("control_mode") == mode and d.get("gain_symbol") == gsym
              and d.get("output") == out and d.get("detail"),
              str({x: d.get(x) for x in ("controlled", "control_mode",
                                         "gain_symbol", "output")}))
    check("五种非受控元件都标成独立元件",
          all((defs.get(k) or {}).get("controlled") is False
              and (defs.get(k) or {}).get("independent") is True
              for k in ("R", "V", "I", "C", "L")),
          str({k: (defs.get(k) or {}).get("controlled") for k in "RVICL"}))
    check("★ C/L 标成'不能参与直流求解'（界面要据此说明，而不是让它悄悄算出个错数）",
          all((defs.get(k) or {}).get("dc_solvable") is False for k in ("C", "L"))
          and all((defs.get(k) or {}).get("dc_solvable") is True
                  for k in ("R", "V", "I", "E", "G", "H", "F")),
          str({k: (defs.get(k) or {}).get("dc_solvable") for k in "RVICLEGHF"}))

    banner("★ 受控源网表：0V 探针 + 语义标记 → 被采样支路必须回到题目那条")
    # 这份网表是 to_spice 自己的输出格式：探针是**为了取电流自动插进去的**，
    # 题目里说的是 R1。两者只靠 * ca-ctrl 那一行分得开 —— 分不开就会
    # 把「被采样支路」显示成 Vsense_R1，用户以为题目里真有这个元件。
    sid_c = _import_text(cli, NET_CTRL)
    r = cli.get(f"/api/session/{sid_c}")
    ir_c = r.json()["ir"]
    h1 = [c for c in ir_c["components"] if c["ref"] == "H1"][0]
    check("被采样支路按语义标记还原为题目里的 R1（不是探针）",
          h1["ctrl"]["ref"] == "R1", str(h1["ctrl"]))
    check("探针仍记为实际取电流的支路（求解走它）",
          h1["ctrl"]["sense_ref"] == "Vsense_R1", str(h1["ctrl"]))
    check("还原这件事在 origin_warnings / diagnostics 里有话说（不静默）",
          any("还原" in w for w in (ir_c.get("origin") or {}).get("warnings") or [])
          or any("还原" in str(d) for d in ir_c.get("diagnostics") or []),
          str(((ir_c.get("origin") or {}).get("warnings") or [])[:2])[:150])

    r = cli.post("/api/solve", json={"sid": sid_c})
    eq("含受控源的网表：POST /api/solve 状态码", r.status_code, 200)
    body = r.json()
    check("含受控源的网表能解出来（ok = true）", body.get("ok") is True,
          str(body.get("error"))[:120])
    # ★ node_table 在 pack 里（/api/solve 的顶层是 ok/pack/params/…）
    eqb = {row["node"]: row["exact"]
           for row in (body.get("pack") or {}).get("node_table") or []}
    check("V(2) = 5（手算：R1/R2 串联分压，探针是 0V 不动它）",
          _fr(eqb.get("2")) == Fraction(5), str(eqb.get("2")))
    check("V(3) = 10（手算：2000·i(R1) = 2000×5mA）",
          _fr(eqb.get("3")) == Fraction(10), str(eqb.get("3")))
    check("返回体带参数表（界面靠它渲染参数页）",
          isinstance(body.get("params"), dict)
          and body["params"].get("items"), "缺少 params")
    check("参数表里标明了单位是 SI（界面表头直接用这句）",
          "SI" in ((body.get("params") or {}).get("units") or {}).get("header", ""),
          str(((body.get("params") or {}).get("units") or {}).get("header"))[:90])
    pvb_c = body.get("param_values_by_binder") or {}
    check("★ param_values_by_binder 的键与 params.by_binder 的键完全一致",
          set(pvb_c) == set((body.get("params") or {}).get("by_binder") or {}),
          f"只在解里 {sorted(set(pvb_c) - set((body['params'].get('by_binder') or {})))[:4]}；"
          f"只在表里 {sorted(set((body['params'].get('by_binder') or {})) - set(pvb_c))[:4]}")
    check("★ 按绑定取到的值就是解里的值（不是拿元件值冒充的）",
          _fr(pvb_c.get("node_u:3")) == Fraction(10)
          and _fr(pvb_c.get("branch_i:H1")) == Fraction(1, 100),
          f"node_u:3={pvb_c.get('node_u:3')!r} branch_i:H1={pvb_c.get('branch_i:H1')!r}")
    check("受控源在 params.controlled 里带控制方式与探针标记",
          [c for c in (body["params"].get("controlled") or [])
           if c["ref"] == "H1"][0].get("sampling_is_probe") is True,
          str([c for c in (body["params"].get("controlled") or [])
               if c["ref"] == "H1"])[:180])
    check("★ 导出的网表里带着语义标记行（否则往返一次被采样支路就丢了）",
          "* ca-ctrl H1 mode=I ref=R1 sense=Vsense_R1"
          in (cli.get(f"/api/session/{sid_c}").json().get("netlist") or ""),
          str([ln for ln in (cli.get(f"/api/session/{sid_c}").json()
                             .get("netlist") or "").splitlines()
               if "ca-ctrl" in ln]))
    check("网表的人读注释里写的是题目支路 R1，并点明探针'不是题目元件'",
          (lambda nl: "由 R1 支路的电流控制" in nl and "不是题目元件" in nl)(
              cli.get(f"/api/session/{sid_c}").json().get("netlist") or ""),
          str([ln for ln in (cli.get(f"/api/session/{sid_c}").json()
                             .get("netlist") or "").splitlines()
               if ln.startswith("* ↓")][:1])[:170])

    banner("★ /api/params：改名保缓存、改方程清缓存、非法项整包不动")
    sid_v = _import_text(cli, NET_VCVS)
    r = cli.post("/api/solve", json={"sid": sid_v})
    check("压控压源基线可解，V(3) = 3·V(2) = 20",
          _fr({row["node"]: row["exact"]
               for row in (r.json().get("pack") or {}).get("node_table") or []}
              .get("3")) == Fraction(20),
          str({row["node"]: row["exact"]
               for row in (r.json().get("pack") or {}).get("node_table") or []}))
    base = r.json()
    keys0 = set(base.get("param_values_by_binder") or {})
    check("基线缓存非空（否则下面两条'保/清'的断言就是空的）", bool(keys0),
          f"键 {len(keys0)} 个")

    # ---- ① 只改名：电路一条方程都没变，缓存必须留着
    r = cli.post("/api/params", json={"sid": sid_v, "renames": {"u_2": "uA"}})
    eq("POST /api/params（只改名）状态码", r.status_code, 200)
    b1 = r.json()
    check("只改名：ok = true", b1.get("ok") is True, str(b1.get("error"))[:110])
    check("改名后参数表里是新名字、旧名字没了",
          "uA" in (b1["params"]["by_binder"] or {}).values()
          and "u_2" not in (b1["params"]["by_binder"] or {}).values(),
          str(sorted((b1["params"]["by_binder"] or {}).values())[:6]))
    check("★ 只改名时**保留**上一次求解的取值（改名不改数，否则用户改个名就看不着数了）",
          set(b1.get("param_values_by_binder") or {}) == keys0,
          f"{len(b1.get('param_values_by_binder') or {})} vs {len(keys0)}")
    check("改过名的参数在表里标成'用户命名'（auto=false）",
          any(e["symbol"] == "uA" and e["auto"] is False
              for e in b1["params"]["items"]), "")

    # ---- ② 改表达式：解会变，旧的一批数当场作废
    r = cli.post("/api/params", json={"sid": sid_v, "exprs": {"E1": "3*uA + 5"}})
    b2 = r.json()
    check("填受控源表达式：ok = true", b2.get("ok") is True,
          str(b2.get("error"))[:110])
    check("★ 改表达式时**清空**上一次求解的取值（拿旧数配新方程最危险）",
          not (b2.get("param_values_by_binder") or {}),
          f"还剩 {len(b2.get('param_values_by_binder') or {})} 项")
    r = cli.post("/api/solve", json={"sid": sid_v})
    b3 = r.json()
    check("改完表达式再求解：V(3) = 3·V(2) + 5 = 25（表达式真被用上了）",
          _fr({row["node"]: row["exact"]
               for row in (b3.get("pack") or {}).get("node_table") or []}
              .get("3")) == Fraction(25),
          str({row["node"]: row["exact"]
               for row in (b3.get("pack") or {}).get("node_table") or []}))
    check("★ 改名连带改写了表达式里的符号（否则表达式就指向空气）",
          "uA" in (b3["ir"]["components"][3].get("ctrl") or {}).get("expr", ""),
          str((b3["ir"]["components"][3].get("ctrl") or {}).get("expr")))

    # ---- ③ 非法项：整包不动（改名也不许生效）
    r = cli.post("/api/params", json={
        "sid": sid_v, "renames": {"uA": "zz"},
        "ctrls": {"E1": {"nodes": ["2", "99"]}}})
    eq("非法控制端：仍然 200（这是输入问题、请用户改，不是服务端故障）",
       r.status_code, 200)
    b4 = r.json()
    check("非法控制端：ok = false 且给出可读原因",
          b4.get("ok") is False and "不在电路里" in (b4.get("error") or ""),
          str(b4.get("error"))[:110])
    # ★ 失败响应里**没有** params 字段（那是"输入有问题"的最小回包）。
    #   要核对"改名到底生效没有"，必须回读会话 —— 界面也是这么做的。
    syms4 = set((cli.get(f"/api/session/{sid_v}").json()
                 ["params"]["by_binder"] or {}).values())
    check("★ 整包被拒后，同一包里的改名**没有生效**（要么全成、要么原样）",
          "uA" in syms4 and "zz" not in syms4, str(sorted(syms4)[:8]))
    r = cli.post("/api/solve", json={"sid": sid_v})
    check("整包被拒后电路仍可解（没被半程状态搞坏）",
          r.json().get("ok") is True, str(r.json().get("error"))[:110])

    # ---- ④ 先回到标准形（把上一步填的表达式清掉），再改控制端看解变不变
    r = cli.post("/api/params", json={"sid": sid_v, "exprs": {"E1": ""}})
    check("清空表达式 = 回到增益的标准形（E1 的增益 3 还在，所以合法）",
          r.json().get("ok") is True
          and (r.json().get("exprs") or [{}])[0].get("mode") == "标准形（增益）",
          str(r.json().get("exprs")) or str(r.json().get("error"))[:100])
    r = cli.post("/api/solve", json={"sid": sid_v})
    nt = {row["node"]: row["exact"]
          for row in (r.json().get("pack") or {}).get("node_table") or []}
    check("回到标准形后 V(3) = 3·V(2) = 20", _fr(nt.get("3")) == Fraction(20), str(nt))

    # ---- ⑤ 合法改控制端：解必须跟着变
    r = cli.post("/api/params", json={"sid": sid_v,
                                      "ctrls": {"E1": {"nodes": ["1", "0"]}}})
    b5 = r.json()
    check("改控制端：ok = true 且报出新控制关系",
          b5.get("ok") is True and (b5.get("ctrls") or [{}])[0].get("nodes") == ["1", "0"],
          str(b5.get("ctrls")))
    check("★ 改控制关系同样清空缓存（控制端变了、方程就变了）",
          not (b5.get("param_values_by_binder") or {}), "")
    r = cli.post("/api/solve", json={"sid": sid_v})
    nt = {row["node"]: row["exact"]
          for row in (r.json().get("pack") or {}).get("node_table") or []}
    check("改控制端后 V(3) = 3·V(1) = 30（改的是被求解的那一份，不是显示层）",
          _fr(nt.get("3")) == Fraction(30), str(nt))

    banner("★ 被跳过的元件必须结构化报到界面上（回导丢件不许静默）")
    # ★ 这里钉的是「**丢了要说**」，不是「不许丢」：
    #   自动版式对并联支路会把两条排在同一轴线上（E1 与 R3 都是 3-0），
    #   回导时两个符号叠在一起、两端被判成同一个结点，于是一起消失。
    #   那是**版式层**的缺陷（另案），但无论修不修，
    #   用户在界面上都必须看到"这两个元件没建起来、为什么"。
    r = cli.post("/api/render", json={"sid": sid_v, "mode": "auto"})
    rj = r.json()
    # ★ /api/render 把版式自检结果**摊在顶层**（{"ok":…, "svg":…, **dg}），
    #   没有嵌套的 diagnostics —— 断言要按真实形状写，不能凭印象。
    check("自动版式对被排在同一轴线的并联支路会报 overlaps（版式自检生效）",
          isinstance(rj.get("overlaps"), list) and len(rj["overlaps"]) > 0,
          str(rj.get("overlaps"))[:140])
    check("★ 有重叠就标 roundtrip_safe=False（该插图不能拿去几何回导）",
          rj.get("roundtrip_safe") is False,
          f"roundtrip_safe={rj.get('roundtrip_safe')!r} layout={rj.get('layout')!r}")
    svg_bad = rj.get("svg") or ""
    r = cli.post("/api/from-svg", json={"svg": svg_bad, "name": "重叠回导",
                                        "new_session": True})
    b6 = r.json()
    check("叠图回导：端点不返回 5xx", r.status_code == 200, r.status_code)
    sk = b6.get("skipped_components")
    check("★ skipped_components 是**结构化**列表（不是一句笼统的警告）",
          isinstance(sk, list) and len(sk) > 0
          and all({"ref", "why", "detail"} <= set(x) for x in sk),
          str(sk)[:180])
    check("★ 每一条都点名位号 + 机器可读原因 + 人读细节",
          all(x["ref"] and x["why"] and x["detail"] for x in sk)
          and {"E1", "R3"} <= {x["ref"] for x in sk},
          str(sk)[:180])
    check("原因如实说是'短路'（两端落进同一个结点）",
          all(x["why"] == "short_circuit" for x in sk if x["ref"] in ("E1", "R3")),
          str([(x["ref"], x["why"]) for x in sk]))
    ow = " ".join(b6.get("origin_warnings") or []) + " ".join(b6.get("warnings") or [])
    check("★ 文字上也说清了后果（'这几条支路就是缺了，不是被偷偷补上'）",
          "缺" in ow or "跳过" in ow or "没" in ow, ow[:170])
    # 干净通道：没有跳过时必须是空列表，而不是字段缺失
    r = cli.post("/api/from-spice", json={"text": NET_PURE_RV})
    check("没有跳过的会话里 skipped_components 是空列表（字段一直在）",
          r.json().get("skipped_components") == [],
          str(r.json().get("skipped_components")))

    banner("其余端点冒烟")
    r = cli.get(f"/api/session/{sid_l}")
    eq("GET /api/session/{sid}", r.status_code, 200)
    r = cli.post("/api/render", json={"sid": sid_l, "mode": "auto"})
    eq("POST /api/render", r.status_code, 200)
    check("render 返回 svg", bool(r.json().get("svg")))
    r = cli.get("/")
    eq("GET / 返回前端", r.status_code, 200)

    print("\n" + "=" * 72)
    print(f"总计失败项：{FAILS}")
    print("=" * 72)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
