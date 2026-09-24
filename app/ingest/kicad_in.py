"""KiCad 通道：``.kicad_sch`` / kicadxml 网表 -> IR（零误差拓扑）。

这是四条通道里**唯一不需要"看"的**：网表本身就是拓扑的真值，
所以这一路的证据来源恒为 ``exact``、置信度 1.0，不存在读图歧义。

★ 但有一个例外必须警惕：**电压源的极性**。
KiCad 网表只告诉你"V1 的 1 脚接 N1、2 脚接 GND"，**不告诉你哪个脚是 + 端** ——
那是符号（libsymbol）层面的语义，不在网表里。
所以这一路出来的电压源，极性一律标成"需人工确认"，
在 WebUI 上一键可以把两端对调。这正是"不许靠肉眼定连接"的落地点：
不是不让你看，是不许**没看就当作对**。

用法：
    kicad_netlist_to_ir(path)            # 直接吃 kicad-cli 导出的 XML 网表
    kicad_sch_to_ir(path, kicad_cli)     # 吃 .kicad_sch，内部调 kicad-cli 先导网表
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..ir.model import Circuit, Component, CircuitError, Evidence
from .values import parse_engineering, parse_resistor

#: KiCad 器件类型 -> 我们的 kind。只收二端线性元件（C1 范围）。
PART_KIND: dict[str, str] = {
    "R": "R", "R_Small": "R", "R_US": "R", "R_Potentiometer": "R",
    "C": "C", "C_Small": "C", "C_Polarized": "C",
    "L": "L", "L_Small": "L",
    "V": "V", "VDC": "V", "VSIN": "V", "VPULSE": "V", "Battery": "V",
    "I": "I", "IDC": "I", "ISIN": "I", "IPULSE": "I",
}

#: 这些是仿真专用的"非电路"元件，静默忽略但不丢信息
SIM_ONLY = {"SIM_SPICE_PARAM", "SIM_PLOT_PARAMS", "SIM_MODEL", "GNDREF", "SPICE_NODE"}

#: 位号前缀兜底推断
PREFIX_KIND = {"R": "R", "C": "C", "L": "L", "V": "V", "I": "I"}

#: 参考节点的候选名（按优先级）
GND_NAMES = ("GND", "0", "GND1", "AGND", "DGND", "EARTH", "gnd")


def _find_kicad_cli(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    import os

    env = os.environ.get("CIRCUIT_AGENT_KICAD_CLI")
    if env and Path(env).is_file():
        return Path(env)
    import shutil
    w = shutil.which("kicad-cli")
    if w:
        return Path(w)
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for ver in ("10.0", "9.0", "8.0", "7.0"):
        for base in (local / "Programs" / "KiCad" / ver / "bin",
                     Path("C:/Program Files/KiCad") / ver / "bin"):
            if (base / "kicad-cli.exe").is_file():
                return base / "kicad-cli.exe"
    return None


def kicad_sch_to_ir(sch_path: str | Path, cli: str | Path | None = None) -> Circuit:
    """把 ``.kicad_sch`` 交给 kicad-cli 导出 XML 网表，再解析成 IR。"""
    sch = Path(sch_path)
    if not sch.is_file():
        raise CircuitError(f"原理图文件不存在：{sch}")
    exe = _find_kicad_cli(cli)
    if exe is None:
        raise CircuitError(
            "未找到 kicad-cli，无法导出网表。请装 KiCad，或设 "
            "CIRCUIT_AGENT_KICAD_CLI 环境变量指向 kicad-cli.exe"
        )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / (sch.stem + ".net")
        proc = subprocess.run(
            [str(exe), "sch", "export", "netlist", "--format", "kicadxml",
             "-o", str(out), str(sch)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not out.is_file():
            raise CircuitError(
                f"kicad-cli 导出网表失败（返回码 {proc.returncode}）："
                f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
            )
        text = out.read_text(encoding="utf-8", errors="replace")
    return kicad_netlist_text_to_ir(text, name=sch.stem,
                                    origin_note=f"kicad-cli sch export netlist ← {sch.name}")


def kicad_netlist_to_ir(net_path: str | Path) -> Circuit:
    p = Path(net_path)
    if not p.is_file():
        raise CircuitError(f"网表文件不存在：{p}")
    return kicad_netlist_text_to_ir(
        p.read_text(encoding="utf-8", errors="replace"), name=p.stem,
        origin_note=f"KiCad XML 网表文件：{p.name}")


def kicad_netlist_text_to_ir(text: str, *, name: str = "kicad",
                            origin_note: str = "KiCad XML 网表") -> Circuit:
    """解析 kicadxml 格式的网表。"""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise CircuitError(
            f"不是合法的 XML（{e}）。本通道吃的是 kicad-cli 导出的 kicadxml 网表；"
            "若你手上是 SPICE 网表，请走 spice.from_spice()。"
        ) from None

    diagnostics: list[dict] = []
    warnings: list[str] = []

    # ---- 1. 元件表：位号 -> (kind, value, part)
    comp_meta: dict[str, dict] = {}
    for comp in root.findall(".//components/comp"):
        ref = (comp.get("ref") or "").strip()
        if not ref:
            continue
        value_el = comp.find("value")
        raw_value = (value_el.text or "").strip() if value_el is not None else ""
        part = ""
        libsrc = comp.find("libsource")
        if libsrc is not None:
            part = (libsrc.get("part") or "").strip()
        if part in SIM_ONLY:
            diagnostics.append({"kind": "ignored_sim_only", "ref": ref, "part": part})
            continue
        kind = PART_KIND.get(part) or PREFIX_KIND.get(ref[:1].upper())
        comp_meta[ref] = {"kind": kind, "raw_value": raw_value,
                          "part": part, "pins": {}}

    # ---- 2. 网络表：net -> [(ref, pin)]
    nets: list[tuple[str, list[tuple[str, str]]]] = []
    for net in root.findall(".//nets/net"):
        nname = (net.get("name") or "").strip() or f"N{net.get('code','?')}"
        members = []
        for node in net.findall("node"):
            r = (node.get("ref") or "").strip()
            p = (node.get("pin") or "").strip()
            if r in comp_meta:
                members.append((r, p))
        if members:
            nets.append((nname, members))

    # ---- 3. 参考节点选取
    ref_node = next((g for g in GND_NAMES if any(n == g for n, _ in nets)), None)
    if ref_node is None:
        # 没有标准地名：挑引脚数最多的网络，并显式提醒人工确认
        counts = {n: len(m) for n, m in nets}
        ref_node = max(counts, key=lambda k: counts[k])
        diagnostics.append({
            "kind": "ref_node_guessed",
            "text": f"网表里没有 GND/0 这类地名，已把连接最多的网络 {ref_node!r} "
                    f"（{counts[ref_node]} 个引脚）当作参考节点。**请在界面上确认。**",
        })

    # ---- 4. 组装元件
    comps: list[Component] = []
    # 反查：ref -> {pin: net}
    for cname, members in nets:
        for r, p in members:
            comp_meta[r]["pins"][p] = cname

    for ref, meta in comp_meta.items():
        kind = meta["kind"]
        pins = meta["pins"]
        if kind is None:
            diagnostics.append({
                "kind": "unsupported_part",
                "ref": ref, "part": meta["part"],
                "text": f"{ref}（KiCad part={meta['part']!r}）无法判定元件类型，"
                        "已排除。若它其实是二端元件，请在此手工补上。",
            })
            continue
        if len(pins) != 2:
            if len(pins) == 0:
                diagnostics.append({
                    "kind": "dangling_component", "ref": ref,
                    "text": f"{ref} 没有连到任何网络（未连线），已排除",
                })
            else:
                diagnostics.append({
                    "kind": "multi_pin_unsupported", "ref": ref,
                    "text": f"{ref} 有 {len(pins)} 个引脚（{sorted(pins)}），"
                            "本版本只支持二端元件，已排除",
                })
            continue

        ordered = sorted(pins.keys())
        p1, p2 = ordered[0], ordered[1]
        a, b = pins[p1], pins[p2]

        if kind == "R":
            value, w = parse_resistor(meta["raw_value"])
        else:
            value, w = parse_engineering(meta["raw_value"], kind=kind)
        warnings.extend(f"{ref}: {x}" for x in w)

        if a == b:
            diagnostics.append({
                "kind": "shorted_component", "ref": ref,
                "text": f"{ref} 两个引脚都接在 {a} 上，被短接，已排除",
            })
            continue

        if kind in ("V", "I"):
            ev = Evidence(
                source="cv", confidence=0.5,
                detail=(f"拓扑来自 KiCad 网表（精确，pin{p1}->{a}, pin{p2}->{b}）；"
                        f"但 **极性无法从网表判定** —— 无法确定 pin{p1} 还是 pin{p2} 是 + 端，"
                        "必须人工确认方向"),
            )
        else:
            ev = Evidence(
                source="exact", confidence=1.0,
                detail=f"来自 KiCad 网表：{a} -- {b}（pin{p1}, pin{p2}）",
            )

        comps.append(Component(
            ref=ref, kind=kind, nodes=(a, b), value=value,
            evidence=ev, note=f"KiCad part={meta['part']}" if meta["part"] else "",
        ))

    if not comps:
        raise CircuitError("网表解析后没有得到任何可用元件")

    circ = Circuit(
        name=name, components=comps, ref_node=ref_node,
        diagnostics=diagnostics,
        origin={"channel": "kicad", "note": origin_note, "warnings": warnings},
    )
    return circ
