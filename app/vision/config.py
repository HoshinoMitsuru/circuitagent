"""视觉通道的配置与密钥层。

这个项目的两条禁令是「不许靠肉眼定连接」和「不许只算一遍」。视觉通道
天然要跟"看不清"打交道，所以这里再加三条自己的规矩：

1. **密钥不进代码、不进日志、不回前端。** 明文只允许存在于
   ``config/secrets.local.json`` 和进程内存里那一份；任何面向界面的
   返回值一律走 ``to_dict(mask=True)``，掩码之外连长度都不给。
2. **缺配置不算错误，"缺配置还硬跑"才算。** 文件不存在、JSON 坏了、
   字段类型不对 —— 一律退回默认值，并把原因记进 ``warnings``，
   由上层决定是"降级到本地 OCR"还是"明确叫用户去填 key"。
   这里绝不抛异常，因为"没配 key"是完全正常的一种状态。
3. **环境变量优先级最高。** 顺序是 环境变量 > secrets.local.json > 内置默认。
   ``sources`` 会逐字段记下取值来源，省得日后排查"为什么它用的是旧 key"。

关于第 1 条的补充：掩码只是防"回显"，不是防"读取"。这个服务是**本机
单用户**定位，任何能访问 8765 端口的人都能通过改配置接口写 key ——
将来若要多人用，鉴权得加在 API 层，而不是指望配置层把 key 藏起来。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..paths import config_dir, data_root

# ★ 这里的 ROOT 是"给人看相对路径"的**基准**，用可写数据根，不是资源根：
#   密钥文件最终落在 data_root()/config/ 下（打包成 exe 后通常是 exe 同级），
#   基准若是只读资源根（sys._MEIPASS），display_path() 永远给绝对路径 ——
#   不算错，但用户看到的是 `C:\Users\...\_MEI123456\config\...` 这种没法读的东西。
#   开发模式下 data_root() 就是项目根，行为与从前完全一致。
ROOT = data_root()
CONFIG_DIR = config_dir()

#: 真正存密钥的文件。**不要提交、不要复制进文档、不要贴进聊天记录。**
SECRETS_LOCAL = CONFIG_DIR / "secrets.local.json"

#: 可以提交的模板，只有字段名和默认值，api_key 是空的。
SECRETS_EXAMPLE = CONFIG_DIR / "secrets.example.json"

#: 文件自身带的版本号，将来改结构时好写迁移。
SCHEMA_ID = "circuit_agent.secrets/1"


def display_path(p: Path) -> str:
    """把路径转成"给人看"的样子：在项目里就给相对路径，不在就给绝对路径。

    ★ 别直接写 ``p.relative_to(ROOT)``：只要 SECRETS_LOCAL 指向项目之外
    （测试重定向到临时目录、用户把配置放到共享盘、以后挪走配置目录都会发生），
    ``relative_to`` 就抛 ``ValueError``。而这里正是**报错文案**要在的地方 ——
    一条本该告诉用户"该写哪个文件"的提示，自己把请求打成 500，比没有提示更糟。
    所以取不到相对路径时退化成绝对路径，绝不抛。
    """
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)

# ---------------------------------------------------------------- 默认值

#: DeepSeek 的 OpenAI 兼容根地址。注意：官方 SDK 的 base_url 就是它，
#: 补路径是 ``/chat/completions``；也接受 ``/v1/chat/completions``。
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: ★ 默认视觉模型。带 ``-exp`` 后缀表示**实验性**，随时可能被下线或改名，
#: 所以它只作为默认值出现在这里，任何地方都不许再硬编码一份 ——
#: 换模型要能只改配置（或环境变量）就生效。
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"

ENV_API_KEY = "CIRCUIT_AGENT_VLM_API_KEY"
ENV_BASE_URL = "CIRCUIT_AGENT_VLM_BASE_URL"
ENV_MODEL = "CIRCUIT_AGENT_VLM_MODEL"
ENV_TIMEOUT = "CIRCUIT_AGENT_VLM_TIMEOUT"
ENV_MODE = "CIRCUIT_AGENT_VISION_MODE"

#: 前端下拉用的预设。**只是替用户少打几个字**，不是能力声明 ——
#: 目前真机验证过能收图的只有 deepseek 那一条，其余是"OpenAI 兼容格式
#: 理论上能通"，界面上必须如实这么标注。
PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek（默认，已验证可收图）",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "note": "DeepSeek 下只有带 vision 能力的模型接受图片，"
                "其余模型会返回 400 This model does not support image。",
    },
    "openai": {
        "label": "OpenAI 或兼容端点（未在本项目实测）",
        "base_url": "https://api.openai.com/v1",
        "model": "",
        "note": "接口格式同为 OpenAI 兼容的 chat/completions + image_url，"
                "但本项目尚未对任何非 DeepSeek 端点做过实测，"
                "能不能收图请自行确认。",
    },
    "custom": {
        "label": "自定义（自填 base_url 与模型名）",
        "base_url": "",
        "model": "",
        "note": "任何 OpenAI 兼容端点都可以。base_url 要写到版本段"
                "（例如 https://host/v1），本工具会自己补 /chat/completions。",
    },
}

VALID_MODES = ("local_first", "local_only", "vlm_only")

#: 升级到 VLM 的触发条件。见 VisionConfig.escalate_on 的注释。
#:
#: ``manual`` 是**默认**：本地跑完就停在界面上，把「为什么建议/不建议升级」
#: 连同本地读出来的文字一起摆给用户看，**由用户点按钮决定要不要花这次调用**。
#: 这不是懒，是省钱的正确位置 —— 升级与否本来就要看用户手里的图源：
#: 手机拍的糊图值得交给模型，教科书的印刷截图本地就够了，机器分不出来。
VALID_ESCALATE = ("manual", "structural", "any", "never")


# ---------------------------------------------------------------- 取值工具
#
# 一律"宽松读、严格用"：读的时候不因为类型不对就炸，用的时候才校验。
# 这样一份手改坏的 JSON 不会让整个服务起不来，只会让某个字段退回默认。


def _as_bool(v: Any, default: bool, warn: list[str], key: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    if isinstance(v, int) and not isinstance(v, bool):
        return bool(v)
    if v is not None:
        warn.append(f"配置项 {key} 期望布尔值，实际是 {type(v).__name__}，已按默认值 {default} 处理。")
    return default


def _as_str(v: Any, default: str, warn: list[str], key: str, *, strip: bool = True) -> str:
    if isinstance(v, str):
        s = v.strip() if strip else v
        return s
    if v is not None:
        warn.append(f"配置项 {key} 期望字符串，实际是 {type(v).__name__}，已按默认值处理。")
    return default


def _as_float(v: Any, default: float, warn: list[str], key: str,
              *, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            warn.append(f"配置项 {key} 期望数字，无法解析 {v!r}，已按默认值 {default} 处理。")
            return default
    elif v is None:
        return default
    else:
        warn.append(f"配置项 {key} 期望数字，实际是 {type(v).__name__}，已按默认值 {default} 处理。")
        return default
    if lo is not None and f < lo:
        warn.append(f"配置项 {key} = {f} 小于下限 {lo}，已按 {lo} 处理。")
        return lo
    if hi is not None and f > hi:
        warn.append(f"配置项 {key} = {f} 大于上限 {hi}，已按 {hi} 处理。")
        return hi
    return f


def _as_int(v: Any, default: int, warn: list[str], key: str,
            *, lo: int | None = None, hi: int | None = None) -> int:
    return int(_as_float(v, float(default), warn, key,
                         lo=None if lo is None else float(lo),
                         hi=None if hi is None else float(hi)))


# ---------------------------------------------------------------- 配置对象


@dataclass
class OcrConfig:
    """本地 OCR 层。默认开 —— 它的价值就是"看得清就不用花钱调模型"。"""

    enabled: bool = True

    #: 语言回退链，按顺序尝试。实测这台机器上只装了 zh-Hans-CN，
    #: en-US / ja-JP / zh-Hant-TW 都没装，指定它们会拿到 None。
    langs: list[str] = field(default_factory=lambda: ["zh-Hans-CN", "en-US"])

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "langs": list(self.langs)}


@dataclass
class VisionConfig:
    """视觉通道的全部可调项。

    分成四组：开关、VLM 端点、VLM 请求参数、本地层门槛。
    """

    # ---- 开关与编排
    enabled: bool = True
    #: local_first = 本地优先、失败才调模型；local_only = 只用本地；
    #: vlm_only = 直接用模型（拿来跟本地结果做交叉验证很有用）。
    mode: str = "local_first"
    #: ★ 用户拍板的选择：本地失败时**整体采信** VLM 给出的网表，
    #: 而不是只让它补几个字。注意"最大信任"不等于"隐去不确定性" ——
    #: 采信的同时仍要记 source="vlm"、留原始返回、跑叠图核对。
    trust_vlm: bool = True

    # ---- VLM 端点
    provider: str = "deepseek"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""

    # ---- VLM 请求参数
    timeout: float = 120.0
    max_tokens: int = 4096
    #: 送进模型前把长边缩到这个值。模型那边反正会把整图降到约 384 token，
    #: 送 8000px 只是白花上传时间和内存。
    max_edge: int = 2048
    #: 电路图转网表是"照抄"任务，不是创作任务，温度必须压到 0。
    temperature: float = 0.0

    # ---- 本地层
    ocr: OcrConfig = field(default_factory=OcrConfig)
    #: 本地层算"成功"的置信度门槛，低于它就走 VLM（或报需人工确认）。
    #: 与 ir.model.CONFIDENCE_GATE 同值，但**语义不同、故意分开两个字段**：
    #: 那个是"单个拓扑判断能不能信"，这个是"整张图能不能免人工"。
    local_min_confidence: float = 0.85

    #: ★ 交给几何层之前，先把**文字区域涂白**。
    #: 默认为 True，而且不建议关掉：文字墨迹与元件墨迹在二值图上没有区别 ——
    #: 文字的「0」有自己的闭合空洞会被判成电容，几块字的墨迹会被当成"元件本体"
    #: 而它的 slot_region 盖住导线、把导线切成碎段，最后 IR 不连通。
    #: 而真实电路图几乎都是有标注的。关掉它只在"图上确认没有文字"时才有意义。
    #: 涂白不改任何坐标，而且擦了哪些区域会逐块写进 diagnostics，可核对。
    blank_text: bool = True

    #: ★ 升级到 VLM 的触发条件。
    #:   "manual"（**默认**）= 本地跑完就停，界面上由用户点按钮决定要不要交给模型。
    #:     理由见 VALID_ESCALATE 的注释：该不该升级取决于图源，机器分不出来，
    #:     而这次调用要花钱。
    #:   "structural" = 只有**结构性**失败才自动调模型：符号定位不到、导线追踪
    #:   不出、拓扑有歧义。这类问题本地做不出来，模型的空间理解能补。
    #:   非结构性失败（某电阻数值 OCR 读不清、位号认成 RI）本地照样出网表，
    #:   把那一格标 needs_human 交给用户手填 —— 不花这笔钱。
    #:   "any" = 任一不完美就调模型（最省心，但手绘图几乎每次都触发）。
    #:   "never" = 一律不调模型（等价于本地独占 + 人工兜底）。
    #: 依据：项目的「不许靠肉眼定连接」针对的是**连接**；数值读不清是
    #: 另一类问题，两者不该用同一条线卡。
    escalate_on: str = "manual"

    # ------------------------------------------------------------ 派生

    @property
    def api_key_present(self) -> bool:
        return bool(self.api_key.strip())

    def mask(self) -> str:
        """掩码。短 key 直接全遮，避免"掩码本身泄漏了大部分内容"。"""
        k = self.api_key.strip()
        if not k:
            return ""
        if len(k) <= 12:
            return "*" * 8
        return f"{k[:4]}{'*' * 8}{k[-4:]}"

    def chat_completions_url(self) -> str:
        """由 base_url 拼出 chat/completions 端点。

        容错三件套：补 ``/v1``（若用户只填了主机名）、去重复斜杠、
        去尾部斜杠。**不主动加 /v1** —— DeepSeek 两种都收，
        而有些自建端点加了反而 404，交给用户的 base_url 决定。
        """
        base = (self.base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not base:
            base = DEFAULT_BASE_URL
        if "://" not in base:
            base = "https://" + base
        return base + "/chat/completions"

    def validate(self) -> list[str]:
        """返回"能不能真的发起调用"层面的问题清单（空 = 可以）。

        这些**不是**配置读取警告，而是调用前的体检结果，措辞要能直接
        摆在界面上告诉用户"下一步该做什么"。
        """
        problems: list[str] = []
        if not self.enabled:
            problems.append("视觉通道总开关是关的。")
        if self.mode not in VALID_MODES:
            problems.append(
                f"mode = {self.mode!r} 不是合法值，只能是 {', '.join(VALID_MODES)}。")
        if self.escalate_on not in VALID_ESCALATE:
            problems.append(
                f"escalate_on = {self.escalate_on!r} 不是合法值，"
                f"只能是 {', '.join(VALID_ESCALATE)}。")
        # 只有真的会走到 VLM 才谈得上缺 key。mode=local_only 或
        # escalate_on=never 时，没 key 是完全正常的配置，不该报问题。
        will_call_vlm = self.enabled and (
            self.mode in ("local_first", "vlm_only") and self.escalate_on != "never")
        if will_call_vlm and not self.api_key_present:
            problems.append(
                "没有配置 VLM 的 api_key。可以在 WebUI 的「视觉设置」里填，"
                f"或写进 {display_path(SECRETS_LOCAL)}，"
                f"或设环境变量 {ENV_API_KEY}。"
                "（本地 OCR 层不需要 key，没配也能用。）")
        if will_call_vlm and not (self.model or "").strip():
            problems.append("没有配置模型名。DeepSeek 的视觉模型是 "
                            f"{DEFAULT_MODEL}，注意它带 -exp 后缀、属于实验性模型。")
        if will_call_vlm and not (self.base_url or "").strip():
            problems.append("没有配置 base_url，默认应为 " + DEFAULT_BASE_URL)
        return problems

    # ------------------------------------------------------------ 序列化

    def to_dict(self, *, mask: bool = True) -> dict[str, Any]:
        """转成可回前端的字典。**默认掩码**，要明文必须显式传 mask=False。"""
        return {
            "schema": SCHEMA_ID,
            "enabled": self.enabled,
            "mode": self.mode,
            "trust_vlm": self.trust_vlm,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.mask() if mask else self.api_key,
            "api_key_present": self.api_key_present,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "max_edge": self.max_edge,
            "temperature": self.temperature,
            "local_min_confidence": self.local_min_confidence,
            "escalate_on": self.escalate_on,
            "blank_text": self.blank_text,
            "ocr": self.ocr.to_dict(),
            "chat_url": self.chat_completions_url(),
            "valid_modes": list(VALID_MODES),
            "valid_escalate": list(VALID_ESCALATE),
            "presets": PRESETS,
            "default_model": DEFAULT_MODEL,
            "default_base_url": DEFAULT_BASE_URL,
            # ★ 走 display_path：配置目录被挪到项目外时也要能给出可读路径
            "secrets_path": display_path(SECRETS_LOCAL),
        }

    def to_file_dict(self) -> dict[str, Any]:
        """写盘用的形状（明文 key，嵌套在 'vision' 下）。"""
        return {
            "schema": SCHEMA_ID,
            "vision": {
                "enabled": self.enabled,
                "mode": self.mode,
                "trust_vlm": self.trust_vlm,
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_tokens": self.max_tokens,
                "max_edge": self.max_edge,
                "temperature": self.temperature,
                "local_min_confidence": self.local_min_confidence,
                "escalate_on": self.escalate_on,
                "blank_text": self.blank_text,
                "ocr": self.ocr.to_dict(),
            },
        }


@dataclass
class LoadedConfig:
    """配置 + 它从哪来的 + 读的过程中出的问题。

    刻意不把 warnings 塞进 VisionConfig：那样 `to_dict()` 就分不清
    "配置内容"和"读配置的过程"，而这两者对前端是两种提示。
    """

    config: VisionConfig = field(default_factory=VisionConfig)
    warnings: list[str] = field(default_factory=list)
    #: 字段名 -> "env" / "file" / "default"，用来回答"这个值哪来的"
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self, *, mask: bool = True) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(mask=mask),
            "warnings": list(self.warnings),
            "sources": dict(self.sources),
        }


# ---------------------------------------------------------------- 读


def _read_raw(path: Path, warn: list[str]) -> dict[str, Any]:
    """读 JSON。不存在 → 安静返回 {}；存在但坏了 → 警告 + 返回 {}。

    "文件不存在"不警告是有意的：那是全新安装的正常状态。
    "存在但坏了"必须警告：那意味着**用户的配置被忽略了**，
    如果悄悄用默认值跑，用户会以为自己的 key 生效了。
    """
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        warn.append(f"读取 {path.name} 失败：{e}。已改用内置默认值。")
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        warn.append(
            f"{path.name} 不是合法 JSON（第 {e.lineno} 行第 {e.colno} 列：{e.msg}），"
            "整份配置已被忽略、改用内置默认值。请修好它，"
            "否则你填的 key 不会生效 —— 而且界面上看起来会像「没配过」。")
        return {}
    if not isinstance(data, dict):
        warn.append(f"{path.name} 的顶层应该是对象，实际是 {type(data).__name__}，已忽略。")
        return {}
    return data


def load_config(path: Path | None = None) -> LoadedConfig:
    """读配置。**永不抛异常**，读不到就用默认值 + warnings。"""
    warn: list[str] = []
    p = path or SECRETS_LOCAL
    raw = _read_raw(p, warn)

    sect = raw.get("vision")
    if sect is None:
        sect = {}
    if not isinstance(sect, dict):
        warn.append("配置里的 vision 段应该是对象，已忽略、改用内置默认值。")
        sect = {}

    cfg = VisionConfig()
    src: dict[str, str] = {}

    def take(key: str, value: Any, setter) -> None:  # type: ignore[no-untyped-def]
        if value is not None:
            src[key] = "file"
            setter(value)

    take("enabled", sect.get("enabled"),
         lambda v: setattr(cfg, "enabled", _as_bool(v, cfg.enabled, warn, "vision.enabled")))
    take("mode", sect.get("mode"),
         lambda v: setattr(cfg, "mode", _as_str(v, cfg.mode, warn, "vision.mode")))
    take("trust_vlm", sect.get("trust_vlm"),
         lambda v: setattr(cfg, "trust_vlm", _as_bool(v, cfg.trust_vlm, warn, "vision.trust_vlm")))
    take("provider", sect.get("provider"),
         lambda v: setattr(cfg, "provider", _as_str(v, cfg.provider, warn, "vision.provider")))
    take("base_url", sect.get("base_url"),
         lambda v: setattr(cfg, "base_url", _as_str(v, cfg.base_url, warn, "vision.base_url")))
    take("model", sect.get("model"),
         lambda v: setattr(cfg, "model", _as_str(v, cfg.model, warn, "vision.model")))
    take("api_key", sect.get("api_key"),
         lambda v: setattr(cfg, "api_key", _as_str(v, cfg.api_key, warn, "vision.api_key", strip=False)))
    take("timeout", sect.get("timeout"),
         lambda v: setattr(cfg, "timeout", _as_float(v, cfg.timeout, warn, "vision.timeout", lo=1.0, hi=1800.0)))
    take("max_tokens", sect.get("max_tokens"),
         lambda v: setattr(cfg, "max_tokens", _as_int(v, cfg.max_tokens, warn, "vision.max_tokens", lo=256, hi=32768)))
    take("max_edge", sect.get("max_edge"),
         lambda v: setattr(cfg, "max_edge", _as_int(v, cfg.max_edge, warn, "vision.max_edge", lo=256, hi=8192)))
    take("temperature", sect.get("temperature"),
         lambda v: setattr(cfg, "temperature", _as_float(v, cfg.temperature, warn, "vision.temperature", lo=0.0, hi=2.0)))
    take("local_min_confidence", sect.get("local_min_confidence"),
         lambda v: setattr(cfg, "local_min_confidence",
                           _as_float(v, cfg.local_min_confidence, warn,
                                     "vision.local_min_confidence", lo=0.0, hi=1.0)))
    take("escalate_on", sect.get("escalate_on"),
         lambda v: setattr(cfg, "escalate_on", _as_str(v, cfg.escalate_on, warn, "vision.escalate_on")))
    # ★ blank_text 必须在这里认领。它曾经是"死开关"：dataclass 有字段、pipeline 读它、
    #   WebUI 有勾选框，但 to_dict / to_file_dict / save_config 三处都没有它 ——
    #   于是勾选框永远显示未勾（读不到值），取消勾选提交后**静默无效**。
    #   一个看起来能用、实际什么都不改的开关，比没有这个开关更坏。
    take("blank_text", sect.get("blank_text"),
         lambda v: setattr(cfg, "blank_text", _as_bool(v, cfg.blank_text, warn, "vision.blank_text")))

    ocr_raw = sect.get("ocr")
    if isinstance(ocr_raw, dict):
        cfg.ocr.enabled = _as_bool(ocr_raw.get("enabled"), cfg.ocr.enabled, warn, "vision.ocr.enabled")
        langs = ocr_raw.get("langs")
        if isinstance(langs, list) and all(isinstance(x, str) for x in langs):
            cfg.ocr.langs = [x.strip() for x in langs if x.strip()]
        elif langs is not None:
            warn.append("vision.ocr.langs 期望字符串数组，已忽略。")
        src["ocr"] = "file"
    elif ocr_raw is not None:
        warn.append("vision.ocr 期望对象，已忽略。")

    # ---- 环境变量覆盖（最高优先级）
    env_map = [
        (ENV_API_KEY, "api_key", lambda v: setattr(cfg, "api_key", v)),
        (ENV_BASE_URL, "base_url", lambda v: setattr(cfg, "base_url", v.strip())),
        (ENV_MODEL, "model", lambda v: setattr(cfg, "model", v.strip())),
    ]
    for env_name, field_name, setter in env_map:
        val = os.environ.get(env_name)
        if val is not None and val.strip() != "":
            setter(val)
            src[field_name] = "env"

    raw_timeout = os.environ.get(ENV_TIMEOUT)
    if raw_timeout is not None and raw_timeout.strip():
        cfg.timeout = _as_float(raw_timeout, cfg.timeout, warn, ENV_TIMEOUT, lo=1.0, hi=1800.0)
        src["timeout"] = "env"

    raw_mode = os.environ.get(ENV_MODE)
    if raw_mode is not None and raw_mode.strip():
        cfg.mode = raw_mode.strip()
        src["mode"] = "env"

    for f in ("enabled", "mode", "trust_vlm", "provider", "base_url", "model",
              "api_key", "timeout", "max_tokens", "max_edge", "temperature",
              "local_min_confidence", "escalate_on", "blank_text"):
        src.setdefault(f, "default")

    if cfg.mode not in VALID_MODES:
        warn.append(f"vision.mode = {cfg.mode!r} 不是合法值，已退回 local_first。"
                    f"（合法值：{', '.join(VALID_MODES)}）")
        cfg.mode = "local_first"
        src["mode"] = "default"

    if cfg.escalate_on not in VALID_ESCALATE:
        warn.append(f"vision.escalate_on = {cfg.escalate_on!r} 不是合法值，"
                    f"已退回 structural。（合法值：{', '.join(VALID_ESCALATE)}）")
        cfg.escalate_on = "structural"
        src["escalate_on"] = "default"

    return LoadedConfig(config=cfg, warnings=warn, sources=src)


# ---------------------------------------------------------------- 写


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """先写临时文件再替换 —— 中途断电也不会留下半截 JSON。

    半截 JSON 的后果很严重：下次启动会走进"存在但坏了"分支，
    用户的 key 被整份忽略，而界面看起来只是"key 没配"。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def save_config(updates: dict[str, Any], path: Path | None = None) -> LoadedConfig:
    """把 ``updates`` 合并进配置文件并写盘，返回重新读出来的配置。

    合并规则（重要）：
    - 只认 ``vision`` 段里的已知字段，未知字段**保留**，不删用户手写的东西；
    - ``api_key`` 传空串 / 传掩码 / 不传 → **保持原值**。这是为了防
      "前端拿掩码回填 → 掩码被存成真 key → key 被悄悄清掉"这个经典事故；
    - 写盘后重新 load，让调用方拿到的一定是"盘上真实生效的东西"。
    """
    p = path or SECRETS_LOCAL
    warn: list[str] = []
    existing = _read_raw(p, warn)
    sect = existing.get("vision")
    if not isinstance(sect, dict):
        sect = {}
    sect = dict(sect)                       # 保留未知字段

    current = load_config(p).config
    known = {
        "enabled": "bool", "mode": "str", "trust_vlm": "bool", "provider": "str",
        "base_url": "str", "model": "str", "timeout": "float", "max_tokens": "int",
        "max_edge": "int", "temperature": "float",
        "local_min_confidence": "float", "escalate_on": "str",
        "blank_text": "bool",
    }
    changed: list[str] = []
    for k, kind in known.items():
        if k not in updates or updates[k] is None:
            continue
        v = updates[k]
        if kind == "bool":
            nv: Any = _as_bool(v, getattr(current, k), warn, f"vision.{k}")
        elif kind == "float":
            nv = _as_float(v, getattr(current, k), warn, f"vision.{k}")
        elif kind == "int":
            nv = _as_int(v, getattr(current, k), warn, f"vision.{k}")
        else:
            nv = _as_str(v, getattr(current, k), warn, f"vision.{k}")
        if nv != getattr(current, k):
            changed.append(k)
        sect[k] = nv

    if "api_key" in updates:
        raw_key = updates.get("api_key")
        if isinstance(raw_key, str):
            new_key = raw_key.strip()
            if new_key == "":
                pass                        # 空串 = 不动
            elif new_key == current.mask():
                pass                        # 掩码 = 不动（前端回填）
            elif set(new_key) == {"*"}:
                pass                        # 纯星号 = 不动
            else:
                if new_key != current.api_key:
                    changed.append("api_key")
                sect["api_key"] = new_key

    ocr_upd = updates.get("ocr")
    if isinstance(ocr_upd, dict):
        ocr_sect = sect.get("ocr")
        if not isinstance(ocr_sect, dict):
            ocr_sect = {}
        ocr_sect = dict(ocr_sect)
        if "enabled" in ocr_upd and ocr_upd["enabled"] is not None:
            ocr_sect["enabled"] = _as_bool(ocr_upd["enabled"], current.ocr.enabled,
                                           warn, "vision.ocr.enabled")
            changed.append("ocr.enabled")
        langs = ocr_upd.get("langs")
        if isinstance(langs, list):
            ocr_sect["langs"] = [str(x).strip() for x in langs if str(x).strip()]
            changed.append("ocr.langs")
        sect["ocr"] = ocr_sect

    existing["schema"] = existing.get("schema") or SCHEMA_ID
    existing["vision"] = sect
    _atomic_write_json(p, existing)

    loaded = load_config(p)
    loaded.warnings = warn + loaded.warnings
    if changed:
        loaded.warnings.append("已写入配置：" + "、".join(sorted(set(changed))) + "。")
    return loaded


def ensure_example(path: Path | None = None) -> Path:
    """确保模板文件存在（内容永远是"没填 key"的样子）。"""
    p = path or SECRETS_EXAMPLE
    if not p.is_file():
        _atomic_write_json(p, VisionConfig(api_key="").to_file_dict())
    return p


def touch_local_template(path: Path | None = None) -> Path | None:
    """首次运行时铺一份只有结构、没有 key 的 secrets.local.json。

    只在文件**完全不存在**时铺，绝不覆盖已有内容。
    """
    p = path or SECRETS_LOCAL
    if p.is_file():
        return None
    _atomic_write_json(p, VisionConfig(api_key="").to_file_dict())
    return p
