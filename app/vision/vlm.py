"""DeepSeek 视觉模型客户端：把一张电路位图交给 VLM，拿回结构化网表。

**为什么单独一个模块**：这一段是全链路唯一"不确定、要花钱、会超时、可能被
服务端改口径"的部分。把它关在一个模块里，好处是上层（编排层）可以完全
不关心 HTTP，只关心"成没成、为什么不成"。

设计要点，逐条都对应一次真实踩坑或一次明确约定：

1. **图片只能进 user 消息。** 放进 system / assistant 会被 400。
2. **模型名不许硬编码。** 默认值来自 ``config.DEFAULT_MODEL``
   （``deepseek-v4-flash-vision-exp``，带 ``-exp`` 后缀 = 实验性，
   随时可能下线或改名）。这里只引用，不复制第二份。
3. **"这个模型不收图"要单独成一类错误。** DeepSeek 下只有带 vision 能力的
   模型收图片，其余一律返回 400 ``This model does not support image``。
   如果把它归到"通用 400"，用户会去查图片格式，而真正的问题是模型名填错了。
4. **先量体裁衣再上传。** 模型侧反正会把整图降到约 800×800
   （约 384 token），送 6000px 原图只是白花上传时间。
5. **限值是硬数字，超了要在本地拦住**：单图 32 MiB、请求体 48 MiB、单边 8192px。
   与其让服务端返回一个看不懂的错，不如本地直接说清是哪个限值超了。

关于第 4 条的一个补充：线稿图和照片要用**不同的编码**。照片走 JPEG 质量 90；
线稿（颜色种类很少）走 PNG，既无损又往往更小 —— 电路图大多是后者。
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, VisionConfig, load_config

# ---------------------------------------------------------------- 硬限值
#
# 这些是**服务端**的限值（来自官方文档），不是我们自己定的口味。
# 本地先判一遍的目的只有一个：让"超限"这条错误的措辞由我们控制。

#: 单张图原始字节上限（base64 内联或 URL 下载后）
MAX_IMAGE_BYTES = 32 * 1024 * 1024

#: 整个请求体上限。base64 会把原始字节撑大约 4/3，所以单图能真送出去的
#: 上限其实比 32 MiB 小 —— 这里两个都查，才能给出准确的提示。
MAX_BODY_BYTES = 48 * 1024 * 1024

#: 单边像素上限。超了服务端会拒。
MAX_SIDE_PX = 8192

#: 模型侧对每张图的 token 占用有个天花板（约 800×800 像素对应 384 token）。
#: 用在界面上显示"这张图大概花多少" —— 它是**估算**，不参与任何判断。
IMAGE_TOKEN_ESTIMATE = 384

#: 送模型前长边缩到这个值。2048 远高于模型实际采样分辨率，绰绰有余；
#: 再大只是浪费上传时间。
DEFAULT_MAX_EDGE = 2048

JPEG_QUALITY = 90

#: 颜色数少于这个阈值就认为是线稿，改用 PNG。电路图几乎都落在这边。
LINEWORK_COLOR_LIMIT = 512


# ---------------------------------------------------------------- 异常
#
# 分这么细不是为了好看，是为了让**每一类错误都能对应一句可执行的建议**。
# "调用失败" 这种话对用户毫无价值。


class VlmError(Exception):
    """视觉模型调用相关的错误基类。"""

    #: 给前端分类用；也方便测试断言"错得对不对"
    kind = "vlm_error"

    def __init__(self, message: str, *, hint: str = "", raw: Any = None) -> None:
        super().__init__(message)
        self.message = message
        #: 「你该做什么」—— 每一类错误都必须给出这一句
        self.hint = hint
        self.raw = raw

    def to_dict(self) -> dict[str, Any]:
        d = {"kind": self.kind, "error": self.message}
        if self.hint:
            d["hint"] = self.hint
        if self.raw is not None:
            d["raw"] = self.raw
        return d


class VlmNotConfigured(VlmError):
    """没配 key / 没配模型 —— 这不是故障，是"还没配"。"""

    kind = "not_configured"


class VlmTransportError(VlmError):
    """连不上：DNS、连接被拒、超时。"""

    kind = "transport"


class VlmModelRejectsImage(VlmError):
    """模型不收图。★ 十有八九是模型名填错了，而不是图片有问题。"""

    kind = "model_rejects_image"


class VlmAuthError(VlmError):
    """401/403：key 不对、过期、或没权限。"""

    kind = "auth"


class VlmRateLimited(VlmError):
    kind = "rate_limited"


class VlmHttpError(VlmError):
    """其它非 2xx。"""

    kind = "http"

    def __init__(self, message: str, *, status: int = 0, hint: str = "",
                 raw: Any = None) -> None:
        super().__init__(message, hint=hint, raw=raw)
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["status"] = self.status
        return d


class VlmBadResponse(VlmError):
    """HTTP 通了，但返回体不是我们能用的形状（截断、空、不是 JSON）。"""

    kind = "bad_response"


# ---------------------------------------------------------------- 图


@dataclass
class PreparedImage:
    """已经"量体裁衣"好的待发图。"""

    data: bytes
    mime: str
    width: int
    height: int
    #: 原始尺寸，用于报告"我们把 4000×3000 降到了 2048×1536"
    original_width: int = 0
    original_height: int = 0
    #: 编码方式的说明，进 warnings 让用户知道图被改过
    encode_note: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def byte_size(self) -> int:
        return len(self.data)

    @property
    def downscaled(self) -> bool:
        return (self.width, self.height) != (self.original_width, self.original_height)

    def to_data_url(self) -> str:
        b64 = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime};base64,{b64}"

    def summary(self) -> str:
        s = f"{self.width}×{self.height} {self.mime} {self.byte_size / 1024:.0f} KiB"
        if self.downscaled:
            s += f"（原图 {self.original_width}×{self.original_height}，已缩小）"
        return s


def _encode_best(im: Any) -> tuple[bytes, str, str]:
    """按图像性质选编码。返回 ``(bytes, mime, 说明)``。"""
    from PIL import Image

    # 线稿判定：转成 RGB 后数颜色。电路图（白底黑线 + 少量彩色标注）
    # 颜色数很少；照片则极多。这个判据简单但够用，而且偏向"宁可走 JPEG"
    # 也不会出错 —— 出错的最坏后果只是文件大一点。
    probe = im if im.mode in ("RGB", "L") else im.convert("RGB")
    small = probe.copy()
    small.thumbnail((256, 256))
    colors = small.getcolors(maxcolors=1 << 20)
    n_colors = len(colors) if colors is not None else 1 << 20

    if n_colors <= LINEWORK_COLOR_LIMIT:
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
        # 偶尔 PNG 反而更大（大面积渐变），那时退回 JPEG
        buf2 = io.BytesIO()
        im.convert("RGB").save(buf2, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        jpg = buf2.getvalue()
        if len(png) <= len(jpg):
            return png, "image/png", f"线稿（{n_colors} 色）→ PNG 无损，{len(png) / 1024:.0f} KiB"
        return jpg, "image/jpeg", (f"线稿但 PNG 偏大（{len(png) / 1024:.0f} KiB）"
                                   f"→ 改用 JPEG，{len(jpg) / 1024:.0f} KiB")

    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), "image/jpeg", f"彩色/照片（{n_colors} 色）→ JPEG q{JPEG_QUALITY}"


def prepare_image(src: Any, *, max_edge: int = DEFAULT_MAX_EDGE) -> PreparedImage:
    """读图 → 按 EXIF 摆正 → 缩到长边 <= max_edge → 重新编码。

    ``src`` 可以是**文件路径 / bytes / PIL.Image**（三者同权）。
    ★ 必须收 PIL.Image：本地那一级（``ocr.recognize`` / ``detect_symbols`` /
    ``build_wire_graph``）三种都收，于是管线里流下来的 ``image`` 本来就可能
    是一个 PIL 对象；如果这一层只认"路径/bytes"，那条升级路径就会以一个
    ``TypeError`` 死掉，而错误信息会被包成"这张图读不开" —— 把
    **类型不认识**说成**文件坏了**，让人抱着一个完好的文件反复另存。
    这类"错误信息指向错误的排查方向"的坑，本项目一律算 bug。

    **不抛 VlmError**，坏图抛 ``VlmBadResponse``（因为它确实是一种
    "送不到模型那儿"的失败，且 hint 很明确）。

    关于 EXIF：手机拍的照片常常带旋转标记。PIL 默认**不**应用它，
    于是竖着拍的照片会以横躺的姿态送进模型。这个坑非常隐蔽 ——
    在本地看图是正的（看图软件会应用 EXIF），只有模型看到的是躺的。
    """
    from PIL import Image, ImageOps

    if isinstance(src, Image.Image):
        # 拷一份：下面会就地 convert/resize，不该改到调用方手里的对象
        im = src.copy()
    elif isinstance(src, (bytes, bytearray, memoryview)):
        try:
            im = Image.open(io.BytesIO(bytes(src)))
            im.load()
        except Exception as e:                 # noqa: BLE001
            raise VlmBadResponse(
                f"这张图（bytes）读不开：{type(e).__name__}: {e}",
                hint="确认文件没损坏、扩展名和真实格式一致（本工具按文件内容判格式，"
                     "不按扩展名）。支持的格式：JPEG / PNG / GIF / WebP / BMP / TIFF。",
            ) from None
    elif isinstance(src, (str, Path)):
        try:
            im = Image.open(src)
            im.load()
        except Exception as e:                 # noqa: BLE001
            raise VlmBadResponse(
                f"这张图（{Path(src).name if str(src) else str(src)}）读不开："
                f"{type(e).__name__}: {e}",
                hint="确认文件没损坏、扩展名和真实格式一致（本工具按文件内容判格式，"
                     "不按扩展名）。支持的格式：JPEG / PNG / GIF / WebP / BMP / TIFF。",
            ) from None
    else:
        raise VlmBadResponse(
            f"不认识的图片来源：{type(src).__name__}。"
            "★ 这一条与图片本身无关 —— 本函数收「文件路径 / bytes / PIL.Image」三种，"
            "传进来的却是别的东西。",
            hint="把图片读成上述三种之一再传进来；若是从别处拿到的对象，"
                 "先确认它到底是不是一张图。",
        )

    ow, oh = im.size
    warns: list[str] = []

    # EXIF 旋转：必须做，否则竖拍的照片模型看到的是躺着的
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:                          # noqa: BLE001
        warns.append("EXIF 方向信息读取失败，按原始方向处理 —— "
                     "如果模型把图看歪了，请先手动摆正再上传。")
    if im.size != (ow, oh):
        warns.append(f"按 EXIF 方向标记旋转过（{ow}×{oh} → {im.size[0]}×{im.size[1]}）。")

    # 透明底：JPEG 不支持 alpha，直接转 RGB 会把透明区变黑。
    # 复合到白底上 —— 电路图几乎都是白底黑线，白底是对的。
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
        warns.append("原图带透明通道，已复合到白底上（否则透明区会变成黑块）。")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    # 缩放到长边 <= max_edge
    long_edge = max(im.size)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_size = (max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale)))
        im = im.resize(new_size, Image.LANCZOS)

    data, mime, note = _encode_best(im)

    prep = PreparedImage(
        data=data, mime=mime, width=im.size[0], height=im.size[1],
        original_width=ow, original_height=oh,
        encode_note=note, warnings=warns,
    )

    # 单边超限：缩过之后一般不会超，但 max_edge 本身可能被配成 > 8192
    if max(prep.width, prep.height) > MAX_SIDE_PX:
        raise VlmBadResponse(
            f"图片单边 {max(prep.width, prep.height)}px 超过模型的 {MAX_SIDE_PX}px 上限，"
            f"而当前 max_edge={max_edge} 没有把它压下来。",
            hint=f"把视觉设置里的 max_edge 调到 {MAX_SIDE_PX} 或更小。",
        )
    if prep.byte_size > MAX_IMAGE_BYTES:
        raise VlmBadResponse(
            f"编码后仍有 {prep.byte_size / 1024 / 1024:.1f} MiB，"
            f"超过单图 {MAX_IMAGE_BYTES // 1024 // 1024} MiB 上限。",
            hint="把 max_edge 调小，或先把原图裁掉无关区域。",
        )
    return prep


def check_body_size(prep: PreparedImage) -> None:
    """base64 会撑大约 4/3，请求体上限比单图上限更早触发。"""
    approx = prep.byte_size * 4 // 3 + 4096
    if approx > MAX_BODY_BYTES:
        raise VlmBadResponse(
            f"这张图 base64 编码后请求体约 {approx / 1024 / 1024:.1f} MiB，"
            f"超过请求体 {MAX_BODY_BYTES // 1024 // 1024} MiB 上限。",
            hint="把 max_edge 调小（例如 1600），或先裁剪原图。",
        )


# ---------------------------------------------------------------- 提示词
#
# ★ 这一段是"契约"，不是"建议"。模型输出的每个字段都要被程序解析，
#   所以措辞上宁可啰嗦，也不能留解释空间。
#
# 几条刻意的设计：
# - 反复强调"图里没画的不要补"。模型见过太多教科书电路，会本能地
#   把"应该有的"补进去，而这恰好破坏了这个项目最看重的东西：
#   只报告看到的事实。
# - 数值要求**照抄图上印的字**（含单位），不要做单位换算、不要补精度。
#   因为 app/ingest/values.py 就是按工程记法解析的，照抄能直接对上；
#   模型一旦"帮忙换算成科学计数法"，反而可能引入量级错误。
# - 看不清的数值要求写 `"?"`。这个记号在 values.py 里已有明确语义
#   （解析成 None + 「需人工确认」），是最合适的落点。
# - 电压源极性必须显式约定，因为 IR 的约定是项目内部的，模型不可能知道。

CONTRACT_SCHEMA = "circuit_agent.netlist/1"

SYSTEM_PROMPT = """\
你是电路图读图器。你的输出会被程序逐字段解析成电路网表。

你没有"补全电路"的任务。你只读图：
- 只报告图里**确实画出来**的东西。
- 图里没画的元件绝不写进去，哪怕你认为这张电路少了个电阻就不成立。
- 看不清就如实说看不清，不要挑一个"看起来像对的"填上。
- 不要做单位换算，不要把数值改成科学计数法，照抄图上印的字。

你默认读者是工程师，他会拿你的输出直接算题。一个编出来的数字比一个
诚实的问号有害得多。"""


def build_user_prompt(*, extra: str = "") -> str:
    """拼出 user 消息里的文字部分（图片以另一段 content 附在同一消息里）。"""
    return f"""\
请读这张电路图，输出**一个 JSON 对象**，不要有任何解释文字、不要用 Markdown 代码块
包裹。顶层结构如下：

{{
  "schema": "{CONTRACT_SCHEMA}",
  "reference_node": "0",
  "reference_evidence": "为什么认为这个节点是参考地（有无接地符号、符号长什么样）",
  "nodes": [
    {{"id": "1", "where": "用文字描述这个结点在图里的位置",
      "evidence": "它为什么是一个独立结点（实心圆点 / 导线拐角 / 三线相交 / 被元件隔开）",
      "confidence": 0.9}}
  ],
  "components": [
    {{"ref": "V1", "kind": "V", "value": "12V", "nodes": ["1", "0"],
      "polarity_note": "电压源：哪个端口是 + 极，图上是怎么标记的",
      "box": [0.05, 0.10, 0.12, 0.08],
      "confidence": 0.95, "where": "左上角"}}
  ],
  "crossings": [
    {{"where": "图里的位置", "connected": false,
      "why": "有无实心圆点 / 是否 T 形接入", "confidence": 0.8}}
  ],
  "unsupported": [
    {{"kind": "受控源 / 开关 / 变压器 / 二极管 / 其它",
      "where": "位置", "desc": "看到什么", "confidence": 0.7}}
  ],
  "unreadable": [
    {{"what": "哪个元件的哪一项", "where": "位置",
      "guessed": "可能是什么（可留空）"}}
  ],
  "warnings": ["任何你觉得读图结果可疑的地方，都写在这里"]
}}

字段规则（每条都要遵守）：

1. **kind 只能是 R / V / I / C / L 五种之一**
   - R 电阻、V 理想电压源、I 理想电流源、C 电容、L 电感。
   - 任何其它元件（受控源、开关、二极管、变压器、运放……）**不要**塞进
     components，写进 unsupported，并说明你看到了什么。
   - 注意：受控源（菱形/菱形带箭头）**不是**独立源，别写成 V 或 I。

2. **value 照抄图上印的字，含单位**
   - 正确：`"12V"`、`"4.7k"`、`"100uF"`、`"2.2mH"`、`"6"`、`"4k7"`。
   - 数值真的看不清就写 `"?"`，不要猜、不要写 `"10k?"` 这种混合形式。
   - 图上只画了符号没标数值，value 写 `"?"`。

3. **nodes 的顺序有语义，必须按下面来**
   - 一般元件（R / I / C / L）：`nodes` 是两端，顺序无所谓，但必须写两端。
   - **电压源 V：`nodes[0]` 必须是标了 `+` 的那一端，`nodes[1]` 是 `−` 端。**
     这条是本工具的硬约定，后续计算全依赖它。找不到极性标记就在
     polarity_note 里说明，并把 confidence 降到 0.5 以下。

4. **结点要分开写清哪些是同一个结点**
   - `nodes` 里用你自己起的 id 字符串（"0"、"1"、"2"…），同一个 id
     表示电气上同一点。**参考地必须叫 "0"。**
   - 只靠符号图形隔开、电气上并不相连的两个点，必须是不同的 id。
   - 电阻框、电容极板、电感线圈是**元件本体**，它两侧是不同结点。
     不要因为两个点离得近就合并它们。

5. **导线交叉 vs T 形接入，必须区分开**
   - 十字交叉且**没有**实心圆点 → 不相连，两条线各走各的。
   - 交叉处**有**实心圆点 → 相连。
   - T 形接入（一条线端点落在另一条线中段）→ 通常相连，即使没有圆点。
   - 每遇到这种地方都写进 crossings，宁可多报。

6. **box 是**粗定位**，不必精确**
   - 格式 `[x, y, w, h]`，都是 0~1 的归一化值，原点在左上角。
   - 这只是用来在界面上叠图核对的提示，**不参与计算**，误差大没关系。
   - 不确定就给个大致位置；实在不知道就省略这个字段。

7. **confidence 是 0~1 的小数**，表示你对该条目的确定程度。
   - 图上清清楚楚 = 0.9 以上；需要推断 = 0.6~0.8；看不清 = 0.5 以下。
   - 不要所有条目都给同一个值，那没有信息量。

8. **reference_node**：图里有接地符号就把那个结点定为 "0"。
   没有接地符号时，选你觉得最合理的那个（通常是连线最多的回流节点），
   并在 reference_evidence 里说明这是你选的、不是图上标的。

9. 如果整张图根本不像电路图，或完全无法辨认，就返回：
   `{{"schema": "{CONTRACT_SCHEMA}", "failed": true, "reason": "..."}}`

只输出 JSON。{extra}"""


# ---------------------------------------------------------------- JSON 提取


def _strip_code_fence(text: str) -> str:
    """剥掉 ```` ```json ... ``` ```` 围栏。

    明明要求"不要用代码块"，但模型十次里总有一两次照旧包。这个不值得
    跟模型较劲，本地剥掉就是了 —— 它不影响正确性，只影响解析。
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _balanced_json(text: str) -> str | None:
    """从文本里扫出第一个**括号配平**的 JSON 对象。

    比 `rfind("}")` 可靠：模型有时会在 JSON 后面附一段说明，
    里面也可能带 `}`。括号配平 + 跳过字符串内的括号，能稳定切出来。
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """从模型回复里取出 JSON 对象。取不出来就抛 ``VlmBadResponse``。

    三级尝试：直接解析 → 剥围栏后解析 → 括号配平切出来再解析。
    多余的温和处理是值得的：多试两次的成本是零，而失败一次的成本是
    整个请求的钱白花 + 用户要重来一次。
    """
    if not text or not text.strip():
        raise VlmBadResponse(
            "模型返回了空内容。",
            hint="偶发情况，重试一次看看。若反复出现，把 max_tokens 调大些"
                 "（推理过程的文字也算在里面）。",
        )
    candidates = [text.strip(), _strip_code_fence(text)]
    balanced = _balanced_json(_strip_code_fence(text))
    if balanced:
        candidates.append(balanced)
    last_err: Exception | None = None
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(obj, dict):
            return obj
        last_err = ValueError(f"顶层是 {type(obj).__name__}，不是对象")
    raise VlmBadResponse(
        f"模型返回的内容不是合法 JSON 对象：{last_err}",
        hint="这通常意味着模型没照契约输出（回复被 max_tokens 截断，"
             "或者它把 JSON 写坏了）。原始回复已随报告附上，可据此判断。",
        raw={"text_head": text[:2000]},
    )


# ---------------------------------------------------------------- HTTP


def _classify_http(status: int, body_text: str, payload: Any) -> VlmError:
    """把一次失败的 HTTP 响应翻成"一句人话 + 一句该做什么"。"""
    # 尽量从返回体里挖出服务端自己给的消息，别用我们猜的
    msg = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "")
        elif isinstance(err, str):
            msg = err
        msg = msg or str(payload.get("message") or "")
    msg = msg or (body_text or "").strip()[:500] or f"HTTP {status}"

    low = msg.lower()

    # ★ 这一类必须排在通用 400 前面，否则会被归错
    if "does not support image" in low or "not support image" in low:
        return VlmModelRejectsImage(
            f"模型不接受图片输入：{msg}",
            hint="这是模型选错了，不是图片有问题。DeepSeek 下只有带视觉能力的模型"
                 f"能收图，请把「视觉设置」里的模型名改成 {DEFAULT_MODEL}"
                 "（注意它带 -exp 后缀），或者换一个你确认支持图片的模型。",
            raw=payload if payload is not None else body_text[:500],
        )

    if status in (401, 403):
        return VlmAuthError(
            f"鉴权失败（HTTP {status}）：{msg}",
            hint="检查 api_key 是否填对、是否已过期，或是否被在控制台重置过。"
                 "注意环境变量 CIRCUIT_AGENT_VLM_API_KEY 的优先级高于配置文件 —— "
                 "如果环境变量里有一份旧 key，改配置文件是不会生效的。",
            raw=payload if payload is not None else body_text[:500],
        )

    if status == 402:
        return VlmHttpError(
            f"服务端返回 402：{msg}",
            hint="通常是账户余额不足。请到服务商控制台确认。",
            status=status, raw=payload if payload is not None else body_text[:500])

    if status == 429:
        return VlmRateLimited(
            f"被限流（HTTP 429）：{msg}",
            hint="等一会儿再试。如果经常发生，说明这个 key 的并发/额度不够。",
            raw=payload if payload is not None else body_text[:500])

    if status >= 500:
        return VlmHttpError(
            f"服务端错误（HTTP {status}）：{msg}",
            hint="这不是你的问题。等一会儿重试；持续失败就要看服务商的状态页了。",
            status=status, raw=payload if payload is not None else body_text[:500])

    if status == 400:
        return VlmHttpError(
            f"请求被拒（HTTP 400）：{msg}",
            hint="常见原因：模型名不存在、response_format 不被该模型支持、"
                 "或图片格式/尺寸超限。原始返回已附在报告里。",
            status=status, raw=payload if payload is not None else body_text[:500])

    return VlmHttpError(f"HTTP {status}：{msg}", status=status,
                        hint="原始返回已附在报告里，可据此进一步判断。",
                        raw=payload if payload is not None else body_text[:500])


@dataclass
class VlmResult:
    """一次成功调用的完整记录。

    ``raw`` 保留服务端的原始返回体 —— 这是"不静默降级"在视觉通道上的
    落实：用户可以自己去看模型到底说了什么，而不是只能相信我们转述的结论。
    """

    ok: bool
    text: str = ""
    payload: dict[str, Any] | None = None
    model: str = ""
    url: str = ""
    elapsed: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    image: PreparedImage | None = None
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model": self.model,
            "url": self.url,
            "elapsed": self.elapsed,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "image": ({"width": self.image.width, "height": self.image.height,
                       "mime": self.image.mime, "bytes": self.image.byte_size,
                       "original_width": self.image.original_width,
                       "original_height": self.image.original_height,
                       "encode_note": self.image.encode_note}
                      if self.image else None),
            "warnings": list(self.warnings),
            "text": self.text,
            "payload": self.payload,
            "raw": self.raw,
        }


def call_vision(
    image: Any,
    *,
    cfg: VisionConfig | None = None,
    prompt_extra: str = "",
    system_prompt: str | None | bool = None,
    max_edge: int | None = None,
) -> VlmResult:
    """调一次视觉模型。失败抛 ``VlmError`` 子类（**不返回 None**）。

    为什么失败要抛异常而不是返回 ``ok=False``：这条链路上"失败"有六种
    截然不同的原因，每一种对应一句不同的建议。抛异常能强制调用方处理它，
    而返回 False 很容易被一句 ``if not r.ok: fallback()`` 糊过去。

    ``image`` 可以是 文件路径 / bytes / PIL.Image，也可以直接给
    ``PreparedImage``，这样编排层可以先做预处理（去水印、增强）再用它发起调用。
    """
    loaded = load_config() if cfg is None else None
    c = cfg or loaded.config                    # type: ignore[union-attr]
    warns: list[str] = []
    if loaded is not None:
        warns.extend(loaded.warnings)

    if not c.enabled:
        raise VlmNotConfigured("视觉通道总开关是关的（vision.enabled = false）。",
                               hint="在「视觉设置」里打开，或把 config 里的 enabled 改成 true。")
    problems = c.validate()
    if problems:
        raise VlmNotConfigured("VLM 配置不完整：" + "；".join(problems),
                               hint="按上面每一条补齐即可。")

    if isinstance(image, PreparedImage):
        prep = image
    else:
        prep = prepare_image(image, max_edge=max_edge or c.max_edge)
    check_body_size(prep)

    messages: list[dict[str, Any]] = []
    # system_prompt 的三态：None=用默认；False=完全不要 system 消息；
    # 字符串=用给定的。留一个"不要 system"的口子，是因为有些兼容端点
    # 对 system 角色挑剔，排查时能一步排除它。
    if system_prompt is None:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    elif system_prompt is not False:
        messages.append({"role": "system", "content": system_prompt})
    # ★ 图片只能出现在 user 消息里。放进 system/assistant → 400。
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": prep.to_data_url()}},
            {"type": "text", "text": build_user_prompt(extra=prompt_extra)},
        ],
    })

    body: dict[str, Any] = {
        "model": c.model,
        "messages": messages,
        "max_tokens": c.max_tokens,
        "temperature": c.temperature,
        "stream": False,
    }

    url = c.chat_completions_url()
    payload, resp_meta = _post(c, url, body, warns)

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VlmBadResponse(
            "服务端返回体里没有 choices 字段。",
            hint="这不像是一次正常的 chat/completions 响应。原始返回已附上。",
            raw=payload)
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    text = msg.get("content")
    if not isinstance(text, str):
        # content 可能是列表（多模态回复）或 None（纯推理无输出）
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text
                           if isinstance(part, dict))
        else:
            text = ""
    finish = first.get("finish_reason")

    result = VlmResult(
        ok=True, text=text, model=str(payload.get("model") or c.model), url=url,
        elapsed=resp_meta["elapsed"], usage=payload.get("usage") or {},
        finish_reason=finish if isinstance(finish, str) else None,
        image=prep, warnings=warns + list(prep.warnings),
        raw={"response_meta": resp_meta, "choices": choices[:1]},
    )

    if finish == "length":
        result.warnings.append(
            "模型回复被 max_tokens 截断（finish_reason=length），"
            "JSON 很可能不完整。请把「视觉设置」里的 max_tokens 调大后重试。"
            "（推理型模型的思考过程也计入 max_tokens。）")

    try:
        result.payload = extract_json(text)
    except VlmBadResponse as e:
        # 把原始 text 挂到异常上再抛 —— 上层要能把"模型到底说了什么"给用户看
        e.raw = {"text": text[:8000], "finish_reason": finish}
        raise
    return result


def _post(c: VisionConfig, url: str, body: dict[str, Any],
          warns: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """发请求，**只发一次**。

    为什么不带 ``response_format={"type": "json_object"}`` 再自动重试：
    那个参数不是所有模型都支持，而"不支持"的表现是 400 —— 于是本地就
    多了一条"先失败一次再成功"的路径。由于本轮无法真机验证它到底支不支持，
    这里选择**不赌**：不发这个参数，改为在 ``extract_json`` 里用三级
    容错把 JSON 从回复里挖出来。挖取的代价是零，赌错的代价是一次调用。
    """
    try:
        import httpx
    except ImportError as e:                   # pragma: no cover
        raise VlmTransportError(
            f"缺少 HTTP 库：{e}",
            hint="在项目环境里执行 pip install httpx。") from None

    headers = {
        "Authorization": f"Bearer {c.api_key.strip()}",
        "Content-Type": "application/json",
    }

    def once(payload_body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.time()
        try:
            with httpx.Client(timeout=c.timeout) as client:
                r = client.post(url, headers=headers, json=payload_body)
        except httpx.TimeoutException as e:
            raise VlmTransportError(
                f"请求超时（{c.timeout:g} 秒）：{e}",
                hint="模型在读图，慢是正常的。可以在「视觉设置」里把 timeout 调大，"
                     "或者把 max_edge 调小以减少上传量。") from None
        except httpx.HTTPError as e:
            raise VlmTransportError(
                f"连不上 {url}：{type(e).__name__}: {e}",
                hint="检查网络与 base_url 是否正确、是否需要代理。",
            ) from None
        elapsed = time.time() - t0

        if r.status_code != 200:
            try:
                body_json: Any = r.json()
            except Exception:                  # noqa: BLE001
                body_json = None
            raise _classify_http(r.status_code, r.text, body_json)
        try:
            data = r.json()
        except Exception as e:                 # noqa: BLE001
            raise VlmBadResponse(
                f"服务端返回的不是 JSON：{e}。前 300 字：{r.text[:300]!r}",
                hint="可能命中了代理/网关的错误页，检查 base_url。") from None
        if not isinstance(data, dict):
            raise VlmBadResponse(f"服务端返回的顶层是 {type(data).__name__}，不是对象。",
                                 raw=data)
        return data, {"elapsed": elapsed, "status": r.status_code}

    return once(body)


def probe(cfg: VisionConfig | None = None) -> dict[str, Any]:
    """给 ``/api/health`` 用的体检报告。**绝不发起网络请求**，也绝不回明文 key。"""
    loaded = load_config() if cfg is None else None
    c = cfg or loaded.config                    # type: ignore[union-attr]
    problems = c.validate()
    return {
        "enabled": c.enabled,
        "mode": c.mode,
        "escalate_on": c.escalate_on,
        "provider": c.provider,
        "model": c.model,
        "base_url": c.base_url,
        "chat_url": c.chat_completions_url(),
        "api_key_present": c.api_key_present,
        "api_key_masked": c.mask(),
        "ocr_enabled": c.ocr.enabled,
        "ocr_langs": list(c.ocr.langs),
        "max_edge": c.max_edge,
        "timeout": c.timeout,
        "configured": not problems,
        "problems": problems,
        "config_warnings": (loaded.warnings if loaded else []),
        "sources": (loaded.sources if loaded else {}),
        "image_token_estimate": IMAGE_TOKEN_ESTIMATE,
        "limits": {"image_bytes": MAX_IMAGE_BYTES, "body_bytes": MAX_BODY_BYTES,
                   "side_px": MAX_SIDE_PX},
    }
