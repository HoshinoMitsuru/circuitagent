"""HTTP 接口层（FastAPI）。只做编排：导入 -> 解析 -> 人工确认 -> 求解 -> 报告。"""

from .server import app, serve

__all__ = ["app", "serve"]
