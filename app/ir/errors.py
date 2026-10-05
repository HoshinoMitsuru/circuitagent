"""IR 层的错误类型。

★ 为什么单独一个文件、而不是留在 ``model.py`` 里：

``model.Circuit`` 需要持有 ``ParamTable``（参数表），于是 ``model`` 要导入
``params``；而 ``params`` 校验失败时又必须抛出 ``CircuitError``。
两者一旦都写在 ``model.py`` 里就是**循环导入**。

拆出本文件后依赖方向变成一条线::

    errors.py  ←  params.py  ←  model.py  ← 其它一切

``model.py`` 仍然 re-export 这两个名字，所以既有的
``from app.ir.model import CircuitError`` 一处都不用改。

这两类的**话术必须分开**，因为要告诉用户的事完全不同：

- ``CircuitError``：**输入有问题**（漏画线、缺数值、类型不支持）→ 请改图；
- ``CircuitUnsatisfiable``：**输入没问题、题目理想化模型自相矛盾** → 请改题。
"""

from __future__ import annotations

from typing import Any


class CircuitError(ValueError):
    """IR 层面的结构性错误（拓扑不连通、元件类型不支持等）。"""


class CircuitUnsatisfiable(CircuitError):
    """约束集**自相矛盾** —— 这张电路在直流稳态下无解。

    与 ``CircuitError`` 分成两类，是因为它们要告诉用户完全不同的事：

    - ``CircuitError``：**输入有问题**（漏画线、缺数值、类型不支持）→ 请改图；
    - ``CircuitUnsatisfiable``：**输入没问题，但题目的理想化模型自相矛盾** → 请改题。

    典型的后者：电感在直流下是理想短路，若它与理想电压源并联，就同时要求
    ``V_a − V_b = E`` 和 ``V_a = V_b``。这在教科书里是"该理想电路无解"，
    而不是"程序算不出来"。

    ★ 之所以单开一个类型，是因为本项目**踩过一次**：化简层把"两端等电位的元件"
    一律当作"被短接、等价于消失"删掉，于是矛盾凭空消失，剩下的电路照常可解，
    三条代码路径一致地给出全零解，功率守恒平凡成立（ΣP = 0），
    最后报告 `overall_pass = True` 并告诉学生"**这张电路的读图与建模可以采信**"。
    静默降级到这种程度，必须有一个能被显式抛出的类型来兜住。

    ``contradictions`` 里每条都说明"谁和谁矛盾、矛盾在哪、建议怎么办"，
    供报告逐条呈现，而不是只丢一句"无解"。
    """

    def __init__(self, message: str, contradictions: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.contradictions: list[dict[str, Any]] = list(contradictions or [])
