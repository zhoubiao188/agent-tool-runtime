"""测试辅助（也供使用者测试自己的工具）。

时间相关的逻辑（熔断窗口、冷却、指数退避）如果不做成可注入的，
测试里就只能靠 ``sleep`` 硬等，既慢又容易 flaky。
所以 ``ToolRuntime`` 允许注入 ``clock`` 与 ``sleeper``，
这里提供两个开箱即用的实现。
"""

from __future__ import annotations


class FakeClock:
    """可手动推进的假时钟。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RecordingSleeper:
    """记录退避序列，但不真的等待；可选地把假时钟一起往前推。"""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.delays: list[float] = []
        self._clock = clock

    def __call__(self, seconds: float) -> None:
        self.delays.append(round(seconds, 6))
        if self._clock is not None:
            self._clock.advance(seconds)

    @property
    def total(self) -> float:
        return round(sum(self.delays), 6)


__all__ = ["FakeClock", "RecordingSleeper"]
