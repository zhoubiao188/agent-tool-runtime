"""熔断器测试。

熔断器解决的问题：当一个下游彻底挂了，Agent 的重试会变成「DDoS 自己」。
连续失败达到阈值就直接拒绝调用，冷却期后再放一次试探请求（半开）。

时间由 FakeClock 精确控制，测试里一次 sleep 都不会发生。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_tool_runtime.errors import ToolExecutionError
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.runtime import ToolRuntime
from agent_tool_runtime.schema import ToolSpec
from agent_tool_runtime.testing import FakeClock
from agent_tool_runtime.tools.demo_tools import make_flaky_tool


class Args(BaseModel):
    payload: str = "p"


def _always_fail(args: Args) -> Any:
    raise ToolExecutionError("下游 503", retryable=True)


def _register_always_fail(registry: ToolRegistry) -> None:
    registry.add(
        ToolSpec("always_fail", "永远失败", Args, _always_fail, max_retries=0,
                 retry_backoff_s=0.0)
    )


def test_circuit_opens_after_threshold(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    _register_always_fail(registry)
    for _ in range(3):
        result = runtime.invoke("always_fail", {})
        assert result.error_code == "tool_execution_error"

    assert runtime.circuit_state()["always_fail"] == "open"

    blocked = runtime.invoke("always_fail", {})
    assert blocked.ok is False
    assert blocked.error_code == "tool_circuit_open"
    assert blocked.attempts == 0, "熔断状态下不应该真的去执行下游"


def test_circuit_recovers_after_cooldown(
    runtime: ToolRuntime,
    registry: ToolRegistry,
    clock: FakeClock,
) -> None:
    # 前 3 次失败 → 熔断；第 4 次开始成功
    registry.add(make_flaky_tool(fail_times=3, max_retries=0, retry_backoff_s=0.0))

    for _ in range(3):
        assert runtime.invoke("flaky_echo", {}).ok is False
    assert runtime.circuit_state()["flaky_echo"] == "open"
    assert runtime.invoke("flaky_echo", {}).error_code == "tool_circuit_open"

    clock.advance(15.1)  # 冷却结束
    recovered = runtime.invoke("flaky_echo", {})
    assert recovered.ok is True
    assert runtime.circuit_state().get("flaky_echo", "closed") == "closed"


def test_half_open_failure_reopens_immediately(
    runtime: ToolRuntime,
    registry: ToolRegistry,
    clock: FakeClock,
) -> None:
    """半开试探若再失败，必须立刻重新熔断，而不是重新数三次。"""
    _register_always_fail(registry)
    for _ in range(3):
        runtime.invoke("always_fail", {})
    assert runtime.circuit_state()["always_fail"] == "open"

    clock.advance(15.1)
    trial = runtime.invoke("always_fail", {})
    assert trial.error_code == "tool_execution_error"  # 真的放出去了
    assert runtime.circuit_state()["always_fail"] == "open"  # 又立刻关上

    clock.advance(1.0)
    assert runtime.invoke("always_fail", {}).error_code == "tool_circuit_open"


def test_failures_outside_window_do_not_open_circuit(
    runtime: ToolRuntime,
    registry: ToolRegistry,
    clock: FakeClock,
) -> None:
    """累计 5 次失败，但每次间隔 31s（> 窗口 30s），不应熔断。"""
    _register_always_fail(registry)
    for _ in range(5):
        runtime.invoke("always_fail", {})
        clock.advance(31.0)
    assert runtime.circuit_state()["always_fail"] == "closed"


def test_validation_errors_do_not_trip_circuit(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    """模型写错参数是「模型的问题」，不该连累整个下游被熔断。"""
    _register_always_fail(registry)
    for _ in range(10):
        runtime.invoke("always_fail", {"payload": {"nested": "not-a-string"}})
    assert runtime.circuit_state().get("always_fail", "closed") == "closed"
