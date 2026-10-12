"""运行时策略测试：参数校验、硬超时、失败分类、指数退避、预算、审计。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_tool_runtime.errors import ToolExecutionError, ToolTimeout
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.runtime import ToolRuntime
from agent_tool_runtime.schema import ToolSpec
from agent_tool_runtime.testing import RecordingSleeper
from agent_tool_runtime.tools.demo_tools import make_flaky_tool, make_slow_tool


class EchoArgs(BaseModel):
    text: str = "hi"
    count: int = 1


def _echo(args: EchoArgs) -> dict[str, Any]:
    return {"text": args.text * args.count}


def _boom_retryable(args: EchoArgs) -> dict[str, Any]:
    raise ToolExecutionError("下游 503", retryable=True)


def _boom_fatal(args: EchoArgs) -> dict[str, Any]:
    raise ToolExecutionError("账号无权限", retryable=False)


# ------------------------------------------------------------------ 参数校验


def test_invalid_args_short_circuits_without_execution(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    calls = {"n": 0}

    def handler(args: EchoArgs) -> Any:
        calls["n"] += 1
        return args.text

    registry.add(ToolSpec("echo", "回显", EchoArgs, handler))
    result = runtime.invoke("echo", {"count": "不是数字"})
    assert result.ok is False
    assert result.error_code == "tool_validation_error"
    assert result.attempts == 0
    assert calls["n"] == 0, "参数不合法时绝不能碰执行体"


def test_unknown_tool_is_structured_not_crash(runtime: ToolRuntime) -> None:
    result = runtime.invoke("not_exist", {})
    assert result.ok is False
    assert result.error_code == "tool_not_found"


def test_bad_json_string_is_handled(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))
    result = runtime.invoke("echo", "{不是 json")
    assert result.ok is False
    assert result.error_code == "tool_execution_error"


# ------------------------------------------------------------------ 超时


def test_timeout_is_enforced(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    """真睡 0.5s，工具超时设 0.3s，必须被判为 timeout 而不是等它睡完。"""
    registry.add(make_slow_tool())
    result = runtime.invoke("slow_echo", {"sleep_s": 0.5})
    assert result.ok is False
    assert result.error_code == "tool_timeout"
    assert result.retryable is True


# ------------------------------------------------------------------ 重试


def test_retry_succeeds_after_transient_failures(
    runtime: ToolRuntime, registry: ToolRegistry, sleeper: RecordingSleeper
) -> None:
    registry.add(make_flaky_tool(fail_times=2, max_retries=3, retry_backoff_s=0.05))
    result = runtime.invoke("flaky_echo", {"payload": "p"})
    assert result.ok is True
    assert result.attempts == 3
    assert result.value["call_no"] == 3
    # 指数退避：第 1 次重试前等 0.05s，第 2 次重试前等 0.1s
    assert sleeper.delays == [0.05, 0.1]


def test_non_retryable_error_gives_up_immediately(
    runtime: ToolRuntime, registry: ToolRegistry, sleeper: RecordingSleeper
) -> None:
    registry.add(ToolSpec("fatal", "权限错误", EchoArgs, _boom_fatal, max_retries=5))
    result = runtime.invoke("fatal", {})
    assert result.ok is False
    assert result.attempts == 1
    assert sleeper.delays == [], "不可重试的错误不应该产生任何退避等待"


def test_retry_budget_is_exhausted(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    registry.add(ToolSpec("flaky", "永远失败", EchoArgs, _boom_retryable, max_retries=2))
    result = runtime.invoke("flaky", {})
    assert result.ok is False
    assert result.attempts == 3  # 1 次首发 + 2 次重试


# ------------------------------------------------------------------ 预算


def test_per_run_call_budget(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))
    runtime.max_calls_per_run = 3
    runtime.reset_budget()
    for _ in range(3):
        assert runtime.invoke("echo", {"text": "x"}).ok is True
    overflow = runtime.invoke("echo", {"text": "x"})
    assert overflow.ok is False
    assert overflow.error_code == "tool_budget_exceeded"


# ------------------------------------------------------------------ 审计


def test_audit_records_every_attempt_path(
    runtime: ToolRuntime, registry: ToolRegistry, audit: Any
) -> None:
    registry.add(make_flaky_tool(fail_times=1, max_retries=2, retry_backoff_s=0.01))
    ok = runtime.invoke("flaky_echo", {"payload": "a"}, trace_id="trace-1")
    assert ok.ok is True
    runtime.invoke("not_exist", {}, trace_id="trace-1")

    rows = audit.recent(10)
    assert len(rows) == 2
    assert {r["trace_id"] for r in rows} == {"trace-1"}

    stats = audit.stats()
    assert stats["flaky_echo"]["calls"] == 1
    assert stats["flaky_echo"]["success_rate"] == 1.0
    assert stats["flaky_echo"]["retries"] == 1
    assert stats["not_exist"]["success_rate"] == 0.0


def test_result_digest_is_stable(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))
    a = runtime.invoke("echo", {"text": "hello"})
    b = runtime.invoke("echo", {"text": "hello"})
    assert a.digest == b.digest
    assert runtime.invoke("echo", {"text": "other"}).digest != a.digest


def test_call_raises_for_direct_use(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    registry.add(make_slow_tool())
    with pytest.raises(ToolTimeout):
        runtime.call("slow_echo", {"sleep_s": 0.4})
