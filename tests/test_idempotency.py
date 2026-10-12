"""幂等与去重测试。

幂等键的真实场景：用户网络抖动连点了两次「生成月度报表」，
Agent 平台重试了同一条指令，消息队列至少一次投递……
没有幂等，这些都会变成「重复下单」「重复发券」「重复扣款」。
"""

from __future__ import annotations

from pathlib import Path

from agent_tool_runtime.audit import AuditStore
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.runtime import ToolRuntime
from agent_tool_runtime.testing import FakeClock, RecordingSleeper
from agent_tool_runtime.tools.demo_tools import make_flaky_tool


def test_idempotent_tool_reuses_first_result(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))
    first = runtime.invoke("flaky_echo", {"payload": "做报表"}, idempotency_key="req-001")
    second = runtime.invoke("flaky_echo", {"payload": "做报表"}, idempotency_key="req-001")

    assert first.ok and second.ok
    assert first.cached is False and second.cached is True
    # 关键断言：第二次没有真的执行工具，返回的是第一次的结果
    assert second.value == first.value
    assert second.value["call_no"] == 1
    assert second.attempts == 0


def test_non_idempotent_tool_always_executes(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    registry.add(make_flaky_tool(fail_times=0, idempotent=False))
    a = runtime.invoke("flaky_echo", {"payload": "p"}, idempotency_key="req-002")
    b = runtime.invoke("flaky_echo", {"payload": "p"}, idempotency_key="req-002")
    assert b.cached is False
    assert b.value["call_no"] == 2


def test_without_key_nothing_is_cached(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))
    a = runtime.invoke("flaky_echo", {"payload": "p"})
    b = runtime.invoke("flaky_echo", {"payload": "p"})
    assert a.cached is False and b.cached is False
    assert b.value["call_no"] == 2


def test_failed_call_is_not_cached(
    runtime: ToolRuntime, registry: ToolRegistry
) -> None:
    """失败的结果绝不能进幂等缓存，否则故障会被永久固化。"""
    registry.add(make_flaky_tool(fail_times=9, max_retries=0, idempotent=True))
    bad = runtime.invoke("flaky_echo", {"payload": "p"}, idempotency_key="req-003")
    assert bad.ok is False
    assert runtime.audit.get_cached("req-003") is None


def test_cache_survives_runtime_restart(tmp_path: Path) -> None:
    """把审计库换成文件后，重启进程依然能命中幂等缓存——这才是生产形态。"""
    db = tmp_path / "audit.db"
    registry = ToolRegistry()
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))

    store1 = AuditStore(str(db))
    runtime1 = ToolRuntime(registry, store1, sleeper=RecordingSleeper(), clock=FakeClock())
    first = runtime1.invoke("flaky_echo", {"payload": "p"}, idempotency_key="order-9527")
    store1.close()

    store2 = AuditStore(str(db))
    runtime2 = ToolRuntime(registry, store2, sleeper=RecordingSleeper(), clock=FakeClock())
    second = runtime2.invoke("flaky_echo", {"payload": "p"}, idempotency_key="order-9527")
    store2.close()

    assert second.cached is True
    assert second.value == first.value


def test_audit_marks_cached_calls(runtime: ToolRuntime, registry: ToolRegistry) -> None:
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))
    runtime.invoke("flaky_echo", {"payload": "p"}, idempotency_key="k")
    runtime.invoke("flaky_echo", {"payload": "p"}, idempotency_key="k")
    rows = runtime.audit.recent(5)
    assert len(rows) == 2
    assert sum(r["cached"] for r in rows) == 1
