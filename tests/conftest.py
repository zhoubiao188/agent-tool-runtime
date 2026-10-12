"""共享测试夹具。

两个关键夹具：

* ``FakeClock`` —— 让「熔断窗口」「冷却时间」这类时间相关的逻辑可被精确断言，
  测试不必真的 sleep 15 秒。
* ``RecordingSleeper`` —— 记录退避序列，验证指数退避确实按 0.05 / 0.1 / 0.2 走。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_tool_runtime.audit import AuditStore  # noqa: E402
from agent_tool_runtime.registry import ToolRegistry  # noqa: E402
from agent_tool_runtime.runtime import ToolRuntime  # noqa: E402
from agent_tool_runtime.testing import FakeClock, RecordingSleeper  # noqa: E402


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeper(clock: FakeClock) -> RecordingSleeper:
    return RecordingSleeper(clock)


@pytest.fixture
def audit() -> AuditStore:
    store = AuditStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def runtime(
    registry: ToolRegistry,
    audit: AuditStore,
    clock: FakeClock,
    sleeper: RecordingSleeper,
) -> ToolRuntime:
    return ToolRuntime(
        registry,
        audit,
        sleeper=sleeper,
        clock=clock,
        circuit_failure_threshold=3,
        circuit_window_s=30.0,
        circuit_cooldown_s=15.0,
        max_calls_per_run=25,
    )
