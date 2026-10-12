"""一个「会随机抽风」的演示工具，用来观察重试、熔断与幂等的行为。

它的价值在于：**让不确定性变得可控可复现**——
指定前 N 次必失败，第 N+1 次成功，于是重试逻辑可以被单元测试精确断言。
"""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ToolExecutionError
from ..schema import ToolSpec


class FlakyArgs(BaseModel):
    payload: str = Field(default="ping", description="要回显的内容")


class SlowArgs(BaseModel):
    sleep_s: float = Field(default=1.0, ge=0.0, le=30.0, description="故意卡住多久")


def make_flaky_tool(
    *,
    fail_times: int = 2,
    retryable: bool = True,
    max_retries: int = 3,
    retry_backoff_s: float = 0.05,
    idempotent: bool = True,
) -> ToolSpec[FlakyArgs, dict[str, Any]]:
    """前 ``fail_times`` 次调用抛错，之后成功。"""
    counter = itertools.count(1)

    def handler(args: FlakyArgs) -> dict[str, Any]:
        n = next(counter)
        if n <= fail_times:
            raise ToolExecutionError(
                f"第 {n} 次调用被注入故障（retryable={retryable}）",
                retryable=retryable,
            )
        return {"payload": args.payload, "call_no": n}

    return ToolSpec(
        name="flaky_echo",
        description="演示用工具：会随机失败，用于验证重试与幂等。",
        args_model=FlakyArgs,
        handler=handler,
        timeout_s=1.0,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        idempotent=idempotent,
        tags=("demo",),
    )


def make_slow_tool() -> ToolSpec[SlowArgs, dict[str, Any]]:
    """一个真的会睡过去的工具，用来验证硬超时。"""

    def handler(args: SlowArgs) -> dict[str, Any]:
        import time

        time.sleep(args.sleep_s)
        return {"slept": args.sleep_s}

    return ToolSpec(
        name="slow_echo",
        description="演示用工具：睡指定秒数后返回，用于验证超时控制。",
        args_model=SlowArgs,
        handler=handler,
        timeout_s=0.3,
        max_retries=0,
        idempotent=False,
        tags=("demo",),
    )
