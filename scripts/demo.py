#!/usr/bin/env python
"""端到端演示：把「生产级工具调用」的四种关键能力跑给你看。

    python scripts/demo.py

不需要任何 API Key、不需要联网——模型用 MockLLM，网络用假 transport。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_tool_runtime import (  # noqa: E402
    AuditStore,
    LLMResponse,
    MockLLM,
    ReActAgent,
    ToolCall,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)
from agent_tool_runtime.tools import build_default_registry  # noqa: E402
from agent_tool_runtime.tools.demo_tools import make_flaky_tool, make_slow_tool  # noqa: E402

LINE = "─" * 78


def title(text: str) -> None:
    print(f"\n{LINE}\n▶ {text}\n{LINE}")


def new_runtime(registry: ToolRegistry, *, audit_path: str = ":memory:") -> ToolRuntime:
    return ToolRuntime(registry, AuditStore(audit_path))


def scenario_1_react_loop() -> None:
    title("场景 1｜ReAct 循环：模型自己查库得出结论")
    workdir = Path(tempfile.mkdtemp(prefix="atr-demo-"))
    registry = build_default_registry(db_path=workdir / "orders.db", include_http=False)
    runtime = new_runtime(registry)

    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="sql_query",
                        arguments={
                            "sql": (
                                "SELECT city, SUM(amount) AS gmv, SUM(quantity) AS qty "
                                "FROM orders GROUP BY city ORDER BY gmv DESC"
                            ),
                            "limit": 10,
                        },
                    )
                ]
            ),
            LLMResponse(content="杭州 GMV 最高（18760 元），其次是上海与深圳（各 17780 元）。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("按 GMV 给城市排名", trace_id="demo-trace-1")
    print(run.trace())
    print(f"\n最终回答：{run.answer}")


def scenario_2_retry() -> None:
    title("场景 2｜瞬时故障：两次抽风被运行时悄悄重试掉，模型无感")
    registry = ToolRegistry()
    registry.add(make_flaky_tool(fail_times=2, max_retries=3, retry_backoff_s=0.05))
    runtime = new_runtime(registry)

    llm = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="flaky_echo", arguments={"payload": "hi"})]),
            LLMResponse(content="已经拿到 echo 结果。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("回显 hi")
    print(run.trace())
    print(f"\n最终回答：{run.answer}")


def scenario_3_timeout_and_circuit() -> None:
    title("场景 3｜硬超时 + 熔断：阻止 Agent 把故障下游打穿")

    class DeadArgs(BaseModel):
        payload: str = "p"

    def always_raise(args: DeadArgs) -> dict[str, str]:
        raise RuntimeError("downstream is dead")

    registry = ToolRegistry()
    registry.add(make_slow_tool())
    registry.add(
        ToolSpec(
            name="broken_downstream",
            description="模拟一个彻底挂掉的下游",
            args_model=DeadArgs,
            handler=always_raise,
            max_retries=0,
        )
    )
    runtime = new_runtime(registry)

    slow = runtime.invoke("slow_echo", {"sleep_s": 1.0})
    print(f"超时保护  -> ok={slow.ok} code={slow.error_code} 耗时={slow.duration_ms:.1f}ms")

    for i in range(1, 5):
        result = runtime.invoke("broken_downstream", {})
        print(
            f"第 {i} 次调用 -> ok={result.ok} code={result.error_code} attempts={result.attempts}"
        )
    print(f"\n熔断状态：{runtime.circuit_state()}")


def scenario_4_idempotency() -> None:
    title("场景 4｜幂等去重：同一条指令重复投递，只真正执行一次")
    registry = ToolRegistry()
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))
    runtime = new_runtime(registry)

    first = runtime.invoke("flaky_echo", {"payload": "生成月报"}, idempotency_key="order-9527")
    second = runtime.invoke("flaky_echo", {"payload": "生成月报"}, idempotency_key="order-9527")
    print(f"第一次 -> call_no={first.value['call_no']} cached={first.cached}")
    print(f"第二次 -> call_no={second.value['call_no']} cached={second.cached}")
    print("结论：第二次没有触碰下游，直接复用第一次的结果。")


def scenario_5_audit() -> None:
    title("场景 5｜审计面板：每个工具的调用数、成功率、重试次数、耗时")
    workdir = Path(tempfile.mkdtemp(prefix="atr-demo-"))
    registry = build_default_registry(db_path=workdir / "orders.db")
    registry.add(make_flaky_tool(fail_times=1, max_retries=2, retry_backoff_s=0.01))
    audit = AuditStore(str(workdir / "audit.db"))
    runtime = ToolRuntime(registry, audit)

    runtime.invoke("sql_query", {"sql": "SELECT COUNT(*) AS n FROM orders"}, trace_id="t1")
    runtime.invoke("calculator", {"expression": "(1250-980)/980*100"}, trace_id="t1")
    runtime.invoke("flaky_echo", {"payload": "x"}, trace_id="t1")
    runtime.invoke("not_exist", {}, trace_id="t1")

    header = f"{'tool':<18}{'calls':>7}{'ok':>5}{'succ':>9}{'retries':>9}{'avg_ms':>10}"
    print(header)
    print("-" * len(header))
    for name, row in audit.stats().items():
        print(
            f"{name:<18}{row['calls']:>7.0f}{row['ok']:>5.0f}"
            f"{row['success_rate']:>9.2%}{row['retries']:>9.0f}{row['avg_ms']:>10.3f}"
        )

    print("\n最近 4 条审计明细：")
    for row in audit.recent(4):
        print(
            f"  #{row['id']} {row['tool']:<18} ok={row['ok']} attempts={row['attempts']} "
            f"cached={row['cached']} err={row['error_code'] or '-'}"
        )
    audit.close()


def main() -> None:
    print("agent-tool-runtime · 端到端演示")
    scenario_1_react_loop()
    scenario_2_retry()
    scenario_3_timeout_and_circuit()
    scenario_4_idempotency()
    scenario_5_audit()
    print(f"\n{LINE}\n全部场景执行完毕。\n{LINE}")


if __name__ == "__main__":
    main()
