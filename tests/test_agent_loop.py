"""Agent 循环测试：验证「模型行为异常时，运行时能兜住」。"""

from __future__ import annotations

from pathlib import Path

from agent_tool_runtime.agent import ReActAgent
from agent_tool_runtime.audit import AuditStore
from agent_tool_runtime.llm import LLMResponse, MockLLM, ToolCall
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.runtime import ToolRuntime
from agent_tool_runtime.schema import ToolSpec
from agent_tool_runtime.testing import FakeClock, RecordingSleeper
from agent_tool_runtime.tools import build_default_registry
from agent_tool_runtime.tools.demo_tools import make_flaky_tool
from pydantic import BaseModel


def _build_runtime(registry: ToolRegistry) -> ToolRuntime:
    return ToolRuntime(
        registry,
        AuditStore(":memory:"),
        sleeper=RecordingSleeper(),
        clock=FakeClock(),
        max_calls_per_run=25,
    )


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)]
    )


class EchoArgs(BaseModel):
    text: str


def _echo(args: EchoArgs) -> dict[str, str]:
    return {"text": args.text}


# ------------------------------------------------------------------ 正常路径


def test_agent_solves_question_with_sql_tool(tmp_path: Path) -> None:
    registry = build_default_registry(db_path=tmp_path / "orders.db", include_http=False)
    runtime = _build_runtime(registry)
    llm = MockLLM(
        [
            call(
                "sql_query",
                sql=(
                    "SELECT city, SUM(amount) AS gmv FROM orders "
                    "GROUP BY city ORDER BY gmv DESC"
                ),
            ),
            LLMResponse(content="杭州最高，18760 元；上海与深圳并列第二，均为 17780 元。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("哪个城市贡献的订单金额最高？")

    assert run.steps[0].result is not None and run.steps[0].result.ok
    assert run.steps[0].result.value["row_count"] == 5
    assert run.steps[0].result.value["rows"][0]["city"] == "杭州"
    assert "杭州" in run.answer
    assert run.model_calls == 2


def test_agent_handles_hallucinated_tool_then_recovers(tmp_path: Path) -> None:
    """模型编了不存在的工具名——运行时返回 tool_not_found，模型换一个。"""
    registry = build_default_registry(db_path=tmp_path / "orders.db", include_http=False)
    runtime = _build_runtime(registry)
    llm = MockLLM(
        [
            call("query_database", sql="SELECT 1"),
            call("sql_query", sql="SELECT COUNT(*) AS n FROM orders"),
            LLMResponse(content="订单表共 10 行。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("订单表有多少条记录？")

    first = run.steps[0]
    assert first.result is not None
    assert first.result.error_code == "tool_not_found"
    assert first.result.retryable is False

    second = run.steps[1]
    assert second.result is not None and second.result.ok
    assert second.result.value["rows"][0]["n"] == 10
    assert "10" in run.answer


def test_agent_recovers_from_dangerous_sql(tmp_path: Path) -> None:
    """模型写了 DROP —— SQL 工具拒绝，错误回灌，模型改用 SELECT。"""
    registry = build_default_registry(db_path=tmp_path / "orders.db", include_http=False)
    runtime = _build_runtime(registry)
    llm = MockLLM(
        [
            call("sql_query", sql="DROP TABLE orders"),
            call("sql_query", sql="SELECT COUNT(*) AS n FROM orders"),
            LLMResponse(content="已改为查询。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("清空订单表后还剩多少条？")

    assert run.steps[0].result is not None
    assert run.steps[0].result.ok is False
    assert "只允许" in (run.steps[0].result.error or "")
    assert run.steps[1].result is not None and run.steps[1].result.ok


def test_transient_failure_is_invisible_to_the_model(
    registry: ToolRegistry,
) -> None:
    """工具前两次抽风，运行时内部重试掉了；模型只看到一个成功的结果。"""
    registry.add(make_flaky_tool(fail_times=2, max_retries=3, retry_backoff_s=0.01))
    runtime = _build_runtime(registry)
    llm = MockLLM(
        [
            call("flaky_echo", payload="hello"),
            LLMResponse(content="拿到了。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("回显 hello")

    assert len(run.steps) == 2  # 1 个工具步 + 1 个收尾步
    tool_step = run.steps[0]
    assert tool_step.result is not None and tool_step.result.ok
    assert tool_step.result.attempts == 3
    assert "attempts=3" in tool_step.summary()


def test_non_retryable_failure_surfaces_to_model(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("strict_echo", "严格回显", EchoArgs, _echo, max_retries=2))

    def bad_args(*_args, **_kwargs) -> LLMResponse:
        return call("strict_echo", text=12345)  # text 需要 str，模型传了 int

    runtime = _build_runtime(registry)
    llm = MockLLM([bad_args, LLMResponse(content="参数错了，我换个说法。")])
    run = ReActAgent(llm, runtime).run("回显")

    step = run.steps[0]
    assert step.result is not None
    assert step.result.ok is False
    assert step.result.attempts == 0


# ------------------------------------------------------------------ 收敛保护


def test_agent_stops_at_max_steps(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))

    class LoopLLM:
        def chat(self, messages, tools=None):  # noqa: ANN001, ANN201
            return call("echo", text="again")

    run = ReActAgent(LoopLLM(), _build_runtime(registry), max_steps=3).run("死循环")
    assert "中止" in run.answer
    assert len(run.steps) == 3


def test_budget_guard_trips_before_dead_loop(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))
    runtime = _build_runtime(registry)
    runtime.max_calls_per_run = 2
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="c2", name="echo", arguments={"text": "b"}),
                    ToolCall(id="c3", name="echo", arguments={"text": "c"}),
                ]
            ),
            LLMResponse(content="收到。"),
        ]
    )
    run = ReActAgent(llm, runtime).run("并发调用三个工具")

    codes = [s.result.error_code if s.result else None for s in run.steps[:3]]
    assert codes == [None, None, "tool_budget_exceeded"]


# ------------------------------------------------------------------ 幂等穿透


def test_agent_run_is_idempotent_across_retries(registry: ToolRegistry) -> None:
    registry.add(make_flaky_tool(fail_times=0, idempotent=True))
    runtime = _build_runtime(registry)
    script = lambda *a, **k: call("flaky_echo", payload="x")  # noqa: E731

    run1 = ReActAgent(
        MockLLM([script, LLMResponse(content="done")]),
        runtime,
        idempotency_prefix="trace-abc",
    ).run("跑一次")
    run2 = ReActAgent(
        MockLLM([script, LLMResponse(content="done")]),
        runtime,
        idempotency_prefix="trace-abc",
    ).run("再跑一次（消息重复投递）")

    assert run1.steps[0].result is not None and run1.steps[0].result.cached is False
    assert run2.steps[0].result is not None and run2.steps[0].result.cached is True


def test_trace_output_is_human_readable(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("echo", "回显", EchoArgs, _echo))
    runtime = _build_runtime(registry)
    llm = MockLLM([call("echo", text="hi"), LLMResponse(content="好了")])
    run = ReActAgent(llm, runtime).run("回显 hi")
    trace = run.trace()
    assert "echo" in trace and "ok" in trace and "final" in trace
