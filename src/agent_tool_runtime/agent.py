"""最小但完整的 ReAct Agent 循环。

刻意写得很短，因为「循环控制」不是这篇文章的重点，
「循环里每一次工具调用都被 Runtime 管住」才是。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm import LLMClient, LLMResponse, ToolCall
from .runtime import ToolCallResult, ToolRuntime

DEFAULT_SYSTEM_PROMPT = """你是一个严谨的任务型 Agent。

工作方式：
1. 先想清楚需要哪些信息，再决定调用哪个工具；
2. 一次只推进一个明确的目标，不要臆造工具名；
3. 工具返回 ok=false 时，先读 error_code：
   - tool_validation_error：你自己参数写错了，改正后重试；
   - tool_not_found：没有这个工具，换一个或直接回答；
   - 其它：可以用不同参数再试一次，但不要连续重试超过两次；
4. 信息足够时立刻给出最终答案，不要为了「多调一次工具」而拖延。
"""


@dataclass
class AgentStep:
    index: int
    tool: str | None
    args: dict[str, Any] | None
    result: ToolCallResult | None
    content: str | None = None

    def summary(self) -> str:
        if self.tool is None:
            return f"[{self.index}] final: {self.content}"
        status = "ok" if (self.result and self.result.ok) else "FAIL"
        extra = ""
        if self.result and not self.result.ok:
            extra = f" ({self.result.error_code}: {self.result.error})"
        if self.result and self.result.cached:
            extra = " (idempotent cache hit)"
        return (
            f"[{self.index}] {self.tool}({json.dumps(self.args or {}, ensure_ascii=False)})"
            f" -> {status} in {self.result.duration_ms if self.result else 0:.1f}ms"
            f" attempts={self.result.attempts if self.result else 0}{extra}"
        )


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    model_calls: int = 0

    def trace(self) -> str:
        return "\n".join(step.summary() for step in self.steps)


class ReActAgent:
    def __init__(
        self,
        llm: LLMClient,
        runtime: ToolRuntime,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
        idempotency_prefix: str | None = None,
    ) -> None:
        self.llm = llm
        self.runtime = runtime
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.idempotency_prefix = idempotency_prefix

    def run(self, question: str, *, trace_id: str | None = None) -> AgentRun:
        self.runtime.reset_budget()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        tools = self.runtime.registry.openai_tools()
        run = AgentRun(question=question, answer="")

        for step_index in range(1, self.max_steps + 1):
            response: LLMResponse = self.llm.chat(messages, tools=tools)
            run.model_calls += 1

            if not response.wants_tools:
                run.answer = response.content or ""
                run.steps.append(
                    AgentStep(index=step_index, tool=None, args=None, result=None,
                              content=run.answer)
                )
                return run

            # 记下模型的「思考/指令」这一轮
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                result = self.runtime.invoke(
                    call.name,
                    call.arguments,
                    idempotency_key=self._key_for(call),
                    trace_id=trace_id,
                )
                run.steps.append(
                    AgentStep(
                        index=step_index,
                        tool=call.name,
                        args=call.arguments,
                        result=result,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.to_content(),
                    }
                )

        run.answer = "（达到最大步数仍未收敛，已中止，请人工介入）"
        return run

    def _key_for(self, call: ToolCall) -> str | None:
        if not self.idempotency_prefix:
            return None
        return f"{self.idempotency_prefix}:{call.id}"


def run_once(question: str, *, runtime: ToolRuntime, llm: LLMClient, **kw: Any) -> AgentRun:
    """一行跑一个 Agent。"""
    return ReActAgent(llm, runtime, **kw).run(question)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentRun",
    "AgentStep",
    "ReActAgent",
    "run_once",
]
