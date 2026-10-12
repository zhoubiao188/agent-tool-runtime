"""LLM 适配层。

刻意保持极简：只用标准库 ``urllib``，不引入任何 SDK。
这样可以跑在离线环境、可以跑在 CI、也可以随时把 ``MockLLM`` 换成真模型，
而 Agent 循环的代码一行都不用改。

这也是本仓库能被测试覆盖的关键——**把不确定的部分关在接口后面**。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .errors import ToolExecutionError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_raw(raw: dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed = {"__raw__": raw_args}
        else:
            parsed = dict(raw_args)
        return ToolCall(
            id=raw.get("id") or f"call_{fn.get('name', 'unknown')}",
            name=fn.get("name") or "",
            arguments=parsed,
        )


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class MockLLM:
    """可编排的假模型：给一个「剧本」，逐个返回预设响应。

    剧本元素可以是 ``LLMResponse``，也可以是 ``(messages, tools) -> LLMResponse``
    的函数——后者能验证「运行时把错误回灌给模型后，模型是否改了参数」。
    """

    def __init__(
        self,
        script: Sequence[LLMResponse | Callable[[Sequence[dict[str, Any]], Any], LLMResponse]],
    ) -> None:
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._script:
            return LLMResponse(content="(剧本已用尽)")
        item = self._script.pop(0)
        if callable(item):
            return item(messages, tools)
        return item


class OpenAIChatLLM:
    """任意 OpenAI 兼容端点（OpenAI / DeepSeek / Qwen / vLLM / Ollama）。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise ToolExecutionError(
                "缺少 OPENAI_API_KEY，无法调用真实模型（可改用 MockLLM 离线运行）"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:500]
            raise ToolExecutionError(
                f"模型接口 HTTP {exc.code}: {detail}", retryable=exc.code >= 500
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ToolExecutionError(f"模型接口不可达: {exc}", retryable=True) from exc

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=[ToolCall.from_raw(tc) for tc in (message.get("tool_calls") or [])],
            usage=payload.get("usage") or {},
        )
