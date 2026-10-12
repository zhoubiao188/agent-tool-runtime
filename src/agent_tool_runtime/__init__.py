"""agent-tool-runtime：给 AI Agent 的工具调用加一层生产级运行时。

核心导出：

    ToolSpec        工具声明（Pydantic 入参 + 执行体 + 超时/重试策略）
    ToolRegistry    工具注册中心
    ToolRuntime     执行引擎（校验 / 幂等 / 熔断 / 超时 / 重试 / 审计）
    ToolCallResult  结构化调用结果，可直接回灌给模型
    ReActAgent      最小可用的 Agent 循环
"""

from .agent import AgentRun, AgentStep, ReActAgent, run_once
from .audit import AuditStore
from .errors import (
    ToolBudgetExceeded,
    ToolCircuitOpen,
    ToolError,
    ToolExecutionError,
    ToolNotFound,
    ToolTimeout,
    ToolValidationError,
)
from .llm import LLMClient, LLMResponse, MockLLM, OpenAIChatLLM, ToolCall
from .registry import ToolRegistry
from .runtime import ToolCallResult, ToolRuntime
from .schema import ToolSpec, tool

__version__ = "0.1.0"

__all__ = [
    "AgentRun",
    "AgentStep",
    "AuditStore",
    "LLMClient",
    "LLMResponse",
    "MockLLM",
    "OpenAIChatLLM",
    "ReActAgent",
    "ToolBudgetExceeded",
    "ToolCall",
    "ToolCallResult",
    "ToolCircuitOpen",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFound",
    "ToolRegistry",
    "ToolRuntime",
    "ToolSpec",
    "ToolTimeout",
    "ToolValidationError",
    "run_once",
    "tool",
    "__version__",
]
