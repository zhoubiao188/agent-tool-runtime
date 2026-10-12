"""工具调用过程中的错误类型。

设计原则：把「错误」分成两类，因为 Agent 运行时对这两类的处理完全不同——

* **可重试（retryable=True）**：网络抖动、限流、下游 5xx、超时。重试有意义。
* **不可重试（retryable=False）**：参数校验失败、权限不足、资源不存在。
  再试一百次也是同样的结果，重试只会放大下游压力。

把「失败分类」做成类型系统的一部分，而不是散落在 if/else 里，
是 Tool Runtime 能稳定工作的前提。
"""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """所有工具异常的基类。"""

    code: str = "tool_error"
    retryable: bool = False

    def __init__(self, message: str, *, tool: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.tool = tool

    def __str__(self) -> str:  # pragma: no cover - 直白的字符串化
        if self.tool:
            return f"[{self.code}] {self.tool}: {self.message}"
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "tool": self.tool,
            "retryable": self.retryable,
        }


class ToolNotFound(ToolError):
    """模型编造了一个不存在的工具名。线上非常常见。"""

    code = "tool_not_found"
    retryable = False


class ToolValidationError(ToolError):
    """入参不符合 schema。不可重试，应该把错误回灌给模型让它自己改。"""

    code = "tool_validation_error"
    retryable = False


class ToolTimeout(ToolError):
    """工具执行超时。下游卡住时重试往往能成功。"""

    code = "tool_timeout"
    retryable = True


class ToolExecutionError(ToolError):
    """工具执行过程中的业务异常，可显式声明是否可重试。"""

    code = "tool_execution_error"

    def __init__(
        self,
        message: str,
        *,
        tool: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, tool=tool)
        self.retryable = retryable


class ToolCircuitOpen(ToolError):
    """熔断器打开：某工具在窗口期内连续失败，暂时拒绝调用，保护下游。"""

    code = "tool_circuit_open"
    retryable = False


class ToolBudgetExceeded(ToolError):
    """单次 Agent 运行的调用预算耗尽，防止死循环烧钱。"""

    code = "tool_budget_exceeded"
    retryable = False
