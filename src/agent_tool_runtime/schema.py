"""工具声明层：用 Pydantic 模型描述入参，让「工具契约」只有一个真源。

一个工具需要同时被三种消费者使用：

1. **模型**：需要 JSON Schema 来决定怎么填参数；
2. **运行时**：需要在调用前做严格的参数校验；
3. **人类**：需要清晰的字段说明与示例。

如果这三者各自维护一份「参数说明」，它们迟早会漂移。
`ToolSpec` 的作用就是把它们收敛到同一个 Pydantic 模型上。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import ToolValidationError

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult")


class ToolSpec(Generic[TArgs, TResult]):
    """一个可被 Agent 调用的工具的完整声明。

    Attributes:
        name: 暴露给模型的函数名，只允许字母/数字/下划线。
        description: 给模型看的功能说明，直接影响调用准确率。
        args_model: 入参的 Pydantic 模型，同时充当校验器与 JSON Schema 来源。
        handler: 真正的执行体，签名必须为 ``handler(args: TArgs) -> TResult``。
        timeout_s: 单次执行的硬超时。
        max_retries: 可重试错误的额外重试次数。
        retry_backoff_s: 指数退避基数。
        idempotent: 声明该工具是否幂等（同样入参重复执行结果一致）。
            只有幂等工具才会启用缓存与去重。
        tags: 分类标签，便于做权限、配额、埋点分组。
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[TArgs],
        handler: Callable[[TArgs], TResult],
        *,
        timeout_s: float = 5.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.2,
        idempotent: bool = False,
        tags: Sequence[str] = (),
    ) -> None:
        if not name or not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError(f"非法的工具名：{name!r}（只允许字母、数字、下划线）")
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须大于 0")
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        if not issubclass(args_model, BaseModel):
            raise TypeError("args_model 必须是 pydantic.BaseModel 的子类")

        params = inspect.signature(handler).parameters
        if len(params) != 1:
            raise TypeError(
                f"工具 {name} 的 handler 必须只接收一个参数（args 模型），"
                f"当前收到 {len(params)} 个：{list(params)}"
            )

        self.name = name
        self.description = description.strip()
        self.args_model = args_model
        self.handler = handler
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.idempotent = idempotent
        self.tags = tuple(tags)

    # ------------------------------------------------------------------ 校验
    def parse(self, raw: Mapping[str, Any] | BaseModel | None) -> TArgs:
        """把模型给的原始参数解析成强类型对象，失败则抛 ToolValidationError。"""
        if raw is None:
            raw = {}
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        if not isinstance(raw, Mapping):
            raise ToolValidationError(
                f"参数必须是 JSON 对象，收到 {type(raw).__name__}", tool=self.name
            )
        try:
            return self.args_model.model_validate(dict(raw))
        except ValidationError as exc:
            # 把 pydantic 的错误压缩成模型能看懂的一句话
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolValidationError(details, tool=self.name) from exc

    # ------------------------------------------------------------ Schema 输出
    def parameters_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        # 去掉 pydantic 加的一些噪音字段，让模型看到的 schema 更小更准
        schema.pop("title", None)
        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        """OpenAI / 兼容协议（DeepSeek、Qwen、vLLM 等）的 function calling 描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def __call__(self, raw: Mapping[str, Any] | None = None) -> TResult:
        return self.handler(self.parse(raw))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolSpec {self.name} timeout={self.timeout_s}s retries={self.max_retries}>"


def tool(
    args_model: type[TArgs],
    *,
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Callable[[Callable[[TArgs], TResult]], ToolSpec[TArgs, TResult]]:
    """把普通函数快速声明成工具的装饰器。

    用法::

        class AddArgs(BaseModel):
            a: float
            b: float

        @tool(AddArgs, description="两数相加")
        def add(args: AddArgs) -> float:
            return args.a + args.b
    """

    def decorator(fn: Callable[[TArgs], TResult]) -> ToolSpec[TArgs, TResult]:
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        return ToolSpec(
            name or fn.__name__,
            description or (doc[0] if doc else fn.__name__),
            args_model,
            fn,
            **kwargs,
        )

    return decorator
