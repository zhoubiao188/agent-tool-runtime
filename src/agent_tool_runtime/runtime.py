"""Tool Runtime —— 本文的主角。

它把「模型说要调工具」和「工具真的被执行」之间的所有脏活接了下来：

    ┌──────────┐   1. 查表     ┌──────────┐
    │  Agent   │──────────────▶│ Registry │
    └────┬─────┘               └──────────┘
         │ 2. invoke(name, args)
         ▼
    ┌───────────────────────────────────────────┐
    │              Tool Runtime                 │
    │  ① 预算检查   ② 参数校验   ③ 幂等命中       │
    │  ④ 熔断判断   ⑤ 超时执行   ⑥ 指数退避重试    │
    │  ⑦ 结果摘要   ⑧ 审计落库                   │
    └───────────────────────────────────────────┘
         │ 3. ToolCallResult
         ▼
    ┌──────────┐
    │  Agent   │ 把结果回灌给模型，继续下一轮
    └──────────┘
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .audit import AuditStore
from .errors import (
    ToolBudgetExceeded,
    ToolCircuitOpen,
    ToolError,
    ToolExecutionError,
    ToolTimeout,
)
from .registry import ToolRegistry
from .schema import ToolSpec

# 这些内建异常通常代表「暂时性」故障，值得重试一次。
_RETRYABLE_BUILTINS = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    InterruptedError,
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ToolCallResult:
    """一次工具调用的完整结果，可直接回灌给模型。"""

    tool: str
    ok: bool
    value: Any = None
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    attempts: int = 0
    duration_ms: float = 0.0
    cached: bool = False
    trace_id: str | None = None
    digest: str | None = None

    def to_content(self) -> str:
        """给模型看的文本。失败时带上错误码，让模型知道该换参数还是该放弃。"""
        if self.ok:
            return json.dumps(
                {"ok": True, "result": self.value, "cached": self.cached},
                ensure_ascii=False,
                default=str,
            )
        return json.dumps(
            {
                "ok": False,
                "error_code": self.error_code,
                "error": self.error,
                "retryable": self.retryable,
            },
            ensure_ascii=False,
        )


class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditStore | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        circuit_failure_threshold: int = 3,
        circuit_window_s: float = 30.0,
        circuit_cooldown_s: float = 15.0,
        max_calls_per_run: int = 25,
    ) -> None:
        self.registry = registry
        self.audit = audit if audit is not None else AuditStore(":memory:")
        self._sleep = sleeper
        self._clock = clock
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_window_s = circuit_window_s
        self.circuit_cooldown_s = circuit_cooldown_s
        self.max_calls_per_run = max_calls_per_run

        self._lock = threading.RLock()
        self._failures: dict[str, list[float]] = {}
        self._opened_at: dict[str, float] = {}
        self._half_open: set[str] = set()
        self._calls_this_run = 0

    # ------------------------------------------------------------------ 预算
    def reset_budget(self) -> None:
        """每次 Agent 运行开始时调用，防止单个会话无限次调用工具。"""
        with self._lock:
            self._calls_this_run = 0

    # ------------------------------------------------------------------ 主入口
    def call(
        self,
        name: str,
        args: Mapping[str, Any] | str | None = None,
        *,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """调用工具，失败时抛 ToolError。适合在你自己的代码里直接用。"""
        result = self.invoke(
            name, args, idempotency_key=idempotency_key, trace_id=trace_id
        )
        if result.ok:
            return result.value
        raise _rebuild(result)

    def invoke(
        self,
        name: str,
        args: Mapping[str, Any] | str | None = None,
        *,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
    ) -> ToolCallResult:
        """调用工具，永不因业务错误抛异常——Agent 循环里请用这个。"""
        started = self._clock()
        try:
            raw = _coerce_args(args)
        except ToolError as exc:
            return self._fail(name, {}, 0, exc, started, trace_id)

        # ---------- ① 预算 ----------
        with self._lock:
            if self._calls_this_run >= self.max_calls_per_run:
                return self._fail(
                    name,
                    raw,
                    0,
                    ToolBudgetExceeded(
                        f"本次运行已调用 {self._calls_this_run} 次，超过预算 "
                        f"{self.max_calls_per_run}",
                        tool=name,
                    ),
                    started,
                    trace_id,
                )
            self._calls_this_run += 1

        # ---------- ② 查表 ----------
        try:
            spec = self.registry.get(name)
        except ToolError as exc:
            return self._fail(name, raw, 0, exc, started, trace_id)

        # ---------- ③ 参数校验 ----------
        try:
            parsed = spec.parse(raw)
        except ToolError as exc:
            return self._fail(name, raw, 0, exc, started, trace_id)

        # ---------- ④ 幂等 ----------
        cache_key = idempotency_key if (spec.idempotent and idempotency_key) else None
        if cache_key:
            hit = self.audit.get_cached(cache_key)
            if hit is not None:
                result = ToolCallResult(
                    tool=name,
                    ok=True,
                    value=hit["value"],
                    attempts=0,
                    duration_ms=round(self._elapsed_ms(started), 3),
                    cached=True,
                    trace_id=trace_id,
                    digest=_digest(hit["value"]),
                )
                self.audit.log_call(
                    tool=name,
                    args=raw,
                    ok=True,
                    attempts=0,
                    duration_ms=result.duration_ms,
                    trace_id=trace_id,
                    cached=True,
                    result_digest=result.digest,
                )
                return result

        # ---------- ⑤ 熔断 ----------
        try:
            self._check_circuit(spec.name)
        except ToolError as exc:
            return self._fail(spec.name, raw, 0, exc, started, trace_id)

        # ---------- ⑥ 执行 + 重试 ----------
        error: ToolError | None = None
        attempts = 0
        for attempt in range(1, spec.max_retries + 2):
            attempts = attempt
            try:
                value = self._execute_with_timeout(spec, parsed, spec.timeout_s)
            except ToolError as exc:
                error = exc
            except BaseException as exc:  # noqa: BLE001 - 兜住一切，转成结构化错误
                error = self._normalize(exc, spec.name)
            else:
                error = None
                digest = _digest(value)
                self._on_success(spec.name)
                if cache_key:
                    self.audit.put_cached(cache_key, spec.name, value)
                duration = self._elapsed_ms(started)
                self.audit.log_call(
                    tool=spec.name,
                    args=raw,
                    ok=True,
                    attempts=attempt,
                    duration_ms=duration,
                    trace_id=trace_id,
                    result_digest=digest,
                )
                return ToolCallResult(
                    tool=spec.name,
                    ok=True,
                    value=value,
                    attempts=attempt,
                    duration_ms=round(duration, 3),
                    trace_id=trace_id,
                    digest=digest,
                )

            # 走到这里说明失败了：决定还要不要重试
            if not error.retryable or attempt > spec.max_retries:
                break
            backoff = spec.retry_backoff_s * (2 ** (attempt - 1))
            self._sleep(backoff)

        assert error is not None
        return self._fail(spec.name, raw, attempts, error, started, trace_id)

    # -------------------------------------------------------------- 内部实现
    def _execute_with_timeout(self, spec: ToolSpec[Any, Any], args: Any, timeout_s: float) -> Any:
        """用独立线程执行工具，从而获得「硬超时」能力。"""
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = spec.handler(args)
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc

        worker = threading.Thread(
            target=target, name=f"tool-{spec.name}", daemon=True
        )
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            raise ToolTimeout(
                f"执行超过 {timeout_s}s 仍未返回（下游可能挂住了）", tool=spec.name
            )
        if "exc" in box:
            raise box["exc"]
        return box.get("value")

    @staticmethod
    def _normalize(exc: BaseException, tool: str) -> ToolError:
        if isinstance(exc, ToolError):
            return exc
        return ToolExecutionError(
            f"{type(exc).__name__}: {exc}",
            tool=tool,
            retryable=isinstance(exc, _RETRYABLE_BUILTINS),
        )

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1000.0)

    def _fail(
        self,
        name: str,
        raw: Mapping[str, Any],
        attempts: int,
        error: ToolError,
        started: float,
        trace_id: str | None,
    ) -> ToolCallResult:
        duration = self._elapsed_ms(started)
        if error.retryable or attempts:
            self._on_failure(name)
        self.audit.log_call(
            tool=name,
            args=dict(raw),
            ok=False,
            attempts=attempts,
            duration_ms=duration,
            trace_id=trace_id,
            error_code=error.code,
            error_message=error.message,
        )
        return ToolCallResult(
            tool=name,
            ok=False,
            error=error.message,
            error_code=error.code,
            retryable=error.retryable,
            attempts=attempts,
            duration_ms=round(duration, 3),
            trace_id=trace_id,
        )

    # ------------------------------------------------------------ 熔断器
    def _check_circuit(self, name: str) -> None:
        now = self._clock()
        with self._lock:
            opened = self._opened_at.get(name)
            if opened is None:
                return
            if now - opened < self.circuit_cooldown_s:
                remain = self.circuit_cooldown_s - (now - opened)
                raise ToolCircuitOpen(
                    f"{name} 处于熔断状态，{remain:.1f}s 后自动进入半开试探",
                    tool=name,
                )
            # 冷却结束：进入半开，放一次请求过去试探
            self._opened_at.pop(name, None)
            self._failures[name] = []
            self._half_open.add(name)

    def _on_success(self, name: str) -> None:
        with self._lock:
            self._failures.pop(name, None)
            self._opened_at.pop(name, None)
            self._half_open.discard(name)

    def _on_failure(self, name: str) -> None:
        now = self._clock()
        with self._lock:
            if name in self._half_open:
                # 半开试探也失败 —— 说明下游还没恢复，立刻重新熔断
                self._half_open.discard(name)
                self._opened_at[name] = now
                self._failures[name] = [now]
                return
            window = [t for t in self._failures.get(name, []) if now - t <= self.circuit_window_s]
            window.append(now)
            self._failures[name] = window
            if len(window) >= self.circuit_failure_threshold and name not in self._opened_at:
                self._opened_at[name] = now

    def circuit_state(self) -> dict[str, str]:
        with self._lock:
            names = set(self._failures) | set(self._opened_at) | self._half_open
            state: dict[str, str] = {}
            for name in names:
                if name in self._opened_at:
                    state[name] = "open"
                elif name in self._half_open:
                    state[name] = "half_open"
                else:
                    state[name] = "closed"
            return state


def _coerce_args(args: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, str):
        text = args.strip() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"参数不是合法 JSON：{exc}", retryable=False
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolExecutionError("参数必须是 JSON 对象", retryable=False)
        return parsed
    return dict(args)


def _rebuild(result: ToolCallResult) -> ToolError:
    """由结果反推一个异常对象，保持 call() 的抛错语义。"""
    mapping = {
        "tool_timeout": ToolTimeout,
        "tool_circuit_open": ToolCircuitOpen,
        "tool_budget_exceeded": ToolBudgetExceeded,
    }
    cls = mapping.get(result.error_code or "", ToolExecutionError)
    exc = cls(result.error or "工具调用失败", tool=result.tool)
    return exc
