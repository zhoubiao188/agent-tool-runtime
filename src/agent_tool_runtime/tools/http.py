"""内置工具之三：HTTP GET。

关键设计：**把网络 I/O 抽象成可注入的 transport**。

这样测试里可以塞一个「第 1、2 次抛连接错误、第 3 次成功」的假 transport，
从而在没有网络、没有 mock server 的情况下验证重试逻辑。
线上则注入默认的 ``urllib`` transport。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from ..errors import ToolExecutionError
from ..schema import ToolSpec

Transport = Callable[[str, float], dict[str, Any]]

_ALLOWED_SCHEMES = ("http", "https")


class HttpGetArgs(BaseModel):
    url: str = Field(description="完整 URL，必须以 http:// 或 https:// 开头")
    timeout_s: float = Field(default=5.0, ge=0.1, le=30.0)
    max_bytes: int = Field(default=8192, ge=256, le=65536)

    @field_validator("url")
    @classmethod
    def _check_scheme(cls, value: str) -> str:
        if urllib.parse.urlparse(value).scheme not in _ALLOWED_SCHEMES:
            raise ValueError("只允许 http/https 协议")
        return value


def urllib_transport(url: str, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agent-tool-runtime/1.0 (+https://github.com/)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            return {
                "status": resp.status,
                "content_type": resp.headers.get("Content-Type", ""),
                "body": body.decode("utf-8", "replace"),
            }
    except urllib.error.HTTPError as exc:
        # 5xx / 429 属于下游暂时性故障，值得重试
        retryable = exc.code >= 500 or exc.code == 429
        raise ToolExecutionError(f"HTTP {exc.code}", retryable=retryable) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ToolExecutionError(f"网络错误：{exc}", retryable=True) from exc


def make_http_get_tool(
    *,
    transport: Transport | None = None,
    max_retries: int = 2,
    timeout_s: float = 5.0,
) -> ToolSpec[HttpGetArgs, dict[str, Any]]:
    tx = transport or urllib_transport

    def handler(args: HttpGetArgs) -> dict[str, Any]:
        raw = tx(args.url, min(args.timeout_s, timeout_s))
        body = raw.get("body", "")
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        body = str(body)
        truncated = len(body.encode("utf-8")) > args.max_bytes
        if truncated:
            body = body.encode("utf-8")[: args.max_bytes].decode("utf-8", "ignore")
        return {
            "status": raw.get("status"),
            "content_type": raw.get("content_type", ""),
            "truncated": truncated,
            "body": body,
        }

    return ToolSpec(
        name="http_get",
        description="对一个 http/https URL 发起 GET 请求并返回响应体（可能被截断）。",
        args_model=HttpGetArgs,
        handler=handler,
        timeout_s=timeout_s + 1.0,
        max_retries=max_retries,
        retry_backoff_s=0.2,
        idempotent=True,
        tags=("network", "readonly"),
    )
