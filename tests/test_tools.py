"""内置工具测试：计算器的安全问题、SQL 的只读边界、HTTP 的可注入 transport。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from agent_tool_runtime.audit import AuditStore
from agent_tool_runtime.errors import ToolExecutionError
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.runtime import ToolRuntime
from agent_tool_runtime.testing import FakeClock, RecordingSleeper
from agent_tool_runtime.tools import build_default_registry, create_demo_db
from agent_tool_runtime.tools.calculator import CALCULATOR, CalcArgs
from agent_tool_runtime.tools.http import HttpGetArgs, make_http_get_tool
from agent_tool_runtime.tools.sql import make_sql_tool


def _runtime(registry: ToolRegistry | None = None) -> ToolRuntime:
    return ToolRuntime(
        registry or ToolRegistry(),
        AuditStore(":memory:"),
        sleeper=RecordingSleeper(),
        clock=FakeClock(),
    )


# ------------------------------------------------------------------ 计算器


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+2*3", 7),
        ("(1250-980)/980*100", 27.551020408163264),
        ("sqrt(2)", 2 ** 0.5),
        ("2**10", 1024),
        ("max(3, 7, 5) - min(1, 2)", 6),
        ("pi", 3.141592653589793),
    ],
)
def test_calculator_happy_path(expression: str, expected: float) -> None:
    assert CALCULATOR.handler(CalcArgs(expression=expression))["value"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "lambda: 1",
        "[x for x in range(3)]",
    ],
)
def test_calculator_rejects_code_injection(expression: str) -> None:
    """计算器的输入是模型生成的字符串，必须当作不可信输入对待。"""
    with pytest.raises(ToolExecutionError):
        CALCULATOR.handler(CalcArgs(expression=expression))


def test_calculator_guardrails_on_huge_numbers() -> None:
    with pytest.raises(ToolExecutionError, match="数量级过大"):
        CALCULATOR.handler(CalcArgs(expression="10**100"))


# ------------------------------------------------------------------ SQL


def test_sql_reads_seeded_orders(tmp_path: Path) -> None:
    db = create_demo_db(tmp_path / "orders.db")
    tool = make_sql_tool(db)
    result = tool.handler(
        tool.args_model(sql="SELECT product, SUM(amount) AS revenue FROM orders GROUP BY product ORDER BY revenue DESC")
    )
    assert result["columns"] == ["product", "revenue"]
    assert result["rows"][0]["product"] == "企业版"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET amount = 0",
        "INSERT INTO orders (id) VALUES (99)",
        "SELECT 1; DROP TABLE orders",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
    ],
)
def test_sql_blocks_writes(tmp_path: Path, sql: str) -> None:
    db = create_demo_db(tmp_path / "orders.db")
    tool = make_sql_tool(db)
    with pytest.raises(ToolExecutionError):
        tool.handler(tool.args_model(sql=sql))


def test_sql_allows_created_at_column(tmp_path: Path) -> None:
    """回归用例：关键字黑名单必须按词边界匹配，否则 created_at 会被误杀。"""
    db = create_demo_db(tmp_path / "orders.db")
    tool = make_sql_tool(db)
    result = tool.handler(tool.args_model(sql="SELECT city, created_at FROM orders LIMIT 2"))
    assert result["row_count"] == 2


def test_sql_limits_returned_rows(tmp_path: Path) -> None:
    db = create_demo_db(tmp_path / "orders.db")
    tool = make_sql_tool(db)
    result = tool.handler(tool.args_model(sql="SELECT * FROM orders", limit=3))
    assert result["row_count"] == 3


def test_sql_database_connection_is_readonly(tmp_path: Path) -> None:
    db = create_demo_db(tmp_path / "orders.db")
    tool = make_sql_tool(db)
    with pytest.raises(ToolExecutionError):
        tool.handler(tool.args_model(sql="SELECT * FROM no_such_table"))


# ------------------------------------------------------------------ HTTP


def test_http_tool_retries_transient_network_errors() -> None:
    attempts = {"n": 0}

    def flaky_transport(url: str, timeout_s: float) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ToolExecutionError("连接被重置", retryable=True)
        return {"status": 200, "content_type": "application/json", "body": '{"ok": true}'}

    registry = ToolRegistry()
    registry.add(make_http_get_tool(transport=flaky_transport, max_retries=3))
    runtime = _runtime(registry)
    result = runtime.invoke("http_get", {"url": "https://example.com/api"})

    assert result.ok is True
    assert result.attempts == 3
    assert json.loads(result.value["body"]) == {"ok": True}


def test_http_tool_rejects_non_http_scheme() -> None:
    with pytest.raises(Exception):
        HttpGetArgs(url="file:///etc/passwd")


def test_http_tool_truncates_large_body() -> None:
    def big_transport(url: str, timeout_s: float) -> dict[str, Any]:
        return {"status": 200, "content_type": "text/plain", "body": "A" * 100_000}

    registry = ToolRegistry()
    registry.add(make_http_get_tool(transport=big_transport))
    runtime = _runtime(registry)
    result = runtime.invoke("http_get", {"url": "https://example.com/big", "max_bytes": 1024})
    assert result.value["truncated"] is True
    assert len(result.value["body"]) <= 1024


def test_http_tool_against_real_local_server() -> None:
    """端到端：真的起一个 HTTP 服务，验证默认 urllib transport 可用。"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            payload = json.dumps({"service": "orders", "status": "healthy"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # noqa: ANN002
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        registry = ToolRegistry()
        registry.add(make_http_get_tool(max_retries=1))
        runtime = _runtime(registry)
        result = runtime.invoke("http_get", {"url": f"http://127.0.0.1:{port}/healthz"})
        assert result.ok is True
        assert result.value["status"] == 200
        assert json.loads(result.value["body"])["status"] == "healthy"
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ 默认注册表


def test_default_registry_shape(tmp_path: Path) -> None:
    registry = build_default_registry(db_path=tmp_path / "orders.db")
    assert registry.names() == ["calculator", "http_get", "sql_query"]
    exported = registry.openai_tools()
    assert all(t["type"] == "function" for t in exported)
    assert all("parameters" in t["function"] for t in exported)
