"""内置工具之二：只读 SQL 查询。

把「SQL 工具」交给 Agent 时，最大的风险不是它查得慢，而是它写了 ``DROP TABLE``。
本模块用三层防御：

1. 语法层：只允许单条 SELECT / WITH 语句，关键字黑名单；
2. 连接层：以 ``mode=ro`` 打开数据库，并开启 ``query_only``；
3. 运行层：交给 Tool Runtime 的硬超时兜底。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ToolExecutionError
from ..schema import ToolSpec

_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create",
    "replace", "attach", "detach", "pragma", "vacuum", "truncate",
)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE
)


class SqlArgs(BaseModel):
    sql: str = Field(
        description="单条只读 SELECT 查询，例如 'SELECT city, SUM(amount) FROM orders GROUP BY city'",
        min_length=6,
        max_length=2000,
    )
    limit: int = Field(default=50, ge=1, le=500, description="最多返回多少行")


def create_demo_db(path: str | Path) -> str:
    """建一个演示用数据库（订单表），返回文件路径。"""
    p = Path(path)
    if p.exists():
        p.unlink()
    conn = sqlite3.connect(p)
    try:
        conn.executescript(
            """
            CREATE TABLE orders (
                id        INTEGER PRIMARY KEY,
                city      TEXT    NOT NULL,
                product   TEXT    NOT NULL,
                amount    REAL    NOT NULL,
                quantity  INTEGER NOT NULL,
                created_at TEXT   NOT NULL
            );
            """
        )
        rows = [
            (1, "杭州", "企业版", 12800.0, 2, "2026-09-01"),
            (2, "上海", "专业版", 4980.0, 3, "2026-09-02"),
            (3, "杭州", "专业版", 4980.0, 1, "2026-09-05"),
            (4, "深圳", "企业版", 12800.0, 1, "2026-09-08"),
            (5, "上海", "企业版", 12800.0, 4, "2026-09-11"),
            (6, "成都", "基础版", 980.0, 10, "2026-09-15"),
            (7, "杭州", "基础版", 980.0, 6, "2026-09-18"),
            (8, "深圳", "专业版", 4980.0, 2, "2026-09-22"),
            (9, "北京", "企业版", 12800.0, 3, "2026-09-25"),
            (10, "北京", "基础版", 980.0, 8, "2026-09-29"),
        ]
        conn.executemany(
            "INSERT INTO orders (id, city, product, amount, quantity, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return str(p)


def _assert_safe(sql: str) -> str:
    text = sql.strip().rstrip(";").strip()
    statements = [s for s in text.split(";") if s.strip()]
    if len(statements) != 1:
        raise ToolExecutionError("只允许执行单条语句", retryable=False)
    lowered = text.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ToolExecutionError("只允许 SELECT / WITH 查询", retryable=False)
    hit = _FORBIDDEN_RE.search(text)
    if hit:
        raise ToolExecutionError(
            f"检测到禁止使用的关键字：{hit.group(0)}", retryable=False
        )
    return text


def make_sql_tool(db_path: str | Path, *, timeout_s: float = 3.0) -> ToolSpec[SqlArgs, dict[str, Any]]:
    """绑定到一个数据库文件的只读 SQL 工具。"""

    def handler(args: SqlArgs) -> dict[str, Any]:
        statement = _assert_safe(args.sql)
        uri = f"file:{Path(db_path).resolve()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
        except sqlite3.Error as exc:
            raise ToolExecutionError(f"无法打开数据库：{exc}", retryable=False) from exc
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(statement)
            columns = [d[0] for d in (cursor.description or [])]
            rows = [dict(r) for r in cursor.fetchmany(args.limit)]
            return {"columns": columns, "row_count": len(rows), "rows": rows}
        except sqlite3.Error as exc:
            # SQL 语法错误属于「模型写错了」，不可重试
            raise ToolExecutionError(f"SQL 执行失败：{exc}", retryable=False) from exc
        finally:
            conn.close()

    return ToolSpec(
        name="sql_query",
        description=(
            "对订单库执行只读 SQL 查询。表 orders 的字段："
            "id, city, product, amount, quantity, created_at。"
            "只允许单条 SELECT。"
        ),
        args_model=SqlArgs,
        handler=handler,
        timeout_s=timeout_s,
        max_retries=1,
        idempotent=True,
        tags=("database", "readonly"),
    )
