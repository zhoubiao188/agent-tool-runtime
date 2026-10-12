"""审计与幂等存储。

Agent 出了问题时，最没用的答案是「模型调了工具然后出错了」。
你需要的是：**哪一次请求、调了哪个工具、参数是什么、试了几次、
每次耗时多少、失败原因是什么、结果摘要是什么。**

这些信息必须落库，而且必须结构化。本模块用 SQLite 实现，
单文件、零依赖、可直接换成 PostgreSQL。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id      TEXT,
    tool          TEXT    NOT NULL,
    args_json     TEXT    NOT NULL,
    ok            INTEGER NOT NULL,
    attempts      INTEGER NOT NULL,
    duration_ms   REAL    NOT NULL,
    error_code    TEXT,
    error_message TEXT,
    cached        INTEGER NOT NULL DEFAULT 0,
    result_digest TEXT,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_audit_tool ON tool_audit(tool, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_audit_trace ON tool_audit(trace_id);

CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT PRIMARY KEY,
    tool        TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class AuditStore:
    """线程安全的 SQLite 审计 + 幂等结果存储。"""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -------------------------------------------------------------- 写审计
    def log_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        ok: bool,
        attempts: int,
        duration_ms: float,
        trace_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        cached: bool = False,
        result_digest: str | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO tool_audit
                    (trace_id, tool, args_json, ok, attempts, duration_ms,
                     error_code, error_message, cached, result_digest, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    tool,
                    json.dumps(args, ensure_ascii=False, sort_keys=True, default=str),
                    1 if ok else 0,
                    attempts,
                    round(duration_ms, 3),
                    error_code,
                    error_message,
                    int(cached),
                    result_digest,
                    time.time(),
                ),
            )
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------------ 读审计
    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_audit ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, dict[str, float]]:
        """按工具聚合成功率与平均耗时——这就是你的 Agent 版 SLA 面板。"""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tool,
                       COUNT(*)                AS calls,
                       SUM(ok)                 AS ok_calls,
                       AVG(duration_ms)        AS avg_ms,
                       MAX(duration_ms)        AS max_ms,
                       SUM(attempts)           AS total_attempts
                FROM tool_audit
                GROUP BY tool
                ORDER BY calls DESC
                """
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            calls = r["calls"] or 0
            # 注意：未执行到工具就失败的调用（未知工具、参数校验失败）attempts=0，
            # 直接相减会得到负数，必须夹紧到 0。
            retries = max(0, (r["total_attempts"] or 0) - calls)
            out[r["tool"]] = {
                "calls": calls,
                "ok": r["ok_calls"] or 0,
                "success_rate": round((r["ok_calls"] or 0) / calls, 4) if calls else 0.0,
                "avg_ms": round(r["avg_ms"] or 0.0, 3),
                "max_ms": round(r["max_ms"] or 0.0, 3),
                "retries": retries,
            }
        return out

    # ------------------------------------------------------------ 幂等缓存
    def get_cached(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT tool, result_json FROM idempotency WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return {"tool": row["tool"], "value": json.loads(row["result_json"])}

    def put_cached(self, key: str, tool: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO idempotency (key, tool, result_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (key, tool, json.dumps(value, ensure_ascii=False, default=str), time.time()),
            )

    def clear_cache(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM idempotency")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
