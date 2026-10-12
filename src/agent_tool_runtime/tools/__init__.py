"""内置工具集，以及一个「一键装好」的默认注册表。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..registry import ToolRegistry
from ..schema import ToolSpec
from .calculator import CALCULATOR
from .http import make_http_get_tool
from .sql import create_demo_db, make_sql_tool

__all__ = [
    "CALCULATOR",
    "build_default_registry",
    "create_demo_db",
    "make_http_get_tool",
    "make_sql_tool",
]


def build_default_registry(
    *,
    db_path: str | Path | None = None,
    include_http: bool = True,
) -> ToolRegistry:
    """组装演示用的默认工具集（计算器 + SQL + HTTP）。"""
    if db_path is None:
        db_path = Path(tempfile.mkdtemp(prefix="atr-demo-")) / "orders.db"
    create_demo_db(db_path)

    registry = ToolRegistry()
    registry.add(CALCULATOR)
    registry.add(make_sql_tool(db_path))
    if include_http:
        registry.add(make_http_get_tool())
    return registry


def as_specs(*specs: ToolSpec[Any, Any]) -> list[ToolSpec[Any, Any]]:
    return list(specs)
