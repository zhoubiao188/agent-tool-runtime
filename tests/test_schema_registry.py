from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent_tool_runtime.errors import ToolNotFound, ToolValidationError
from agent_tool_runtime.registry import ToolRegistry
from agent_tool_runtime.schema import ToolSpec, tool


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: float
    b: float = Field(default=0.0)


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1, max_length=20)


def _add(args: AddArgs) -> float:
    """两数相加。"""
    return args.a + args.b


def test_spec_exports_valid_json_schema() -> None:
    spec = ToolSpec("add", "两数相加", AddArgs, _add)
    schema = spec.parameters_schema()
    assert schema["type"] == "object"
    assert set(schema["properties"]) >= {"a", "b"}
    assert "title" not in schema  # 已剥离 pydantic 噪音字段
    assert schema["required"] == ["a"]

    as_tool = spec.to_openai_tool()
    assert as_tool["type"] == "function"
    assert as_tool["function"]["name"] == "add"


def test_spec_rejects_bad_declarations() -> None:
    with pytest.raises(ValueError):
        ToolSpec("bad name!", "x", AddArgs, _add)
    with pytest.raises(ValueError):
        ToolSpec("ok", "x", AddArgs, _add, timeout_s=0)
    with pytest.raises(TypeError):
        ToolSpec("ok", "x", AddArgs, lambda x, y: x)  # 参数个数不对
    with pytest.raises(TypeError):
        ToolSpec("ok", "x", dict, _add)  # args_model 不是 BaseModel


def test_parse_reports_readable_validation_error() -> None:
    spec = ToolSpec("query", "查询", QueryArgs, lambda a: a.keyword)
    with pytest.raises(ToolValidationError) as exc:
        spec.parse({"keyword": ""})
    assert "keyword" in str(exc.value)

    with pytest.raises(ToolValidationError):
        spec.parse({"keyword": "ok", "unexpected": 1})  # type: ignore[arg-type]


def test_tool_decorator_uses_docstring_as_description() -> None:
    decorated = tool(AddArgs)(_add)
    assert decorated.name == "_add"
    assert decorated.description.splitlines()[0].startswith("两数相加")


def test_registry_detects_name_collision(registry: ToolRegistry) -> None:
    a = ToolSpec("dup", "第一个", AddArgs, _add)
    b = ToolSpec("dup", "第二个", AddArgs, _add)
    registry.add(a)
    with pytest.raises(ValueError, match="工具名冲突"):
        registry.add(b)
    registry.add(b, override=True)  # 显式覆盖是允许的
    assert len(registry) == 1


def test_registry_raises_tool_not_found_with_hint(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("add", "两数相加", AddArgs, _add))
    with pytest.raises(ToolNotFound) as exc:
        registry.get("missing")
    assert "add" in str(exc.value)  # 报错里带出可用工具名，排查成本骤降


def test_registry_subset_builds_per_role_toolset(registry: ToolRegistry) -> None:
    registry.add(ToolSpec("add", "两数相加", AddArgs, _add))
    registry.add(ToolSpec("query", "查询", QueryArgs, lambda a: a.keyword))
    sub = registry.subset(["query"])
    assert sub.names() == ["query"]
    assert sub.openai_tools()[0]["function"]["name"] == "query"
