"""工具注册中心。

只有一件事需要强调：**注册时就要做冲突检测**。
模型拿到的工具名是扁平字符串，一旦出现同名工具，
模型可能调用到「你没想让它调」的那一个——这类事故排查起来极其昂贵。
"""

from __future__ import annotations

from typing import Any, Iterator

from .errors import ToolNotFound
from .schema import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec[Any, Any]] = {}

    # ---------------------------------------------------------------- 注册
    def add(self, spec: ToolSpec[Any, Any], *, override: bool = False) -> "ToolRegistry":
        if spec.name in self._tools and not override:
            raise ValueError(f"工具名冲突：{spec.name!r} 已注册")
        self._tools[spec.name] = spec
        return self

    def extend(self, specs: list[ToolSpec[Any, Any]], *, override: bool = False) -> "ToolRegistry":
        for spec in specs:
            self.add(spec, override=override)
        return self

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    # ---------------------------------------------------------------- 查询
    def get(self, name: str) -> ToolSpec[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools)) or "<空>"
            raise ToolNotFound(
                f"未知工具 {name!r}，可用工具：{available}", tool=name
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec[Any, Any]]:
        return [self._tools[n] for n in self.names()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """一次性导出给模型看的工具清单。"""
        return [spec.to_openai_tool() for spec in self.specs()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        """按名字裁剪出一个子注册表——实现「按角色/场景下发不同工具集」。"""
        reg = ToolRegistry()
        for n in names:
            reg.add(self.get(n))
        return reg

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec[Any, Any]]:
        return iter(self.specs())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolRegistry {len(self._tools)} tools: {', '.join(self.names())}>"
