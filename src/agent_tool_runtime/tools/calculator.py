"""内置工具之一：计算器。

不调用 ``eval``，而是把表达式解析成 AST 后白名单求值。
Agent 场景下的输入是「模型生成的字符串」，直接 eval 等于把 RCE 递到模型手上。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ToolExecutionError
from ..schema import ToolSpec, tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_MAX_ABS = 1e18


class CalcArgs(BaseModel):
    expression: str = Field(
        description="要计算的数学表达式，例如 '(1250-980)/980*100' 或 'sqrt(2)'",
        min_length=1,
        max_length=500,
    )


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolExecutionError("只支持数字常量", retryable=False)
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(
                f"不支持的运算符：{type(node.op).__name__}", retryable=False
            )
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise ToolExecutionError("不支持的一元运算", retryable=False)
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ToolExecutionError(f"未知标识符：{node.id}", retryable=False)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ToolExecutionError("只允许使用白名单数学函数", retryable=False)
        if node.keywords:
            raise ToolExecutionError("不支持关键字参数", retryable=False)
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ToolExecutionError(
        f"不支持的语法节点：{type(node).__name__}（出于安全考虑已拒绝）",
        retryable=False,
    )


@tool(
    CalcArgs,
    description="计算数学表达式，支持 + - * / // % ** 与 sqrt/log/abs/min/max 等函数。",
    timeout_s=2.0,
    max_retries=0,
    idempotent=True,
    tags=("local", "pure"),
)
def calculator(args: CalcArgs) -> dict[str, Any]:
    try:
        tree = ast.parse(args.expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(f"表达式语法错误：{exc.msg}", retryable=False) from exc
    value = _eval_node(tree)
    if isinstance(value, complex):
        raise ToolExecutionError("结果不是实数", retryable=False)
    if abs(value) > _MAX_ABS:
        raise ToolExecutionError("计算结果数量级过大，已拒绝", retryable=False)
    return {"expression": args.expression, "value": value}


CALCULATOR: ToolSpec[CalcArgs, dict[str, Any]] = calculator
