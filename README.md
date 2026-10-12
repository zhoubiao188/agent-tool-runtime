# agent-tool-runtime

> 给 AI Agent 的工具调用加一层生产级运行时：**参数校验 · 硬超时 · 指数退避重试 · 熔断 · 幂等去重 · 全链路审计**。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 它解决什么问题

大多数 Agent Demo 的工具调用长这样：

```python
result = globals()[tool_name](**json.loads(tool_args))   # 💥
```

这条线在演示里跑得很好，在生产里会依次炸出下面这些坑：

| 线上真实故障 | 裸调用的后果 | 本项目的处理 |
| --- | --- | --- |
| 模型把 `city` 传成了 `{"city": 123}` | 下游接口 500，或者更糟——写进脏数据 | Pydantic 校验，**根本不执行** |
| 下游接口卡住 90 秒 | 整个 Agent 会话被拖死 | 线程级**硬超时**，超时即放弃 |
| 下游偶发 503 / 网络抖动 | 一次失败，任务直接报错给用户 | 只对**可重试错误**做指数退避重试 |
| 下游彻底挂掉，Agent 还在疯狂重试 | 自己把自己打成 DDoS | **熔断器**：连续失败即拒绝调用 |
| 用户手抖点了两次「生成月报」 | 重复下单、重复发券、重复扣款 | **幂等键** + 结果缓存，第二次直接复用 |
| 出事了想复盘 | 只有一句「工具调用失败」 | SQLite **审计表**：参数/次数/耗时/错误码全留痕 |
| 模型开始死循环调工具 | 账单失控 | 单次运行**预算上限** |

## 架构

```
                    ┌──────────────┐
   question ───────▶│  ReActAgent  │
                    └──────┬───────┘
                           │ 1. 带上 tools 清单问模型
                           ▼
                    ┌──────────────┐
                    │   LLMClient  │  MockLLM / 任意 OpenAI 兼容端点
                    └──────┬───────┘
                           │ 2. 模型返回 tool_calls
                           ▼
                    ┌──────────────────────────────────────────────┐
                    │               ToolRuntime                    │
                    │                                              │
                    │  ① 预算检查 ──▶ ToolBudgetExceeded            │
                    │  ② 注册表查表 ─▶ ToolNotFound                 │
                    │  ③ 参数校验 ──▶ ToolValidationError（不执行）  │
                    │  ④ 幂等命中 ──▶ 直接返回缓存结果               │
                    │  ⑤ 熔断判断 ──▶ ToolCircuitOpen               │
                    │  ⑥ 线程硬超时 ─▶ ToolTimeout                   │
                    │  ⑦ 指数退避重试（仅 retryable）                │
                    │  ⑧ 审计落库 + 结果摘要                        │
                    └──────────────────┬───────────────────────────┘
                                       │ 3. ToolCallResult
                    ┌──────────────────▼───────────────────────────┐
                    │  把 {ok, error_code, retryable} 回灌给模型，   │
                    │  让模型自己决定「改参数重试」还是「放弃」        │
                    └──────────────────────────────────────────────┘
```

**设计上最重要的一个决定**：错误不是抛给调用方就完事，而是被结构化成
`{ok, error_code, retryable, error}` 回灌给模型。模型看到 `tool_validation_error`
就知道是自己参数写错了，看到 `tool_not_found` 就知道该换个工具。
这比在 prompt 里写一百句「不要乱调工具」都管用。

## 快速开始

```bash
git clone https://github.com/your-name/agent-tool-runtime.git
cd agent-tool-runtime
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                # 66 passed
python scripts/demo.py  # 端到端演示，无需 API Key、无需联网
```

## 用法

### 1. 声明一个工具

```python
from pydantic import BaseModel, Field
from agent_tool_runtime import tool

class RefundArgs(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{8}$", description="订单号")
    amount: float = Field(gt=0, le=100000, description="退款金额（元）")

@tool(RefundArgs, description="对指定订单发起退款", timeout_s=8.0,
      max_retries=2, idempotent=True)   # 退款必须幂等
def refund(args: RefundArgs) -> dict:
    return payment_gateway.refund(args.order_id, args.amount)
```

`idempotent=True` 意味着：只要调用方传了同样的幂等键，第二次调用不会真的打到支付网关。

### 2. 组装运行时

```python
from agent_tool_runtime import AuditStore, ToolRegistry, ToolRuntime

registry = ToolRegistry().add(refund)
runtime = ToolRuntime(
    registry,
    AuditStore("agent_audit.db"),   # 生产环境换成 PostgreSQL 即可
    max_calls_per_run=25,
)
```

### 3. 交给 Agent

```python
from agent_tool_runtime import MockLLM, ReActAgent, OpenAIChatLLM

agent = ReActAgent(OpenAIChatLLM(model="gpt-4o-mini"), runtime)
run = agent.run("把 ORD-20261012 的 299 元退给用户", trace_id=request_id)
print(run.trace())
print(run.answer)
```

`run.trace()` 的输出（来自真实运行）：

```
[1] sql_query({"sql": "SELECT city, SUM(amount) AS gmv ..."}) -> ok in 0.4ms attempts=1
[2] final: 杭州 GMV 最高（18760 元），其次是上海与深圳（各 17780 元）。
```

### 4. 打开审计面板

```python
for name, row in runtime.audit.stats().items():
    print(name, row)
```

```
tool                calls   ok     succ  retries    avg_ms
----------------------------------------------------------
calculator              1    1  100.00%        0     0.162
flaky_echo              1    1  100.00%        1    12.741
not_exist               1    0    0.00%        0     0.015
sql_query               1    1  100.00%        0     0.454
```

## 演示：四种能力实测

`python scripts/demo.py` 的真实输出（节选）：

```
▶ 场景 2｜瞬时故障：两次抽风被运行时悄悄重试掉，模型无感
[1] flaky_echo({"payload": "hi"}) -> ok in 154.3ms attempts=3

▶ 场景 3｜硬超时 + 熔断：阻止 Agent 把故障下游打穿
超时保护  -> ok=False code=tool_timeout 耗时=301.9ms
第 1 次调用 -> ok=False code=tool_execution_error attempts=1
第 2 次调用 -> ok=False code=tool_execution_error attempts=1
第 3 次调用 -> ok=False code=tool_execution_error attempts=1
第 4 次调用 -> ok=False code=tool_circuit_open attempts=0
熔断状态：{'broken_downstream': 'open', 'slow_echo': 'closed'}

▶ 场景 4｜幂等去重：同一条指令重复投递，只真正执行一次
第一次 -> call_no=1 cached=False
第二次 -> call_no=1 cached=True
```

## 测试

```
$ pytest -q
..................................................................       [100%]
66 passed in 1.30s
```

全部测试**离线可跑、无 sleep**：

* 时间通过 `FakeClock` 注入 → 熔断窗口 / 冷却期的断言不需要真的等 15 秒；
* 等待通过 `RecordingSleeper` 注入 → 可以精确断言退避序列是 `[0.05, 0.1]`；
* 网络通过 `transport` 注入 → 可以构造「前两次连接重置、第三次成功」；
* 模型通过 `MockLLM` 注入 → 可以构造「模型先编一个不存在的工具名」。

## 目录结构

```
agent-tool-runtime/
├── src/agent_tool_runtime/
│   ├── schema.py        # ToolSpec：Pydantic 入参模型 = 工具契约的唯一真源
│   ├── registry.py      # ToolRegistry：注册 + 冲突检测 + 导出 JSON Schema
│   ├── runtime.py       # ToolRuntime：本项目的核心
│   ├── audit.py         # SQLite 审计表 + 幂等结果表
│   ├── errors.py        # 错误分类（retryable 是类型的一部分）
│   ├── llm.py           # MockLLM / OpenAIChatLLM（零 SDK 依赖）
│   ├── agent.py         # 最小 ReAct 循环
│   ├── testing.py       # FakeClock / RecordingSleeper
│   └── tools/
│       ├── calculator.py   # AST 白名单求值，拒绝 eval
│       ├── sql.py          # 三层防御的只读 SQL 工具
│       ├── http.py         # 可注入 transport 的 HTTP GET
│       └── demo_tools.py   # 可复现故障的演示工具
├── tests/               # 66 个用例
└── scripts/demo.py
```

## 已知取舍

* **超时靠线程**：超时后工作线程不会被强杀，只是被抛弃。对纯计算任务
  这没问题，但如果工具会一直持有数据库连接，请把超时下沉到下游客户端。
  真正的解法是 `asyncio.wait_for`（协程可取消），本项目为保持零依赖没走这条路。
* **熔断器在进程内**：多副本部署时每个副本各自熔断。横向扩展时把它换成
  Redis 计数器（`INCR` + `EXPIRE`）即可，接口无需改动。
* **幂等缓存无 TTL**：`idempotency` 表会持续增长，生产环境请加定期清理。
* **审计写 SQLite**：单机够用；写入量大时应改为异步批量写入，避免
  审计本身成为请求路径上的瓶颈。

## License

MIT
