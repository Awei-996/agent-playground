# Chapter 01：ReAct 旅行助手

一个基于 **ReAct**（Reasoning + Acting）模式的入门 Agent 示例：LLM 按 `Thought → Action → Observation` 循环推理，调用工具查询天气并推荐景点。

## 项目结构

```
code/
├── main.py                    # v1 Agent 主循环
├── OpenAICompatibleClient.py  # 兼容 OpenAI 接口的 LLM 客户端
├── tools.py                   # 工具定义（天气、景点搜索）
├── .env                       # API 密钥与环境变量（勿提交）
├── pyproject.toml             # 依赖管理（uv）
└── v2/                        # v2 优化版（独立目录，见 v2/README.md）
```

## v2 优化版

v2 位于 [`v2/`](v2/)，与 v1 完全隔离，实现了 Tool Calling 优先 + 文本 ReAct 自动回退 + 规范多轮 messages。

```bash
cd hello-agents/chapter-01/code/v2
cp ../.env .env    # 首次配置（可选，也可自动回退 ../.env）
uv sync
uv run python main.py
```

详见 [v2/README.md](v2/README.md)。

## 快速开始

### 1. 安装依赖

```bash
cd hello-agents/chapter-01/code
uv sync
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
TAVILY_API_KEY=你的Tavily密钥
```

### 3. 运行

```bash
uv run python main.py
```

程序会循环调用 LLM，执行 `get_weather`、`get_attraction` 等工具，直到输出 `Finish[最终答案]` 或达到 5 轮上限。

---

## 开发备忘

### uv 装了依赖，IDE 仍提示找不到包？

`uv sync` 会把包装进项目内的 `.venv`，但 Cursor / VS Code 的 Python 解释器可能仍指向系统 Python。

**解决方法：**

1. `Cmd+Shift+P` → **Python: Select Interpreter** → 选择 `.venv/bin/python`
2. 或在仓库根目录 `.vscode/settings.json` 中指定解释器路径
3. 终端验证：`uv run python -c "import requests; print('OK')"`

### 启动报错 `Missing credentials`？

`main.py` 通过 `os.environ.get()` 读取配置，Python **不会自动**加载 `.env` 文件。

本项目已在代码中使用 `python-dotenv`：

```python
from dotenv import load_dotenv
load_dotenv()
```

若未加载 dotenv，也可显式指定：

```bash
uv run --env-file .env python main.py
```

---

## 核心概念

### Chat Completions 的消息角色

`OpenAICompatibleClient` 每次调用构造两条消息：

```python
messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': prompt}
]
```

| 角色 | 作用 | 本项目中的内容 |
|------|------|----------------|
| `system` | 长期规则：身份、工具说明、输出格式 | `AGENT_SYSTEM_PROMPT` |
| `user` | 当前任务与上下文 | 用户请求 + 历史 Thought/Action/Observation |
| `assistant` | 模型历史回复 | 未单独使用，而是拼入 `user` 字符串 |
| `tool` | 工具返回结果 | 未单独使用，以 `Observation:` 文本形式写入 `user` |

**为何分角色？** 模型在训练时学习了不同角色的语义权重：`system` 约束行为方式，`user` 提供待处理内容。本项目采用简化版 ReAct——把模型输出和环境反馈都拼进一个 `user` 消息，便于理解原理。

### Chat Completions 还有哪些用法？

除当前的 `system + user` 文本 ReAct 外，同一 API 还支持：

- **多轮 messages**：将模型回复标为 `assistant`，比全拼进 `user` 更规范
- **Tool Calling**：用 `tools` 参数 + `tool` 角色替代文本解析 `Action: get_weather(...)`
- **流式输出**：`stream=True`，逐 token 返回
- **多模态**：`content` 数组中混合文本与图片
- **结构化输出**：`response_format` 约束 JSON 格式

其他 API 路线：

| API | 适用场景 |
|-----|----------|
| Completions（旧） | 单一 prompt 补全，已逐步淘汰 |
| Responses API | OpenAI 新一代接口，内置工具与推理链 |
| Embeddings | 文本向量化，RAG 检索 |
| Assistants API | 服务端托管对话与工具 |

演进路径参考：**文本 ReAct → 多轮 messages → Tool Calling → Agent 框架**。

---

## 工作流程

```
用户请求
    ↓
┌─ 循环（最多 5 轮）──────────────────────┐
│  LLM 输出 Thought + Action              │
│      ↓                                  │
│  解析 Action                            │
│      ├─ Finish[...]  → 结束             │
│      └─ 调用工具 → Observation → 下一轮  │
└─────────────────────────────────────────┘
    ↓
最终答案
```
