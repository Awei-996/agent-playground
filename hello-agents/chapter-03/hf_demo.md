# Hugging Face 本地推理 Demo

本示例在 `bpe.py` 理解分词原理之后，用 **Hugging Face Transformers** 加载真实对话模型，走通「加载 → 格式化 → 编码 → 生成 → 解码」的完整链路。

## 文件说明

| 文件 | 说明 |
|------|------|
| `hf_demo.py` | 使用 Qwen1.5-0.5B-Chat 做本地对话生成的最小示例 |
| `bpe.py` | 从零实现 BPE，理解 token 从哪来（见 [README.md](./README.md)） |

## 依赖

见 `pyproject.toml`：

- `torch>=2.12.1`
- `transformers>=5.12.1`

首次运行会从 [Hugging Face Hub](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) 下载模型权重（约 1GB），之后使用本地缓存。

## 运行

本章使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境。

```bash
cd hello-agents/chapter-03

# 首次：创建虚拟环境并安装依赖
uv sync

# 运行 Demo
uv run python hf_demo.py
```

Windows PowerShell 等价命令：

```powershell
cd E:\agent-playground\hello-agents\chapter-03
uv sync
uv run python hf_demo.py
```

> 请使用 `hf_demo.py`，**不要**将脚本命名为 `transformers.py`。若当前目录存在同名文件，Python 会 import 本地脚本而非 Hugging Face 库，导致 `ImportError`。

### 下载较慢时（可选）

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run python hf_demo.py
```

### 设备

脚本会自动选择设备：

- 有 NVIDIA GPU + CUDA → `cuda`（推荐）
- 否则 → `cpu`（可运行，但加载与生成较慢）

## 预期输出

```
Using device: cuda
模型和分词器加载完成！
text: <|im_start|>system ... （Qwen Chat 模板格式）
编码后的输入文本:
{'input_ids': tensor(...), 'attention_mask': tensor(...)}
generated_ids: tensor(...)

模型的回答:
你好！我是...（模型自我介绍）
```

## 完整流程

```
messages（system / user 对话）
        ↓ apply_chat_template
text（按 Chat 模板拼好的 prompt 字符串）
        ↓ tokenizer 编码
model_inputs（input_ids 张量）
        ↓ model.generate（自回归续写）
generated_ids（prompt + 新生成 token）
        ↓ 切片去掉 prompt 部分
generated_ids（仅回答部分）
        ↓ batch_decode
response（可读文本）
```

## 代码说明

### 1. 加载模型与分词器

```python
model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
```

- `AutoTokenizer`：加载与模型配套的分词器（内部为 BPE 等子词方案的训练产物）
- `AutoModelForCausalLM`：**Decoder-only** 因果语言模型，适合对话与续写
- 分词器与模型必须来自同一 `model_id`

### 2. Chat 模板

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

| 参数 | 含义 |
|------|------|
| `tokenize=False` | 只返回格式化后的字符串，暂不转 token id |
| `add_generation_prompt=True` | 在末尾加上「轮到 assistant 回答」的提示 |

对话模型在训练时使用固定格式（如 `<\|im_start\|>system` 等标记），推理时必须保持一致。

### 3. 编码

```python
model_inputs = tokenizer([text], return_tensors="pt").to(device)
```

将文本转为 PyTorch 张量，主要字段：

- `input_ids`：每个 token 对应的整数 ID
- `attention_mask`：有效 token 为 1，padding 为 0

### 4. 生成

```python
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
```

`model.generate()` 原理：**自回归**——反复预测「下一个 token」，拼到序列末尾，直到遇到结束符或达到 `max_new_tokens` 上限。

```
循环: forward → 取最后位置 logits → 选一个 token → 拼到序列末尾
```

默认多为贪心解码（取概率最高的 token）；也可通过 `temperature`、`top_p`、`do_sample` 等参数控制采样。

### 5. 截取与解码

```python
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

`generate()` 返回的是 **prompt + 生成内容** 的完整 token 序列，需去掉输入部分再解码，否则会把 system / user 内容一并打印出来。

## 与 `bpe.py` 的关系

| | `bpe.py` | `hf_demo.py` |
|---|----------|--------------|
| 目的 | 理解 BPE 训练与 merge 规则 | 使用工业界现成的 tokenizer + 模型 |
| 分词 | 手算 merge、看词表如何形成 | `tokenizer.encode` / `decode` 一键完成 |
| 模型 | 无 | Qwen1.5 Decoder-only LM |
| 阶段 | 类比「离线训练分词器」 | **推理阶段**（Agent 对话属于此类） |

一句话：`bpe.py` 讲「token 怎么来的」；`hf_demo.py` 讲「有了 tokenizer 和模型之后，Agent 输入输出怎么跑通」。

## 架构说明

`Qwen/Qwen1.5-0.5B-Chat` 是 **Decoder-only** 架构（与 GPT、LLaMA 同族）：

- 只有 Decoder 层，无独立 Encoder
- **Causal mask**：每个位置只能看到左侧 token
- 训练目标：给定前文，预测下一个 token
- 推理方式：`generate()` 自回归续写

## 常见问题

### `ImportError: cannot import name 'AutoModelForCausalLM' from 'transformers'`

当前目录下有名为 `transformers.py` 的文件，与 Hugging Face 库冲突。请运行 `hf_demo.py`，或删除/重命名冲突文件。

### 显存不足

可换更小的模型，或在 CPU 上运行（较慢）。也可减小 `max_new_tokens`。

### 修改对话内容

编辑 `hf_demo.py` 中 `messages` 列表的 `content` 字段即可。

## 自定义参数（可选）

```python
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=256,      # 限制生成长度
    do_sample=True,          # 开启采样
    temperature=0.7,         # 越高越随机
    top_p=0.9,               # nucleus sampling
)
```

## 一句话总结

> **本 Demo 用 Hugging Face 加载 Decoder-only 对话模型，经 Chat 模板与分词器把多轮消息变成 token，再通过 `generate()` 自回归生成回答，最后解码回文字——这是本地 LLM 与 Agent 推理的最小闭环。**
