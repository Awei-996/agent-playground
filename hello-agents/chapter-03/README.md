# Chapter 03：BPE 分词器入门

本章通过一段可运行的 BPE（Byte Pair Encoding）代码，理解大语言模型如何把文本切成 token，以及分词器在 Agent 对话中的位置。

## 文件说明

| 文件 | 说明 |
|------|------|
| `bpe.py` | BPE 训练的核心逻辑：统计相邻词元对、合并最高频对 |
| `hf_demo.py` | Hugging Face 本地推理：加载 Qwen 对话模型并生成回答 |
| `hf_demo.md` | `hf_demo.py` 的运行说明、流程与原理 |
| `main.py` | 章节入口（占位） |

## 运行

本章使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境（见 `pyproject.toml`、`.python-version`）。

```bash
cd hello-agents/chapter-03

# 首次：创建虚拟环境并安装依赖（当前无第三方依赖）
uv sync

# 运行 BPE 示例
uv run python bpe.py
```

等价写法（先激活虚拟环境再运行）：

```bash
uv sync
uv venv --python 3.13   # uv sync 已自动创建 .venv 时可跳过
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python bpe.py
```

> 不要直接用系统 `python bpe.py`，否则会绕过项目虚拟环境，Python 版本也可能与 `.python-version`（3.13）不一致。

输出每次合并选中的词元对，以及合并后的词表。

### Hugging Face 本地推理 Demo

```bash
uv run python hf_demo.py
```

详见 [hf_demo.md](./hf_demo.md)（依赖 `torch`、`transformers`，首次运行需下载模型）。

---

## 三个核心概念

### 1. 分词规则（Merge Rules）

BPE **训练**的产物，记录「按什么顺序、把哪些相邻片段合并」。

本例语料训练 4 次后，得到 4 条规则：

```
规则 1:  u   + g    → ug
规则 2:  ug  + </w>  → ug</w>
规则 3:  u   + n    → un
规则 4:  un  + </w>  → un</w>
```

规则有**严格顺序**：编码新文本时，必须按训练时的顺序依次尝试合并。

### 2. 子词（Subword）

不是整词，也不一定是单字符，而是语料中**反复出现的常用片段**。

| 子词 | 含义 |
|------|------|
| `ug` | 词中间的 `ug` |
| `ug</w>` | 以 `ug` 结尾的词（`</w>` 表示词尾） |
| `un` | 词中间的 `un` |
| `un</w>` | 以 `un` 结尾的词 |

训练结束后，每个词被切成 2 段：

```
hug  →  [ h ]  [ ug</w> ]
pug  →  [ p ]  [ ug</w> ]
pun  →  [ p ]  [ un</w> ]
bun  →  [ b ]  [ un</w> ]
```

公共后缀（`ug</w>`、`un</w>`）被多个词共享，这就是子词的价值。

### 3. 词表（Vocabulary）

模型能认识的**全部 token 及其 ID**。

```
初始词表（单字符）:  h, u, g, p, n, b, </w>
训练新增子词:        ug, ug</w>, un, un</w>
```

真实 GPT 词表有数万～十几万个 token，原理相同，只是规模更大。

---

## 完整训练流程（本例语料）

语料：`hug`、`pug`、`pun`、`bun`（各出现 1 次）。

初始表示（字符用空格分开，词尾加 `</w>`）：

```
h u g </w>
p u g </w>
p u n </w>
b u n </w>
```

### 第 1 轮：统计 → 合并 `u + g → ug`

统计相邻对频率（节选）：

```
('u','g'): 2    ← hug、pug 都有，频次最高（并列时取字典中先出现的）
('u','n'): 2
('g','</w>'): 2
...
```

合并后：

```
h ug </w>
p ug </w>
p u n </w>
b u n </w>
```

### 第 2 轮：合并 `ug + </w> → ug</w>`

```
h ug</w>
p ug</w>
p u n </w>
b u n </w>
```

### 第 3 轮：合并 `u + n → un`

```
h ug</w>
p ug</w>
p un </w>
b un </w>
```

### 第 4 轮：合并 `un + </w> → un</w>`

```
h ug</w>
p ug</w>
p un</w>
b un</w>
```

---

## 代码说明

### `get_stats(vocab)` — 统计词元对频率

遍历词表中每个词，统计所有**相邻词元对**的出现次数。

```python
# 例：'h u g </w>' 贡献 ('h','u')、('u','g')、('g','</w>') 各 +1
pairs[symbols[i], symbols[i+1]] += freq
```

### `merge_vocab(pair, v_in)` — 执行合并

在词表中，把 `pair[0] pair[1]`（带空格）替换为 `pair[0]pair[1]`（无空格）。

```python
# 例：('u','g') 合并后，'h u g </w>' → 'h ug </w>'
```

正则 `(?<!\S)...(?!\S)` 确保只匹配完整的相邻词元，不会误匹配子串。

### 主循环

```python
for i in range(num_merges):
    pairs = get_stats(vocab)           # 统计
    best = max(pairs, key=pairs.get)     # 选最高频对（并列取第一个）
    vocab = merge_vocab(best, vocab)     # 合并
```

---

## 训练 vs 使用：分词器什么时候用？

### 训练阶段（离线，做一次）

```
大量语料 → BPE 训练 → 分词规则 + 词表 → 保存为 tokenizer 文件
```

`bpe.py` 模拟的就是这个过程。OpenAI、Meta 等在发布模型前已完成，使用者直接下载。

### 推理阶段（在线，每次对话都用）

凡是进出模型的文本，都要经过分词器：

```
用户输入 / 系统提示 / 历史对话 / 工具描述
        ↓ encode（文字 → token ID）
      模型计算
        ↓ decode（token ID → 文字）
    显示给用户的回复
```

**Agent 对话属于推理阶段，分词器一直在工作。**

典型 Agent 一轮对话中，以下部分都会 tokenize：

- 用户消息
- System prompt
- 历史上下文
- Tool / Function 定义
- 工具返回结果
- 模型生成的回复

以下环节**不需要**分词器：调用搜索引擎、查数据库等外部工具本身。

### 为什么 Agent 开发者需要了解 token？

| 场景 | 说明 |
|------|------|
| Context 窗口 | 「128K 上下文」指 128K 个 **token**，不是 128K 个汉字 |
| API 计费 | 按输入 token + 输出 token 计费 |
| 截断策略 | 历史太长时，需 tokenize 后计算长度再截断 |

---

## 编码新词：规则怎么用？

训练完成后，遇到**没见过的词**，按 merge 规则顺序切分，而不是查整词表。

### 例：`mug`（训练集里没有）

```
初始:     m u g </w>
规则 1:   m ug </w>       (u + g → ug)
规则 2:   m ug</w>        (ug + </w> → ug</w>)
结果:     [ m ] [ ug</w> ]
```

和 `hug → [h][ug</w]`、`pug → [p][ug</w>` 模式一致：首字母 + 共享词尾子词。

### 例：`up`（无法合并）

```
初始:     u p </w>
规则 1:   需要 u 后面是 g → 不适用
规则 3:   需要 u 后面是 n → 不适用
结果:     [ u ] [ p ] [ </w> ]
```

---

## 与真实 GPT Tokenizer 的对应

| | 本例 `bpe.py` | 真实 GPT Tokenizer |
|---|--------------|-------------------|
| 语料规模 | 4 个词 | 海量文本 |
| 合并次数 | 4 次 | 数万次 |
| 词表大小 | ~11 个 token | ~10 万个 token |
| 训练产物 | 注释中手算的流程 | `merges.txt` + `vocab.json` |
| 使用方式 | 同一套 encode/decode 逻辑 | `tiktoken` / `transformers` 库 |

---

## 一句话总结

> **BPE 从字符出发，反复合并语料中最常相邻出现的片段，学到一套分词规则和词表。Agent 每次和模型对话时，都通过这套规则把文字翻译成 token，再把模型输出翻译回文字。**

分词器是「人类文字」和「模型数字」之间的翻译官。
