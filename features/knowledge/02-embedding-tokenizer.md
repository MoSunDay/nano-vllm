Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 2: Embedding & Tokenizer

> 模型不认识文字，只认识数字。Tokenizer 负责切词，Embedding 负责翻译。

---

## 2.1 Tokenization —— 文字变数字

### 什么是 Token

Token 是模型处理文本的最小单位。它不一定是"一个字"，也可能是一个词的一部分。

```
"Hello world" → [15496, 995]          # 英文：基本一个词一个 token
"你好世界"     → [108386, 103924]       # 中文：通常一个汉字一个 token
"nanovllm"    → [259, 6834, 76, 2293]  # 罕见词被切成多个子词
```

### BPE（Byte Pair Encoding）算法

大多数大模型使用 BPE 或其变体来切词。核心思路：

```
1. 从字符级别开始： "hello" → ['h', 'e', 'l', 'l', 'o']
2. 统计相邻字符对频率： ('l', 'l') 出现 1 次
3. 合并最高频的字符对： 'll' → 新 token
4. 重复直到达到目标词表大小
```

词表大小是固定的（Qwen3 的词表有 151936 个 token），每个 token 对应一个唯一整数 ID。

### 示例：使用 Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 编码：文字 → token IDs
text = "Hello, nano-vllm！你好，大模型！"
token_ids = tokenizer.encode(text)
print(f"原文: {text}")
print(f"Token IDs: {token_ids}")
print(f"Token 数量: {len(token_ids)}")

# 逐个查看 token
tokens = tokenizer.convert_ids_to_tokens(token_ids)
for tid, token in zip(token_ids, tokens):
    print(f"  ID={tid:>7d}  Token={token}")

# 解码：token IDs → 文字
decoded = tokenizer.decode(token_ids)
print(f"解码回原文: {decoded}")

# 特殊 token
print(f"\nBOS token: {tokenizer.bos_token} (ID={tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID={tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID={tokenizer.pad_token_id})")
```

---

## 2.2 Embedding 层 —— 数字变向量

Token ID 只是一个整数（如 15496），模型无法直接对整数做矩阵运算。Embedding 层把每个整数映射成一个高维向量（如 4096 维）。

### 为什么不能简单地给每个词编个号

比如猫=1，狗=2，鱼=3？

因为编号暗示了"猫和狗的差距 = 1，猫和鱼的差距 = 2"，这种人为的距离关系是错的。用高维向量，模型自己学会哪些词真正相似（"猫"和"狗"的向量会比"猫"和"汽车"更相似），而不是被编号强加一个错误的关系。

**一句话：模型是数学运算器，只认识浮点数，不认识汉字。没有 Embedding，文字和数学之间的桥梁断了，模型根本无法开始工作。**

### 本质：查表操作

Embedding 层就是一个巨大的查找表（look-up table）：

```
表的大小：词表大小 × 隐藏维度 = 151936 × 4096

输入 token_id = 15496
输出 embedding_table[15496]  → 一个 4096 维的向量
```

这个查找表本身也是模型的参数（通过训练学到的）。

### nano-vllm 中的实现

```python
# nanovllm/layers/embed_head.py:9
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        # 权重矩阵：shape = (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings_per_partition, embedding_dim))

    def forward(self, x):
        # x 是 token ID 列表
        y = F.embedding(x, self.weight)  # 查表操作
        return y
```

使用位置在模型的最开始：

```python
# nanovllm/models/qwen3.py:178
class Qwen3Model(nn.Module):
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)  # ← Embedding 查表
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

### 示例：手动实现 Embedding

```python
import torch
import torch.nn.functional as F

vocab_size = 151936
hidden_dim = 4096

# Embedding 表（随机初始化，实际由训练得到）
embedding_table = torch.randn(vocab_size, hidden_dim)

# 输入 token IDs
token_ids = torch.tensor([15496, 995])  # "Hello world"

# 查表
embeddings = F.embedding(token_ids, embedding_table)
print(f"Token IDs: {token_ids.shape}")         # torch.Size([2])
print(f"Embeddings: {embeddings.shape}")        # torch.Size([2, 4096])

# 等价于用 one-hot 做矩阵乘法
one_hot = F.one_hot(token_ids, vocab_size).float()  # (2, 151936)
embeddings_v2 = one_hot @ embedding_table            # (2, 4096)
print(f"两种方式结果一致: {torch.allclose(embeddings, embeddings_v2)}")
```

---

## 2.3 词汇表大小

### 为什么是 151936？

词汇表大小是训练时决定的，需要权衡：
- **太大**：Embedding 表占显存（151936 × 4096 × 2 bytes ≈ 1.2 GB），LM Head 同样大小
- **太小**：表达力不够，文本被切成太多 token，处理效率低

Qwen3 选择了 151936，覆盖了常见的多语言字符、代码符号等。

---

## 2.4 位置信息 —— RoPE 的引入

### 问题

Embedding 只包含"这个 token 是什么"的信息，不包含"这个 token 在第几个位置"。

"我爱你" 和 "你爱我" 的三个 token embedding 分别是一样的，模型怎么区分顺序？

### 方案：位置编码

传统方案是在 Embedding 后直接加上一个位置向量。但 RoPE（旋转位置编码）用了更优雅的方式——在注意力计算时通过"旋转"把位置信息编码进 Q 和 K。这会在 Phase 3 详细讲解。

### nano-vllm 中的调用顺序

```python
# nanovllm/models/qwen3.py:72
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split(...)
    q, k = self.rotary_emb(positions, q, k)  # ← 在这里加位置信息
    o = self.attn(q, k, v)
    return output
```

注意 `positions` 参数从最外层一路传到 RoPE 层，它在模型 forward 的第一步就准备好了：

```python
# nanovllm/engine/model_runner.py:129
def prepare_prefill(self, seqs):
    for seq in seqs:
        positions.extend(range(start, end))  # ← 生成位置序列 [0, 1, 2, ...]
```

---

## 2.5 LM Head —— 向量变回文字

### 是什么

LM Head 是模型最后一层，把隐藏状态向量映射到词表大小的 logits，再变成概率——"下一个词有 60% 可能是'猫'，30% 是'鱼'……"。

### 为什么必须存在

模型最后一层输出的是一个向量（比如 4096 维的浮点数数组）。这堆数字对人毫无意义。模型的内部表示（4096 维向量）和最终输出（词表里选一个词）是两个不同的空间。LM Head 就是这个"翻译器"。

没有它，模型肚子里想了很多，但说不出来——就像一个人脑子在转但嘴巴张不开。

---

## 小结

| 概念 | 一句话 | 代码位置 |
|------|--------|---------|
| Tokenizer | BPE 算法把文字切成 token ID | 外部库 transformers |
| Embedding | 查表操作，token ID → 高维向量 | `layers/embed_head.py:9` |
| 词汇表 | 固定大小的查找表，trade-off 表达力和效率 | config.vocab_size |
| 位置信息 | Embedding 不含位置，由后续的 RoPE 编码 | `layers/rotary_embedding.py` |

数据流至此：**文字 → token IDs → Embedding 向量**（形状 `(seq_len, hidden_dim)`）

下一步：[03-attention.md](03-attention.md) —— 每个字怎么"看到"所有其他字？
