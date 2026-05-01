Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 3: 注意力机制（Attention）

> 注意力让每个 token 能"看到"所有其他 token，并决定关注谁。

---

## 3.1 为什么需要注意力

### RNN 的致命缺陷

RNN（循环神经网络）逐个处理 token，每一步的隐藏状态只依赖前一步：

```
h1 → h2 → h3 → ... → h100
```

问题：h100 携带的信息经过 99 次变换，早期的信息（h1）几乎丢失了。这就是**长距离依赖问题**。

### 如果没有注意力会怎样

每个词经过矩阵变换后，各自独立地变成新的向量。词和词之间没有任何交互。"猫"不知道"鱼"的存在，"吃"不知道谁是主语谁是宾语。

注意力让每个词可以去"看"句子中所有的其他词，并决定**该关注谁**。比如"吃"这个词会强烈关注"猫"和"鱼"——因为它们是它的主语和宾语。这种"谁和谁有关"的关系，就是注意力机制学到的。

**一句话：没有注意力，每个 token 是孤岛；有了注意力，词与词之间才建立了联系。**

### Self-Attention 的解法

Self-Attention 让每个 token 直接和所有其他 token 交互，一步到位，没有"信息衰减"：

```
token_1 能同时"看到" token_1, token_2, ..., token_100
token_2 能同时"看到" token_1, token_2, ..., token_100
...
```

代价：计算量从 O(n) 变成 O(n²)，这也是为什么长上下文很贵。

---

## 3.2 Q / K / V 三个角色

注意力机制借用了信息检索的类比：

| 角色 | 类比 | 作用 |
|------|------|------|
| **Q（Query）** | 搜索关键词 | "我想找什么？" |
| **K（Key）** | 文档标题/标签 | "我有什么信息？" |
| **V（Value）** | 文档正文内容 | "我的实际内容是什么？" |

每个 token 同时拥有 Q、K、V 三个向量，通过三个不同的线性变换从隐藏状态得到。

### nano-vllm 中的 QKV 投影

```python
# nanovllm/models/qwen3.py:42
self.qkv_proj = QKVParallelLinear(
    hidden_size,       # 输入维度
    self.head_dim,     # 每个头的维度
    self.total_num_heads,      # Q 的头数
    self.total_num_kv_heads,   # K/V 的头数（GQA 时更少）
)

# nanovllm/models/qwen3.py:77
qkv = self.qkv_proj(hidden_states)  # 一次矩阵乘法算出 Q、K、V
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

为什么合并成一次矩阵乘法？因为 `qkv_proj` 的权重是 `[W_q; W_k; W_v]` 拼接的，一次 `x @ W_qkv` 就能得到三个结果。

---

## 3.3 注意力计算

核心公式：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

逐步拆解：

```python
# 伪代码
scores = Q @ K.T              # (seq_len, seq_len) 每对 token 的相似度
scores = scores / sqrt(d)     # 缩放，防止 softmax 溢出
mask = causal_mask(scores)    # 因果掩码：未来 token 不可见
weights = softmax(scores + mask)  # (seq_len, seq_len) 注意力权重
output = weights @ V          # (seq_len, d_v) 加权求和
```

### 为什么要除以 √d？

Q 和 K 的点积的方差和维度 d 成正比。d=128 时，点积可能到几百，softmax 会进入梯度极小的区域（saturation）。除以 √d 把方差归一化回 1。

```python
# nanovllm/models/qwen3.py:39
self.scaling = self.head_dim ** -0.5  # 1/√d
```

### 因果掩码（Causal Mask）

生成第 i 个 token 时，不能"偷看"第 i+1 之后的 token。因果掩码把上三角设为 -∞：

```
     t1  t2  t3  t4
t1 [  0  -∞  -∞  -∞ ]     t1 只能看 t1
t2 [  0   0  -∞  -∞ ]     t2 能看 t1, t2
t3 [  0   0   0  -∞ ]     t3 能看 t1, t2, t3
t4 [  0   0   0   0  ]     t4 能看所有
```

在 nano-vllm 中，因果掩码由 FlashAttention 内部处理（`causal=True` 参数）。

### 示例：手动计算注意力

```python
import torch
import torch.nn.functional as F

seq_len = 4
head_dim = 8

Q = torch.randn(seq_len, head_dim)
K = torch.randn(seq_len, head_dim)
V = torch.randn(seq_len, head_dim)

# 1. 计算注意力分数
scores = Q @ K.T  # (4, 4)
print(f"分数:\n{scores}")

# 2. 缩放
scores = scores / (head_dim ** 0.5)

# 3. 因果掩码
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
scores = scores + mask
print(f"掩码后:\n{scores}")

# 4. Softmax
weights = F.softmax(scores, dim=-1)
print(f"注意力权重:\n{weights}")
# 每行和为 1，上三角为 0

# 5. 加权求和
output = weights @ V  # (4, 8)
print(f"输出形状: {output.shape}")
```

---

## 3.4 多头注意力（MHA）

### 为什么分多个头

把一个大的注意力拆成多个小的"头"，每个头可以学到不同的关注模式：
- 头 1 可能关注语法关系（主语→谓语）
- 头 2 可能关注指代关系（代词→实体）
- 头 3 可能关注位置关系（相邻词）

### 怎么分

把 `hidden_dim=4096` 拆成 `num_heads=32` 个头，每个头 `head_dim=128`：

```
Q: (seq_len, 4096) → (seq_len, 32, 128)
K: (seq_len, 4096) → (seq_len, 32, 128)
V: (seq_len, 4096) → (seq_len, 32, 128)

每个头独立计算注意力，最后拼接回 (seq_len, 4096)
```

---

## 3.5 GQA（Grouped Query Attention）

### MHA 的问题

每个头都有独立的 K 和 V。32 个头意味着 32 套 K/V，KV cache 很大。

### GQA 的解法

让多组 Q 共享同一组 K/V：

```
MHA:  32 Q 头 → 32 K/V 头  (1:1)
GQA:  32 Q 头 → 8 K/V 头   (4:1)  ← Qwen3 用这个
MQA:  32 Q 头 → 1 K/V 头   (32:1)
```

Qwen3 有 32 个 Q 头但只有 8 个 KV 头，KV cache 直接缩小 4 倍！

### nano-vllm 中的体现

```python
# nanovllm/models/qwen3.py:33
self.total_num_kv_heads = num_kv_heads  # 8（比 Q 头少）

# nanovllm/layers/attention.py:43
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads      # 32 / tp_size
        self.num_kv_heads = num_kv_heads  # 8 / tp_size
```

FlashAttention 内部自动处理 Q 头多于 KV 头的情况（通过广播 K/V）。

---

## 3.6 KV Cache

### 问题

Decode 阶段，每步只生成 1 个新 token。但注意力需要所有历史 token 的 K 和 V。

如果每次都从头计算所有 K/V：生成长度 n 的序列，总计算量 O(n³)。太慢！

### 解法：缓存

每步只需要：
1. 计算新 token 的 K 和 V
2. 把它们追加到缓存中
3. 新 token 的 Q 和缓存中所有 K/V 做注意力

这样每步计算量 O(n)，总计算量 O(n²)。

### nano-vllm 中的 KV cache 存储

```python
# nanovllm/layers/attention.py:11  — Triton kernel 把 K/V 写入 cache
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 把当前 token 的 K/V 写入对应 slot
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

KV cache 的结构：`(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`

- `2`：K 和 V 各一份
- `num_blocks`：总共有多少个 block（取决于显存大小）
- `block_size=256`：每个 block 存 256 个 token 的 KV

---

## 3.7 注意力陷阱（Attention Sink & Lost-in-the-Middle）

### 注意力分配不均

Softmax 把注意力权重归一化为概率分布（总和=1）。这意味着：
- 如果某些 token "太显眼"（K 和 Q 高度匹配），会挤占其他 token 的注意力份额
- 模型倾向把大量注意力分配给**开头几个 token**和**最近的 token**
- 中间位置的信息容易被"忽略"

### Attention Sink（注意力汇）

研究发现，无论输入什么内容，第一个 token 总是获得极高的注意力权重。它充当了一个"垃圾收集器"，聚合了不需要的信息。

```
Token:     [BOS] The cat sat on the mat ...
Attention:  0.7   0.05  0.03  0.02  ...  0.1
            ↑ Attention Sink            ↑ 最近 token 也有较高权重
```

### Lost-in-the-Middle（中间丢失）

在长上下文中（如 100K tokens），模型往往：
- 能记住**开头**的信息
- 能记住**结尾**的信息
- 对**中间**的信息回忆能力显著下降

### 缓解方案

| 方案 | 原理 |
|------|------|
| StreamingLLM | 保留开头的 attention sink tokens，滑动窗口处理新 token |
| 注意力sink注入 | 在输入开头人为加入占位 token |
| 长上下文训练 | 训练时用长序列，让模型学会均匀分配注意力 |

### 示例：可视化注意力分布

```python
import torch
import torch.nn.functional as F

seq_len = 100
head_dim = 64

Q = torch.randn(seq_len, head_dim)
K = torch.randn(seq_len, head_dim)

scores = Q @ K.T / (head_dim ** 0.5)

# 模拟因果掩码
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
weights = F.softmax(scores + mask, dim=-1)

# 查看第 50 个 token 的注意力分布
w = weights[50, :51]  # 只能看到前 51 个 token
print(f"第 50 个 token 的注意力分布:")
print(f"  前 5 个 token 的权重和: {w[:5].sum().item():.4f}")
print(f"  中间 token (20-40) 的权重和: {w[20:40].sum().item():.4f}")
print(f"  最近 5 个 token 的权重和: {w[-5:].sum().item():.4f}")
print(f"  总和: {w.sum().item():.4f}")
```

---

## 小结

| 概念 | 一句话 | 代码位置 |
|------|--------|---------|
| Q/K/V | 查询、键、值，三种线性变换 | `layers/linear.py:96` QKVParallelLinear |
| 注意力计算 | softmax(QK^T/√d) × V | `layers/attention.py:59` |
| 多头注意力 | 拆成多个小注意力，学不同模式 | `models/qwen3.py:62` |
| GQA | 多组 Q 共享 K/V，省显存 | config.num_key_value_heads |
| KV Cache | 缓存历史 K/V 避免重算 | `layers/attention.py:11` store_kvcache_kernel |
| 注意力陷阱 | 首尾 token 吸走注意力，中间信息丢失 | 研究发现，非代码实现 |

下一步：[04-transformer-block.md](04-transformer-block.md) —— 注意力之外还有什么组件？
