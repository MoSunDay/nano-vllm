Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 6: 采样（Sampling）

> 模型输出的不是文字，而是词表上每个字的概率分布。采样就是从概率分布中"抽"出一个字。

---

## 6.1 Logits → 概率

### 为什么采样器必须存在

模型输出的是一个**概率分布**——"下一个词有 60% 可能是'鱼'，30% 是'肉'……"。它给你一张菜单，但没有帮你点菜。

采样器根据概率和 temperature（温度）来"点菜"：
- **temperature = 0**：永远点概率最高的那道菜（确定性输出）
- **temperature = 1**：完全按概率比例随机点
- **temperature 很高**：开始乱点，连概率很低的菜都可能选到（有创造性但可能胡说）

**一句话：没有采样器，模型只能输出"菜单"（概率），不能"上菜"（具体文字）。**

### LM Head 的输出

模型的最后一层（LM Head）输出的是 **logits**——一个形状为 `(batch_size, vocab_size)` 的张量，每个值代表词表中对应 token 的"原始分数"。

```python
# nanovllm/layers/embed_head.py:56
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        logits = F.linear(x, self.weight)  # (batch, vocab_size)
        return logits

# nanovllm/models/qwen3.py:212
def compute_logits(self, hidden_states):
    return self.lm_head(hidden_states)  # 隐藏状态 → logits
```

### Softmax 变概率

logits 不是概率（可以是负数，可以很大）。通过 softmax 变成概率：

```
probs[i] = exp(logits[i]) / Σ exp(logits[j])
```

性质：所有概率 > 0 且总和 = 1。

---

## 6.2 Temperature（温度）

### 是什么

Temperature 控制 softmax 的"尖锐程度"：

```
probs = softmax(logits / T)
```

| 温度 | 效果 | 直觉 |
|------|------|------|
| T → 0 | 概率集中在最大值上 | "选最保险的"，接近 greedy |
| T = 1 | 标准 softmax | "按原样" |
| T → ∞ | 概率趋近均匀分布 | "大胆尝鲜" |

### 示例

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, -1.0])

for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = F.softmax(logits / T, dim=-1)
    print(f"T={T:5.1f}: probs={[f'{p:.4f}' for p in probs.tolist()]}")

# T=  0.1: probs=['0.9987', '0.0009', '0.0003', '0.0000']  ← 几乎只选第一个
# T=  0.5: probs=['0.8562', '0.1096', '0.0297', '0.0045']
# T=  1.0: probs=['0.6439', '0.2369', '0.0949', '0.0244']  ← 标准
# T=  2.0: probs=['0.4018', '0.2891', '0.1987', '0.1104']
# T= 10.0: probs=['0.2619', '0.2489', '0.2347', '0.2545']  ← 接近均匀
```

### nano-vllm 中的使用

```python
# nanovllm/layers/sampler.py:8
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # logits / T
        probs = torch.softmax(logits, dim=-1)
        ...
```

---

## 6.3 nano-vllm 的采样算法：Gumbel Sampling

### 标准做法 vs Gumbel

标准做法：`probs → multinomial(probs)` 抽样。

nano-vllm 用了更高效的方法——**Gumbel-max trick**：

```python
# nanovllm/layers/sampler.py:11
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

逐步拆解：

```python
# 1. 生成指数分布的随机数
gumbel_noise = -log(Uniform(0, 1))  # 等价于 torch.empty_like(probs).exponential_(1)

# 2. 概率除以噪声
adjusted = probs / gumbel_noise

# 3. 取 argmax
token = argmax(adjusted)
```

### 为什么这样等价

数学上可以证明：`argmax(log(probs) + gumbel_noise)` 和 `multinomial(probs)` 的分布完全一致。nano-vllm 的写法是它的一个数值等价变体。

好处：只需要一次 argmax，不需要真正的采样操作，GPU 上更快。

---

## 6.4 Top-K / Top-P

### 问题

纯随机采样可能选到概率很低的 token，导致输出不连贯。

### Top-K

只保留概率最高的 K 个 token，其余设为 0，再归一化采样：

```
K = 5: 只考虑概率最高的 5 个 token
```

### Top-P（Nucleus Sampling）

按概率从大到小排序，累积到 P 就停止，只保留这些 token：

```
P = 0.9: 选最少的 token，使它们的累积概率 ≥ 0.9
```

### nano-vllm 的选择

当前 nano-vllm 没有实现 Top-K/Top-P，只用了 Temperature 采样。这是为了保持代码简洁。

### 示例：实现 Top-K 和 Top-P

```python
import torch
import torch.nn.functional as F

def top_k_sampling(logits, k=5, temperature=1.0):
    logits = logits / temperature
    # 只保留 top-k
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_value = top_k_values[:, -1:].expand(-1, logits.size(-1))
    logits = torch.where(logits < min_value,
                         torch.full_like(logits, float('-inf')),
                         logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 移除累积概率超过 p 的 token
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_probs[sorted_mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # 采样
    sampled_indices = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(1, sampled_indices)

# 测试
logits = torch.tensor([[2.0, 1.0, 0.1, 3.0, 0.5]])
print(f"Top-K (k=2): {top_k_sampling(logits, k=2)}")
print(f"Top-P (p=0.9): {top_p_sampling(logits, p=0.9)}")
```

---

## 6.5 重复惩罚（Repetition Penalty）

### 问题

模型有时会陷入"复读机"——反复生成相同的 token 或短语。因为已经生成的 token 会持续影响后续的概率分布。

### 解法

对已经生成过的 token 的 logits 做惩罚：

```python
if token_id in generated_tokens:
    if logits[token_id] > 0:
        logits[token_id] /= penalty  # 正的变小
    else:
        logits[token_id] *= penalty  # 负的更负
```

当前 nano-vllm 没有实现重复惩罚。

---

## 6.6 nano-vllm 完整的采样流程

```python
# nanovllm/engine/model_runner.py:214
def run(self, seqs, is_prefill):
    # 1. 准备输入
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs)

    # 2. 前向传播 → logits
    logits = self.run_model(input_ids, positions, is_prefill)

    # 3. 采样 → token_id（只有 rank 0 做）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    return token_ids
```

```python
# nanovllm/layers/sampler.py — Sampler
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # 除以 temperature
    probs = torch.softmax(logits, dim=-1)                         # softmax
    sample_tokens = probs.div_(                                   # Gumbel-max trick
        torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return sample_tokens
```

### 示例：完整的采样模拟

```python
import torch
import torch.nn.functional as F

vocab_size = 10

# 模拟 logits（模型输出）
logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 4.0, 0.2, 0.3, 1.5, 0.8]])
print(f"Logits: {logits}")

# Temperature = 0.5（保守）
probs_low_t = F.softmax(logits / 0.5, dim=-1)
print(f"\nT=0.5 概率: {probs_low_t}")
print(f"  最高概率 token: {probs_low_t.argmax().item()} (prob={probs_low_t.max().item():.4f})")

# Temperature = 1.0（正常）
probs_mid_t = F.softmax(logits / 1.0, dim=-1)
print(f"\nT=1.0 概率: {probs_mid_t}")
print(f"  最高概率 token: {probs_mid_t.argmax().item()} (prob={probs_mid_t.max().item():.4f})")

# Temperature = 2.0（冒险）
probs_high_t = F.softmax(logits / 2.0, dim=-1)
print(f"\nT=2.0 概率: {probs_high_t}")
print(f"  最高概率 token: {probs_high_t.argmax().item()} (prob={probs_high_t.max().item():.4f})")

# 可以看到：温度越低，概率越集中在最大值（token 5）上
# 温度越高，概率分布越均匀
```

---

## 小结

| 概念 | 一句话 | 代码位置 |
|------|--------|---------|
| Logits | 模型输出的原始分数，形状 (batch, vocab) | `models/qwen3.py:212` |
| Softmax | 把分数变成概率分布 | `layers/sampler.py:10` |
| Temperature | 控制概率分布的"尖锐度" | `layers/sampler.py:9` |
| Gumbel-max trick | 用 argmax 模拟采样，GPU 上更快 | `layers/sampler.py:11` |
| Top-K/P | 限制采样范围（nano-vllm 未实现） | - |

下一步：[07-inference-engine.md](07-inference-engine.md) —— 怎么让推理又快又省？
