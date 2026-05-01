Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 1: 参数（Parameters）

> 参数 = 模型通过训练学到的知识，存储在矩阵里。

---

## 1.1 什么是参数

大模型的"参数"就是一堆数字矩阵。这些矩阵在做一件事：**线性变换**。

```python
# 线性变换的本质
y = x @ W + b
#   输入 × 权重矩阵 + 偏置

# 在 PyTorch 中
import torch.nn as nn
linear = nn.Linear(4096, 4096)  # 权重: (4096, 4096) = 16,777,216 个参数
print(f"参数数量: {linear.weight.numel()}")  # 16777216
print(f"参数形状: {linear.weight.shape}")     # torch.Size([4096, 4096])
```

这些参数分布在哪里？以 Qwen3 为例（具体数值因模型规模而异）：

> 以下数字以 hidden_size=4096, intermediate_size=11008, num_heads=32, num_kv_heads=8(GQA), num_layers=36 为例。

| 层 | 参数矩阵 | 形状（单层） | 参数量 |
|----|---------|-------------|--------|
| Embedding | `embed_tokens.weight` | (151936, 4096) | ~622M |
| Attention ×36层 | `qkv_proj.weight` | (6144, 4096) × 36 | ~907M |
| Attention ×36层 | `o_proj.weight` | (4096, 4096) × 36 | ~604M |
| MLP ×36层 | `gate_up_proj.weight` | (22016, 4096) × 36 | ~3.2B |
| MLP ×36层 | `down_proj.weight` | (4096, 11008) × 36 | ~1.6B |
| RMSNorm ×73个 | `weight` | (4096,) × 73 | ~0.3M |
| LM Head | `lm_head.weight` | (151936, 4096) | ~622M 或 0（tied） |

其中 `qkv_proj` 的输出维度 = (num_heads + 2×num_kv_heads) × head_dim = (32+16)×128 = 6144。

**总计约 70~80 亿**（取决于是否 tie LM Head）。MLP 层占大部分（gate+up+down 约 4.8B）。

### 示例：查看模型各层参数

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
print(f"词汇表大小: {config.vocab_size}")           # 151936
print(f"隐藏维度: {config.hidden_size}")            # 4096
print(f"中间维度: {config.intermediate_size}")       # 11008
print(f"注意力头数: {config.num_attention_heads}")   # 32
print(f"KV 头数: {config.num_key_value_heads}")     # 8 (GQA)
print(f"层数: {config.num_hidden_layers}")          # 36

head_dim = config.hidden_size // config.num_attention_heads  # 128

# 手动计算参数量
h = config.hidden_size
i = config.intermediate_size
v = config.vocab_size
n = config.num_hidden_layers
kv = config.num_key_value_heads

embed = v * h
qkv = n * (h + 2 * kv * head_dim) * h
o_proj = n * h * h
mlp = n * (h * i * 2 + i * h)  # gate_up and down
lm_head = 0  # 如果 tie_word_embeddings

total = embed + qkv + o_proj + mlp + lm_head
print(f"估算总参数量: {total / 1e9:.2f}B")
```

---

## 1.2 参数怎么来的

参数通过**训练**得到。训练的本质是一个优化过程：

```
1. 初始化：所有参数随机赋值
2. 前向传播：输入文本 → 模型输出预测概率
3. 计算 loss：预测概率和真实文本的差异
4. 反向传播：计算 loss 对每个参数的梯度
5. 更新参数：参数 -= 学习率 × 梯度
6. 重复 2-5 数万亿次
```

训练完成后，参数被**冻结**（不再更新），用于推理。推理只做前向传播。

### 示例：模拟参数更新

```python
import torch

W = torch.randn(3, 3, requires_grad=True)  # 随机初始化
x = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 1.0, 0.0])

# 前向传播
y = x @ W
loss = ((y - target) ** 2).mean()

# 反向传播
loss.backward()
print(f"梯度:\n{W.grad}")

# 参数更新
learning_rate = 0.01
with torch.no_grad():
    W -= learning_rate * W.grad
print(f"更新后的权重:\n{W}")
```

---

## 1.3 参数怎么存的

训练好的参数存储在 `safetensors` 文件中。每个文件是一个键值对映射：`层名 → 权重张量`。

在 nano-vllm 中，参数加载逻辑在 `nanovllm/utils/loader.py`：

```python
# nanovllm/utils/loader.py:12
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 处理 packed modules（qkv 合并、gate_up 合并）
                ...
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))
```

### Packed Modules（合并打包）

原始模型文件中，q_proj、k_proj、v_proj 是分开存储的。但推理时合并成一个矩阵更高效：

```python
# nanovllm/models/qwen3.py:187
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),     # q_proj 合并到 qkv_proj 的 q 部分
        "k_proj": ("qkv_proj", "k"),     # k_proj 合并到 qkv_proj 的 k 部分
        "v_proj": ("qkv_proj", "v"),     # v_proj 合并到 qkv_proj 的 v 部分
        "gate_proj": ("gate_up_proj", 0),  # gate_proj 合并到 gate_up_proj 的前半
        "up_proj": ("gate_up_proj", 1),    # up_proj 合并到 gate_up_proj 的后半
    }
```

这样 `qkv_proj` 的权重就是 `[q_weights; k_weights; v_weights]`，一次矩阵乘法就能同时算出 Q、K、V。

### 示例：查看 safetensors 文件内容

```python
from safetensors import safe_open

# 假设模型路径
model_path = "/path/to/Qwen3-8B"

for file in sorted(glob(f"{model_path}/*.safetensors")):
    print(f"\n=== {file} ===")
    with safe_open(file, "pt", "cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
```

---

## 1.4 参数量和能力的关系

**Scaling Law**（缩放定律）：模型的损失（loss）大致和参数量的幂律成反比。

```
Loss ∝ 1 / N^α    （N = 参数量，α ≈ 0.076）
```

直觉：参数越多 → 模型能存储的"知识模式"越多 → 预测越准。但有边际递减效应——从 1B 到 7B 的提升远大于从 70B 到 77B。

---

## 1.5 显存占用计算

推理时，显存被三部分瓜分：

| 部分 | 计算方式 | 7B FP16 示例 |
|------|----------|-------------|
| 模型参数 | 参数量 × 字节数 | 7B × 2 = **14 GB** |
| KV Cache | 见下方公式 | **数 GB**（取决于序列长度和 batch） |
| 临时缓冲 | 激活值、中间结果 | **~1 GB** |

### KV Cache 显存公式

```
KV Cache = 2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × bytes_per_element
```

示例：Qwen3-8B，batch=32，seq_len=4096，FP16：
```
2 × 36 × 32 × 4096 × 8 × 128 × 2 bytes ≈ 18 GB
```

这就是为什么推理引擎需要精心管理 KV cache——它可能比模型本身还大！

### nano-vllm 中的 KV cache 分配

```python
# nanovllm/engine/model_runner.py:103
def allocate_kv_cache(self):
    # 计算每个 block 的字节数
    block_bytes = (2 * num_hidden_layers * block_size
                   * num_kv_heads * head_dim * dtype_itemsize)
    # 用剩余显存计算能分配多少 block
    num_blocks = (total * gpu_util - used - peak + current) // block_bytes
    # 分配 KV cache
    self.kv_cache = torch.empty(2, num_hidden_layers, num_blocks,
                                block_size, num_kv_heads, head_dim)
```

---

## 小结

| 概念 | 一句话 |
|------|--------|
| 参数 | 矩阵，通过训练学到的知识 |
| 训练 | 重复"前向→计算 loss→反向→更新"数万亿次 |
| 存储 | safetensors 文件，支持 packed modules 合并 |
| Scaling Law | 参数越多能力越强，但有边际递减 |
| 显存 | 模型参数 + KV cache + 临时缓冲，KV cache 可能最大 |

下一步：[02-embedding-tokenizer.md](02-embedding-tokenizer.md) —— 文字和数字怎么互转？
