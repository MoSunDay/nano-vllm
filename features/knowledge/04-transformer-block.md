Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 4: Transformer Block 的其他组件

> 注意力是"看哪里"，下面这些是"怎么处理看到的信息"。

一个完整的 Transformer Block（Decoder Layer）长这样：

```
输入 hidden_states
  │
  ├── + 残差连接 ─────────────────────────────────────────────┐
  │                                                           │
  ├── RMSNorm（归一化）                                       │
  ├── Attention（注意力）                                     │
  │                                                           │
  ├── + 残差连接 ───────────────────────────────────────┐     │
  │                                                     │     │
  ├── RMSNorm（归一化）                                 │     │
  ├── MLP（gate_up → SiLU → down）                     │     │
  │                                                     │     │
  └─────────────────────────────────────────────────────┘─────┘
  │
  输出 hidden_states（形状不变）
```

---

## 4.1 RMSNorm —— 归一化

### 为什么需要归一化

经过多层变换后，数值可能越来越大或越来越小（梯度爆炸/消失）。归一化把数值"拉回"到一个合理的范围。

### 如果没有它会怎样

数据经过一层又一层的矩阵乘法，数值会发生什么？

- 乘以大于 1 的数 → 越来越大 → 几层之后变成 `inf`（爆炸）
- 乘以小于 1 的数 → 越来越小 → 几层之后变成 `0`（消失）

不管哪种，后面的层收到的都是垃圾数据，什么都学不了。这就是经典的**梯度消失/爆炸**问题。

有了 RMSNorm：每过一层，就把数据"拉回来"——太大了就缩小，太小了就放大，始终维持在一个合理的范围内。就像开车时定速巡航，不管上坡下坡，速度始终稳定。

**一句话：没有归一化，深层网络根本训练不动，数字几层之后就废了。**

### LayerNorm vs RMSNorm

| | LayerNorm | RMSNorm |
|---|---------|---------|
| 公式 | `(x - μ) / √(σ² + ε) × γ + β` | `x / √(mean(x²) + ε) × γ` |
| 减均值 | 是 | 否 |
| 偏置 β | 有 | 无 |
| 计算量 | 稍大 | 稍小 |

RMSNorm 发现减去均值对效果影响不大，省掉后更快。Qwen3 使用 RMSNorm。

### nano-vllm 中的实现

```python
# nanovllm/layers/layernorm.py:5
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数 γ

    @torch.compile
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()                                    # 转 FP32 提高精度
        var = x.pow(2).mean(dim=-1, keepdim=True)        # 均方值
        x.mul_(torch.rsqrt(var + self.eps))               # 除以 RMS
        x = x.to(orig_dtype).mul_(self.weight)            # 乘以 γ
        return x
```

### nano-vllm 中的融合版本

推理时经常把"残差加法 + RMSNorm"合成一步，避免多读一次显存：

```python
# nanovllm/layers/layernorm.py:28
@torch.compile
def add_rms_forward(self, x, residual):
    x = x.float().add_(residual.float())   # 残差加法
    residual = x.to(orig_dtype)             # 保存新的 residual
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

### 示例

```python
import torch

x = torch.randn(2, 4096) * 100  # 数值范围很大
print(f"归一化前: 均值={x.mean():.2f}, 标准差={x.std():.2f}")

# RMSNorm
rms = x.float().pow(2).mean(dim=-1, keepdim=True)
x_normed = x / torch.sqrt(rms + 1e-6)
print(f"归一化后: RMS={x_normed.float().pow(2).mean(dim=-1).sqrt().mean():.4f}")
# RMS 应该接近 1.0
```

---

## 4.2 残差连接（Residual Connection）

### 是什么

把层的输入直接加到输出上：`output = layer(input) + input`

### 为什么有效

想象一条 28 层的信息流水线。如果没有残差连接，第 28 层的信息必须经过 28 次变换才能追溯到原始输入——信息在传递中不断损失。

有了残差连接，相当于建了一条"信息高速公路"：即使中间某些层学到的变换很小（接近零），原始信息也能无损传递。

```
input ──────→ [Layer] ──→ + ──→ output
  │                        ↑
  └────────────────────────┘  （直接传递，无变换）
```

### nano-vllm 中的残差

```python
# nanovllm/models/qwen3.py:146
class Qwen3DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        # 第一次残差：attention 输入
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
            #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
            #                          归一化后送入 attention              保存原始值作为残差
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            #                          融合: x = norm(hidden + residual)
        hidden_states = self.self_attn(positions, hidden_states)

        # 第二次残差：MLP 输入
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
        # 返回的 residual 会在下一层继续被加上
```

---

## 4.3 MLP / FFN —— 两层线性变换

### 结构

MLP（多层感知器）是 Transformer Block 中的"思考"部分：

```
input (4096)
  → gate_proj: (4096 → 11008)  ─┐
  → up_proj:   (4096 → 11008)  ─┤→ SiLU(gate) * up  →  (11008)
                                 │
  → down_proj: (11008 → 4096)  ←┘
```

为什么中间维度（11008）比隐藏维度（4096）大？因为模型需要在一个更高维的空间里做非线性变换，才能学到复杂的模式。

### 门控机制（Gated MLP）

gate 投影和 up 投影的输出通过 SiLU 门控：

```
output = SiLU(gate_proj(x)) * up_proj(x)
```

- `sigmoid(gate_proj(x))` 的值在 0~1 之间，充当"门控信号"
- `SiLU(x) = x * sigmoid(x)`：当 x>0 时近似 x（通过），当 x<0 时近似 0（阻止），但不像 ReLU 直接截断
- `up_proj(x)` 产出实际的"内容"
- 逐元素相乘 = 选择性地通过信息

### nano-vllm 中的实现

```python
# nanovllm/layers/activation.py:6
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)  # 把 gate_up 的输出从中间切开
        return F.silu(x) * y   # SiLU(gate) * up

# nanovllm/models/qwen3.py:91
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate 和 up 各一个 intermediate_size
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # (batch, 2 * intermediate_size)
        x = self.act_fn(gate_up)        # SiLU(gate) * up → (batch, intermediate_size)
        x = self.down_proj(x)           # (batch, hidden_size)
        return x
```

### 示例

```python
import torch
import torch.nn.functional as F

hidden_dim = 8
intermediate_dim = 16

x = torch.randn(1, hidden_dim)

# gate 和 up 是两个独立的线性变换
gate_weight = torch.randn(intermediate_dim, hidden_dim)
up_weight = torch.randn(intermediate_dim, hidden_dim)
down_weight = torch.randn(hidden_dim, intermediate_dim)

gate = F.linear(x, gate_weight)  # (1, 16)
up = F.linear(x, up_weight)      # (1, 16)

# 门控
activated = F.silu(gate) * up    # SiLU(gate) * up

# 投影回原维度
output = F.linear(activated, down_weight)  # (1, 8)
print(f"输入: {x.shape}, 输出: {output.shape}")

# 合并成一次矩阵乘法（nano-vllm 的做法）
merged_weight = torch.cat([gate_weight, up_weight], dim=0)  # (32, 8)
gate_up = F.linear(x, merged_weight)  # (1, 32)
gate, up = gate_up.chunk(2, dim=-1)    # 各 (1, 16)
activated = F.silu(gate) * up
output = F.linear(activated, down_weight)
print(f"合并版本输出: {output.shape}")
```

---

## 4.4 激活函数（SiLU）

### 为什么需要激活函数

线性变换 `y = Wx + b` 无论叠加多少层，等价于一个线性变换。加一个非线性激活函数，模型才能学到复杂的非线性模式。

### 如果没有它会怎样

矩阵乘法的组合还是矩阵乘法。100 层线性变换叠在一起，数学上等价于 1 层线性变换。模型再深也没有用——它只能学到输入输出之间的直线关系，永远无法学会"如果 A 且 B 则 C"这种非线性逻辑。

SiLU 在矩阵乘法之间插入非线性，让每一层不再白叠。没有非线性激活，深度网络就是浅层网络，深层毫无意义。

### SiLU（Sigmoid Linear Unit）

```
SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))
```

特点：
- 当 x 很大时，sigmoid(x) ≈ 1，SiLU(x) ≈ x（通过）
- 当 x 很小时，sigmoid(x) ≈ 0，SiLU(x) ≈ 0（阻止）
- 当 x < 0 时，SiLU(x) < 0（允许负值，不像 ReLU 直接截断）

### 示例

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 100)

relu = F.relu(x)
silu = F.silu(x)
sigmoid = torch.sigmoid(x)

print(f"x = -5: SiLU={F.silu(torch.tensor(-5.0)):.4f}, ReLU={F.relu(torch.tensor(-5.0)):.4f}")
print(f"x =  0: SiLU={F.silu(torch.tensor(0.0)):.4f}, ReLU={F.relu(torch.tensor(0.0)):.4f}")
print(f"x =  5: SiLU={F.silu(torch.tensor(5.0)):.4f}, ReLU={F.relu(torch.tensor(5.0)):.4f}")
# SiLU 在负值区域有轻微的负输出，梯度更平滑
```

---

## 4.5 RoPE（旋转位置编码）

### 问题

Embedding 不含位置信息。"我爱你"和"你爱我"的 token embedding 完全相同，模型无法区分。

### 如果没有它会怎样

矩阵乘法不关心顺序——"猫""吃""鱼"三个向量，打乱顺序算出来的结果是一样的。没有 RoPE，模型把"我爱你"和"你爱我"看成完全一样的东西，分不清主语和宾语。

RoPE 用旋转给每个位置加不同的标记，而且旋转角度差天然编码了相对距离——位置 1 和位置 3 的关系，与位置 5 和位置 7 的关系是一样远的（相对距离 = 2）。这让模型能理解"谁在谁前面"以及"隔了多远"。

### 思路

不是在 Embedding 后加位置向量，而是在注意力计算时，通过**旋转** Q 和 K 来注入位置信息。

### 数学直觉

把 Q/K 的每两个维度看作一个二维平面上的点。RoPE 根据位置 index 对这个点做旋转：

```
位置 0: 不旋转
位置 1: 旋转 θ
位置 2: 旋转 2θ
位置 3: 旋转 3θ
...
```

两个位置 i 和 j 的 Q·K 点积会自然地包含 (i-j) 的相对位置信息，因为旋转角度差 = (i-j)θ。

### nano-vllm 中的实现

```python
# nanovllm/layers/rotary_embedding.py:6
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)  # 拆成两半
    y1 = x1 * cos - x2 * sin                     # 旋转公式的前半
    y2 = x2 * cos + x1 * sin                     # 旋转公式的后半
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

# nanovllm/layers/rotary_embedding.py:17
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        # 预计算所有位置的 cos/sin
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position_embeddings)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # (max_pos, 1, dim)
        self.register_buffer("cos_sin_cache", cache)

    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # 按位置索引
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
```

### 为什么比绝对位置编码好

绝对位置编码直接加到 Embedding 上，位置信息在多层传播中会衰减。

RoPE 在每层注意力的 Q/K 上直接注入位置，效果更持久。而且天然编码**相对位置**（两个位置的旋转角度差），这对语言理解更自然。

### 示例

```python
import torch

# 2D 旋转的直观理解
def rotate_2d(x, y, angle):
    cos_a = torch.cos(torch.tensor(angle))
    sin_a = torch.sin(torch.tensor(angle))
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return new_x, new_y

# 原始点
x, y = 1.0, 0.0

# 旋转不同角度（相当于不同位置）
for pos in range(5):
    angle = pos * 0.5
    rx, ry = rotate_2d(x, y, angle)
    print(f"位置 {pos}, 旋转 {angle:.1f} rad → ({rx:.3f}, {ry:.3f})")

# 两个位置的点积 = cos(角度差)，即编码了相对位置
# pos=0 和 pos=2 的点积 = cos(0) * cos(1.0) + sin(0) * sin(1.0) = cos(1.0)
# pos=1 和 pos=3 的点积 = cos(0.5) * cos(1.5) + sin(0.5) * sin(1.5) = cos(1.0)
# 相对位置差都是 2，点积相同！
```

---

## 小结：一个完整 Transformer Block 的数据流

```python
# nanovllm/models/qwen3.py:146 — Qwen3DecoderLayer.forward()

# 输入: hidden_states (seq_len, 4096), residual (第一次为 None)

# ① 第一个残差 + RMSNorm
hidden_states, residual = self.input_layernorm(hidden_states, residual)
#   RMSNorm(hidden_states + residual) → hidden_states
#   保存 (hidden_states + residual) → residual

# ② 注意力
hidden_states = self.self_attn(positions, hidden_states)
#   QKV 投影 → RoPE → FlashAttention → 输出投影

# ③ 第二个残差 + RMSNorm
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
#   RMSNorm(hidden_states + residual) → hidden_states
#   保存 (hidden_states + residual) → residual

# ④ MLP
hidden_states = self.mlp(hidden_states)
#   gate_up 投影 → SiLU(gate) * up → down 投影

# 输出: hidden_states (seq_len, 4096), residual (seq_len, 4096)
# → 传给下一个 block
```

下一步：[05-moe.md](05-moe.md) —— 什么是 MoE（混合专家）？
