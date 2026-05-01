Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 5: MoE（Mixture of Experts，混合专家）

> 不是每个 token 都需要用全部参数，让"专家"各司其职。

---

## 5.1 为什么需要 MoE

### Dense 模型的困境

传统的 Dense 模型（如 Qwen3-7B 的 Dense 版本），每次推理每个 token 都要经过**全部**参数的计算。

问题：
- 想要更强的能力 → 需要更多参数 → 但每次推理更慢、更费显存
- 参数量和推理成本的矛盾

### MoE 的解法

把一个大的 MLP 拆成 N 个小的 MLP（**专家**），每次只激活其中 K 个。这样：
- 总参数量可以很大（如 600 亿），提供强大的知识容量
- 每次推理只用到一小部分（如 80 亿），保持高速度

```
Dense MLP:  x → [一个巨大的 MLP] → y    全部参数都参与

MoE MLP:   x → [路由器] → 选择专家 2, 5, 7
               → expert_2(x) * w2 + expert_5(x) * w5 + expert_7(x) * w7 → y
               只有 3/8 的参数参与
```

### 类比

- **Dense 模型** = 一个全科医生看所有病，什么都会但每样都不精
- **MoE 模型** = 一个医院，有 64 个专科医生，每次根据症状挂号找 8 个看
- **路由器** = 导诊台

---

## 5.2 MoE 的基本结构

### 标准的 MoE 层替换

在 Transformer Block 中，MoE 替换的是 MLP 部分（注意力层不变）：

```
标准 Block:    Attention → MLP (一个大的 FFN)
MoE Block:     Attention → Router → N 个 Expert FFN → 加权组合
```

### 具体流程

```
输入 x (hidden_dim)
  │
  ├── Router: gate(x) → softmax → 得到 N 个专家的权重
  │
  ├── 选择 Top-K 个专家（如 K=8）
  │
  ├── Expert_3(x) * w3 + Expert_17(x) * w17 + ... + Expert_42(x) * w42
  │
  └── 输出 (hidden_dim)
```

---

## 5.3 路由器（Router / Gate）

### 结构

路由器是一个很小的线性层 + softmax：

```python
# 伪代码
gate_weights = nn.Linear(hidden_dim, num_experts)  # (4096, 64)
logits = gate_weights(x)                            # (64,)
weights = softmax(logits)                           # (64,) 每个专家的权重
top_k_weights, top_k_indices = weights.topk(k=8)    # 选前 8 个
```

### 为什么要 softmax

softmax 保证权重为正且总和为 1，可以解释为"概率"——路由器有多大的信心把这个 token 交给这个专家。

### 负载均衡问题

如果所有 token 都去找同一个专家（比如 Expert 0 最"受欢迎"），那么：
- Expert 0 成为瓶颈（串行处理所有 token）
- 其他专家闲置（浪费参数）

解决方法：**Auxiliary Loss（辅助损失）**。训练时额外加一个损失项，惩罚专家间负载不均。

---

## 5.4 细粒度专家（Fine-grained Experts）

### 标准做法

每个专家是一个完整的 FFN：`gate_proj + up_proj + down_proj`，中间维度和 Dense MLP 一样。

### 细粒度做法（Qwen3-MoE 采用）

把标准 FFN 拆成更小的专家：
- 标准专家：`hidden_dim → intermediate_dim → hidden_dim`（中间维度 11008）
- 细粒度专家：`hidden_dim → intermediate_dim/4 → hidden_dim`（中间维度 2752）

好处：
- 更多专家（64 → 64×4=256 个），更灵活的组合
- 每个专家更小，激活的参数总量可控

---

## 5.5 Shared Expert（共享专家）

### 问题

有些通用知识（如语法规则、常见搭配）是所有 token 都需要的，不应该被路由分散。

### 解法

除了 N 个被路由的专家外，额外保留一个（或几个）**始终激活**的共享专家：

```
输出 = Shared_Expert(x) + Σ(top_k_routed_experts(x) * weights)
        ↑ 始终参与        ↑ 路由器选择的 K 个
```

这样通用知识由共享专家负责，专业知识由路由专家负责。

---

## 5.6 Qwen3 的 MoE 变体示例

以 Qwen3-30B-A3B 为例（总参数 30B，每次激活 3B）：

```
64 个路由专家（细粒度），每次激活 8 个
4 个共享专家（始终激活）

每个 token 实际使用: 8 + 4 = 12 个专家
激活参数: ~3B（远小于总参数 30B）
```

### 示例：模拟 MoE 前向传播

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleExpert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_up = nn.Linear(hidden_dim, intermediate_dim * 2, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

class SimpleMoE(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SimpleExpert(hidden_dim, intermediate_dim // 4)
            for _ in range(num_experts)
        ])
        # 共享专家
        self.shared_expert = SimpleExpert(hidden_dim, intermediate_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # 路由
        logits = self.gate(x)                               # (batch, num_experts)
        weights = F.softmax(logits, dim=-1)                  # (batch, num_experts)
        top_k_weights, top_k_indices = weights.topk(self.top_k, dim=-1)  # (batch, top_k)

        # 归一化 top-k 权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 计算路由专家的输出
        output = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_weights[i, j].item()
                expert_output = self.experts[expert_idx](x[i:i+1])
                output[i] += expert_weight * expert_output.squeeze(0)

        # 加上共享专家
        output = output + self.shared_expert(x)

        return output

# 测试
hidden_dim = 256
intermediate_dim = 512
num_experts = 8
top_k = 2

moe = SimpleMoE(hidden_dim, intermediate_dim, num_experts, top_k)
x = torch.randn(4, hidden_dim)  # 4 个 token
y = moe(x)
print(f"输入: {x.shape}, 输出: {y.shape}")

# 查看每个 token 选择了哪些专家
logits = moe.gate(x)
weights = F.softmax(logits, dim=-1)
_, indices = weights.topk(top_k, dim=-1)
for i in range(4):
    print(f"Token {i} 选择专家: {indices[i].tolist()}")
```

---

## 5.7 Expert Parallelism（专家并行）

### 怎么把专家分布到多张 GPU

既然每次只用 K 个专家，可以让不同 GPU 负责不同的专家：

```
GPU 0: Expert 0, 1, 2, 3
GPU 1: Expert 4, 5, 6, 7
GPU 2: Expert 8, 9, 10, 11
GPU 3: Expert 12, 13, 14, 15
```

token 路由到某专家时，数据发到对应 GPU，计算完再发回来。这需要 All-to-All 通信。

---

## 小结

| 概念 | 一句话 |
|------|--------|
| MoE | 多个专家，每次只激活 K 个，大参数量但低推理成本 |
| 路由器 | 小线性层 + softmax，决定 token 给哪个专家 |
| 负载均衡 | 防止所有 token 挤向同一个专家 |
| 细粒度专家 | 把大 FFN 拆成多个小 FFN，更灵活 |
| 共享专家 | 始终激活的专家，负责通用知识 |
| Expert Parallelism | 不同专家放不同 GPU |

**注意**：当前 nano-vllm 仅支持 Qwen3 Dense 版本（非 MoE）。但理解 MoE 对理解模型架构演进很重要。

下一步：[06-sampling.md](06-sampling.md) —— 从概率到文字？
