Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# layers 模块

可复用的神经网络层组件，服务于模型实现。

## 职责

提供 attention、linear、activation、normalization、positional embedding、sampling、embedding/head 等基础层。

## 关键组件

### Attention (`attention.py`)
- `store_kvcache`：Triton kernel，按 slot_mapping 将 KV 写入 paged KV cache
- `Attention` 模块：
  - prefill + prefix cache：使用 `flash_attn_varlen_func`，支持 block_table
  - decode：使用 `flash_attn_with_kvcache`
  - KV cache 存储由 ModelRunner 在初始化时注入（`k_cache` / `v_cache`）

### Linear (`linear.py`)
- `LinearBase`：基类，定义 `weight_loader` 协议，自动感知 tp_rank / tp_size
- `ReplicatedLinear`：无分片
- `ColumnParallelLinear`：按列切分输出维度
- `MergedColumnParallelLinear`：gate + up 合并切分（用于 MLP）
- `QKVParallelLinear`：Q/K/V 合并切分，按 shard_id 区分加载
- `RowParallelLinear`：按行切分输入维度，forward 时 all-reduce

### Embedding & LM Head (`embed_head.py`)
- `VocabParallelEmbedding`：按词表切分，forward 时 mask + all-reduce
- `ParallelLMHead`：继承 embedding，仅在 prefill 时取每个序列最后一个 token 计算 logits；多卡时 gather 到 rank 0

### RMSNorm (`layernorm.py`)
- `rms_forward`：标准 RMSNorm（`@torch.compile`）
- `add_rms_forward`：fused residual + RMSNorm（`@torch.compile`）

### RotaryEmbedding (`rotary_embedding.py`)
- 预计算 cos/sin cache，`@torch.compile` 加速 apply
- `get_rope()` 通过 `lru_cache` 单例化

### Activation (`activation.py`)
- `SiluAndMul`：SiLU 门控（`@torch.compile`），用于 MLP gate_up_proj 后

### Sampler (`sampler.py`)
- `@torch.compile` 加速的 temperature scaling + softmax + Gumbel-like 采样
- 公式：`probs / exponential(1).clamp_min(1e-10)` → argmax

## 依赖

- [utils/context](../utils/index.md)：`get_context()` 获取调度上下文
- flash-attn, triton, torch.distributed

## 设计约定

- 所有权重层实现 `weight_loader` 方法，供 `utils/loader` 统一调用
- TP 相关层在 `__init__` 时自动切分，forward 时透明处理通信
- `@torch.compile` 用于热点小 kernel（RMSNorm、RoPE、SiLuAndMul、Sampler）
