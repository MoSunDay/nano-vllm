Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# models 模块

具体模型架构实现。

## 职责

将 layers 组件组装为完整的 transformer 模型，适配 HuggingFace checkpoint。

## 当前实现

### Qwen3 (`qwen3.py`)

- `Qwen3ForCausalLM`：顶层模型，包含 `Qwen3Model` + `ParallelLMHead`
  - `packed_modules_mapping` 定义权重合并映射（q/k/v → qkv_proj, gate/up → gate_up_proj）
  - `compute_logits()` 单独调用，支持 CUDA graph 路径
- `Qwen3Model`：embedding + N 层 decoder + final RMSNorm
- `Qwen3DecoderLayer`：pre-norm 架构，fused residual add
- `Qwen3Attention`：GQA 支持，可选 qkv_bias，无 bias 时使用 q_norm / k_norm（Qwen3 特有）
- `Qwen3MLP`：gate_up_proj (SiLU) + down_proj

## 权重加载协议

模型通过 `packed_modules_mapping` 声明哪些 HF 权重应合并到单个参数。`utils/loader` 据此查找对应参数的 `weight_loader` 完成分片加载。

## 依赖

- [layers](../layers/index.md)：所有层组件
- transformers：`Qwen3Config`

## 扩展方式

添加新模型时：
1. 创建新文件（如 `llama.py`）
2. 实现对应的 `*ForCausalLM` 类，包含 `packed_modules_mapping`
3. 在 `ModelRunner.__init__` 中根据 hf_config 选择模型类
