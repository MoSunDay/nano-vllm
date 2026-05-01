Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# nano-vllm 逻辑结构

轻量级 LLM 推理引擎，从零实现 vLLM 核心功能。纯 Python，约 1200 行，支持 prefix caching、tensor parallelism、CUDA graph、torch.compile 等优化。

## 模块索引

| 模块 | 职责 |
|------|------|
| [engine](agents/engine/index.md) | 调度、KV cache block 管理、序列生命周期、模型执行编排 |
| [layers](agents/layers/index.md) | 注意力、线性层、激活函数、RMSNorm、RoPE、采样器、embedding/LM head |
| [models](agents/models/index.md) | 具体模型实现（Qwen3） |
| [utils](agents/utils/index.md) | 全局上下文传递、权重加载 |

## 入口

- `nanovllm.LLM`（即 `LLMEngine`）是唯一的用户入口，提供 `generate()` 方法
- `nanovllm.SamplingParams` 控制 temperature、max_tokens、ignore_eos

## 外部依赖

torch, triton, transformers, flash-attn, xxhash, safetensors

## 功能索引

参见 [features/index.md](features/index.md)
