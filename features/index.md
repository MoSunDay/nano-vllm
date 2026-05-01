Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# nano-vllm 功能索引

## 核心能力

| 功能 | 说明 |
|------|------|
| 离线批量推理 | 接受多条 prompt，自动 prefill + decode 循环，返回生成文本 |
| Tensor Parallelism | 多 GPU 并行，通过 NCCL + SharedMemory 通信，支持 1-8 卡 |
| Prefix Caching | 基于 block 内容哈希的 KV cache 共享，跨序列复用相同前缀 |
| CUDA Graph | decode 阶段预捕获多 batch size 的 CUDA graph，减少 CPU 开销 |
| Chunked Prefill | 长 prompt 分块处理，避免单序列占用全部调度预算 |
| 抢占调度 | KV cache 内存不足时 preempt 运行中序列，释放 block 给高优先级请求 |

## 使用方式

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256))
```

## 限制

- 仅支持 Qwen3 模型架构
- 仅离线推理（无 serving）
- 不支持 greedy sampling（temperature > 0 强制要求）
- CUDA graph 仅用于 decode 且 batch ≤ 512
- 需要 GPU（CUDA）

## 相关逻辑

- [agents/engine](../agents/engine/index.md)
- [agents/layers](../agents/layers/index.md)
- [agents/models](../agents/models/index.md)
