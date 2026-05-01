Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# engine 模块

推理引擎核心，负责请求调度、KV cache 管理、序列生命周期和模型执行。

## 职责

- 将用户请求调度为 prefill/decode 两个阶段
- 管理 KV cache block 的分配、释放和 prefix caching
- 编排模型前向执行、采样、CUDA graph 捕获
- 管理多 GPU tensor parallel 进程间通信

## 关键抽象

### Sequence (`sequence.py`)
- 代表一个推理序列，携带 token_ids、block_table、状态（WAITING / RUNNING / FINISHED）
- 支持 pickle 序列化（`__getstate__` / `__setstate__`），用于跨进程传输
- prefill 时传输完整 token_ids；decode 时仅传 last_token

### Scheduler (`scheduler.py`)
- 维护 waiting / running 两个队列
- `schedule()` 返回 `(seqs, is_prefill)`：
  - prefill 阶段：chunked prefill，受 `max_num_batched_tokens` 和 `max_num_seqs` 限制
  - decode 阶段：每 seq 每步 1 token，block 不足时 preempt（抢占）running 尾部序列
- `postprocess()` 处理采样结果、更新 block hash、判断序列完成

### BlockManager (`block_manager.py`)
- 以 fixed-size block（默认 256 tokens）管理 KV cache
- prefix caching：通过 xxhash 计算 block 内容哈希，实现跨序列 block 共享
- 引用计数管理 block 生命周期
- `can_allocate()` 检查 prefix 命中；`allocate()` 复用共享 block 或分配新 block
- `may_append()` 在 decode 阶段按需扩展 block_table

### ModelRunner (`model_runner.py`)
- 初始化模型、分配 KV cache、warmup、CUDA graph 捕获
- `prepare_prefill()` / `prepare_decode()` 将 Sequence 列表转为模型输入张量
- 通过全局 Context 单例传递调度信息到各层
- tensor parallel：rank 0 通过 SharedMemory + Event 协调其他 rank
- CUDA graph：预捕获 batch size 为 1,2,4,8,16,32,...,512 的 decode graph

## 主流程

```
LLMEngine.generate(prompts, sampling_params)
  → add_request()           # tokenize + 创建 Sequence + 入队 waiting
  → step() 循环:
      → scheduler.schedule()          # 决定本轮 prefill/decode + 选出 seqs
      → model_runner.run(seqs, is_prefill)  # 准备输入 → 模型前向 → 采样
      → scheduler.postprocess()       # 更新 block hash、追加 token、判断完成
  → 解码输出 token_ids 为文本
```

## 依赖

- [layers](../layers/index.md)：Attention、Sampler
- [models](../models/index.md)：Qwen3ForCausalLM
- [utils](../utils/index.md)：Context、load_model
- transformers：AutoTokenizer（encode/decode）
- flash-attn：`flash_attn_varlen_func`、`flash_attn_with_kvcache`

## 非目标

- 不处理分布式推理以外的服务层逻辑（无 HTTP server）
- 不实现 speculative decoding、quantization 等
