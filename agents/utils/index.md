Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# utils 模块

全局基础设施工具。

## 职责

提供跨层共享的全局上下文和模型权重加载。

## 关键组件

### Context (`context.py`)
- 全局单例 `_CONTEXT`（`Context` dataclass），存储当前推理步的调度信息
- 字段：`is_prefill`, `cu_seqlens_q/k`, `max_seqlen_q/k`, `slot_mapping`, `context_lens`, `block_tables`
- `set_context()` / `get_context()` / `reset_context()`：由 ModelRunner 在每步调用
- 各层（Attention、LMHead）通过 `get_context()` 读取调度参数，避免层层传参

### Loader (`loader.py`)
- `load_model(model, path)`：遍历 safetensors 文件，按 `packed_modules_mapping` 解析合并权重
- 调用参数上的 `weight_loader` 方法完成 TP 分片加载

## 设计要点

- Context 采用全局可变状态模式，生命周期为单次 `run()` 调用
- Loader 不依赖模型类型，通过 `packed_modules_mapping` + `weight_loader` 协议实现通用加载
