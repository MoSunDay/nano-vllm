Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 7: 推理引擎

> 推理的核心瓶颈是**显存带宽**，不是算力。推理引擎的每一个优化都围绕这个事实展开。

---

## 7.1 Prefill vs Decode

### 为什么调度器必须存在

没有调度器，所有用户请求同时涌入 GPU。但 GPU 显存是有限的——每个请求都要占 KV cache，显存不够就 OOM（内存溢出），程序直接崩了。调度器像领位员控制餐厅流量：显存充足时放更多请求并行，显存紧张时让新请求排队。还要平衡 prefill 和 decode 两类任务的资源占用。

### 两个阶段

| | Prefill（预填充） | Decode（解码） |
|---|---------|--------|
| 什么时候 | 处理 prompt 的时候 | 逐 token 生成的时候 |
| 每次 token 数 | 整个 prompt（可能数千个） | 1 个 |
| 计算特性 | **compute-bound**（算力瓶颈） | **memory-bound**（显存带宽瓶颈） |
| KV Cache | 写入 | 读取 + 追加 1 个 |

### 为什么 Decode 是 memory-bound

Decode 每步只生成 1 个新 token，但要读取**所有历史 token 的 KV cache** 来做注意力。

```
序列长度 = 4096, 每步：
- 计算: 1 个 token 的矩阵乘法（很少）
- 读取: 4096 个 token 的 KV cache（很多）
- 瓶颈: 从显存搬数据的速度，不是 GPU 的计算速度
```

### nano-vllm 中的调度

```python
# nanovllm/engine/scheduler.py:25
def schedule(self):
    # 先做 prefill
    while self.waiting and ...:
        seq = self.waiting[0]
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        ...
    if scheduled_seqs:
        return scheduled_seqs, True   # is_prefill = True

    # 再做 decode
    while self.running and ...:
        seq.num_scheduled_tokens = 1   # 每次 1 个 token
        ...
    return scheduled_seqs, False      # is_prefill = False
```

---

## 7.2 KV Cache 管理

### 为什么 BlockManager 必须存在

没有块管理器，每个请求独占连续显存来存 KV cache——预分配了 2048 个 token 的空间但只用了 100 个，剩下 1948 个全空着被占着（浪费空间）；两个请求开头一样，各自的 KV cache 分别存了一份完全相同的内容（无法复用，重复计算）。块管理器把 KV cache 切成固定大小的块，用到多少就分配多少，相同前缀的块还能跨请求共享。

### 核心问题

显存有限。多个请求同时运行，每个请求都需要自己的 KV cache。怎么分配？

### PagedAttention

借鉴操作系统的**虚拟内存分页**思想：

1. 把 KV cache 分成固定大小的 **block**（每个 256 个 token）
2. 每个 sequence 按需分配 block，不需要连续的显存空间
3. 用 **block table** 记录每个 sequence 的 block 映射

```
Sequence A: [block_3] [block_7] [block_12]
Sequence B: [block_1] [block_5]
Sequence C: [block_2] [block_9] [block_11] [block_15]

block_table:
  A → [3, 7, 12, -1]
  B → [1, 5, -1, -1]
  C → [2, 9, 11, 15]
```

### nano-vllm 中的 BlockManager

```python
# nanovllm/engine/block_manager.py:26
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]  # 所有 block
        self.free_block_ids = deque(range(num_blocks))         # 空闲 block 队列
        self.used_block_ids = set()                             # 已用 block 集合
        self.hash_to_block_id = dict()                          # 哈希 → block 映射（prefix cache 用）
```

### block_size 为什么是 256

太小（如 16）→ 管理开销大，block table 太长
太大（如 4096）→ 内部碎片严重，浪费显存
256 是一个经验平衡点。

### 示例：模拟 block 分配

```python
from collections import deque

class SimpleBlockManager:
    def __init__(self, num_blocks, block_size=256):
        self.block_size = block_size
        self.free_blocks = deque(range(num_blocks))

    def allocate(self, num_tokens):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return None  # 显存不足
        block_table = [self.free_blocks.popleft() for _ in range(num_blocks_needed)]
        return block_table

    def deallocate(self, block_table):
        self.free_blocks.extend(block_table)

    def append(self, block_table, total_tokens):
        # 新 token 需要新 block 吗？
        if total_tokens % self.block_size == 1:  # 刚好需要新 block
            if self.free_blocks:
                block_table.append(self.free_blocks.popleft())
                return True
            return False  # 显存不足
        return True

bm = SimpleBlockManager(100, block_size=4)  # 小 block 方便演示

# 分配 sequence（7 个 token 需要 2 个 block）
table = bm.allocate(7)
print(f"Sequence block_table: {table}")          # [0, 1]
print(f"剩余空闲 blocks: {len(bm.free_blocks)}")  # 98

# 生成 1 个 token（总共 8 个，不需要新 block）
bm.append(table, 8)
print(f"生成后 block_table: {table}")  # [0, 1]（没变）

# 再生成 1 个 token（总共 9 个，需要新 block）
bm.append(table, 9)
print(f"生成后 block_table: {table}")  # [0, 1, 2]
```

---

## 7.3 Prefix Caching（前缀缓存）

### 问题

多个请求可能有相同的 system prompt：

```
请求 A: "你是一个翻译助手..." + "请翻译 Hello"
请求 B: "你是一个翻译助手..." + "请翻译 World"
```

相同的 system prompt 部分不需要重复计算 KV cache！

### 怎么做

给每个 block 算一个哈希值（基于内容），相同内容的 block 共享：

```python
# nanovllm/engine/block_manager.py:36
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前一个 block 的哈希（链式哈希）
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

关键：哈希是**链式的**——每个 block 的哈希依赖前一个 block 的哈希。这保证了相同 token 序列的前缀一定映射到相同的 block。

```python
# nanovllm/engine/block_manager.py:58
def can_allocate(self, seq):
    h = -1
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
    return num_cached_blocks
```

### 引用计数

一个 block 可能被多个 sequence 引用（共享）。只有引用计数归零才能释放：

```python
# nanovllm/engine/block_manager.py:94
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)  # 真正释放
```

---

## 7.4 Continuous Batching

### Static Batching 的问题

传统方式（static batching）：等 batch 中最长的序列完成才能返回结果。短序列白白等待。

### Continuous Batching

每个 sequence 独立调度：
- 完成的立即移出，腾出位置给等待的 sequence
- 每个 iteration 的 batch 组成可能不同

```python
# nanovllm/engine/scheduler.py:81
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if token_id == self.eos or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # 立即释放
            self.running.remove(seq)             # 立即移出
```

---

## 7.5 Tensor Parallelism（张量并行）

### 核心思想

把一个大矩阵拆到多张 GPU 上，每张 GPU 算一部分，最后合并。

### Column Parallel（列并行）

把权重矩阵**竖着切**（沿输出维度），每张卡存一列：

```
原始 W: (4096, 4096)
GPU 0: W[:, :2048]    算出 y 的前半
GPU 1: W[:, 2048:]    算出 y 的后半
拼接 → 完整的 y
```

```python
# nanovllm/layers/linear.py:54
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, divide(output_size, tp_size), ...)  # 输出维度除以 tp_size

    def forward(self, x):
        return F.linear(x, self.weight)  # 每张卡独立计算，输出是完整结果的一部分
```

### Row Parallel（行并行）

把权重矩阵**横着切**（沿输入维度），每张卡算一部分后 AllReduce 求和：

```
原始 W: (4096, 4096)
GPU 0: W[:2048, :]    算 x[:2048] @ W[:2048, :] = y_part_0
GPU 1: W[2048:, :]    算 x[2048:] @ W[2048:, :] = y_part_1
y = y_part_0 + y_part_1  （AllReduce）
```

```python
# nanovllm/layers/linear.py:131
class RowParallelLinear(LinearBase):
    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.tp_size > 1:
            dist.all_reduce(y)  # 所有卡的结果求和
        return y
```

### Attention 层的 TP 组合

```
QKV 投影: ColumnParallel（每张卡负责一部分头）
   ↓ 每张卡独立做注意力
O 投影:   RowParallel（每张卡的结果 AllReduce）
```

### MLP 层的 TP 组合

```
gate_up 投影: ColumnParallel（每张卡负责一部分中间维度）
   ↓ SiLU(gate) * up
down 投影:   RowParallel（AllReduce）
```

这样每个 Transformer Block 需要 **2 次 AllReduce**（分别在 O 投影和 down 投影处），这是经典的 TP 通信模式：每个"列并行→行并行"的组合产生一次 AllReduce。

---

## 7.6 CUDA Graph

### 问题

Decode 阶段每步只算 1 个 token，GPU 的计算量很小。但每次都有 CPU→GPU 的 kernel launch 开销（约 5-10μs），大量小 kernel 累积起来很慢。

### 解法

CUDA Graph 把一系列 GPU 操作"录制"下来，之后一次性"回放"，跳过 CPU 的调度：

```python
# nanovllm/engine/model_runner.py:222
def capture_cudagraph(self):
    for bs in reversed(self.graph_bs):  # 对不同 batch size 分别录制
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        self.graphs[bs] = graph  # 保存录制的 graph

# 回放时
graph.replay()  # 一次性执行所有 kernel
```

### 为什么需要多个 graph

不同 batch size 的 CUDA Graph 不能通用（显存布局不同）。nano-vllm 预录制了 `[1, 2, 4, 8, 16, 32, ...]` 等 batch size 的 graph。

```python
# nanovllm/engine/model_runner.py:234
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

运行时选一个 ≥ 当前 batch size 的最小 graph，多余的 slot 用 padding 填充。

---

## 7.7 torch.compile

### 是什么

PyTorch 2.0 的 JIT 编译器。把 Python 动态图分析、优化、编译成高效的 GPU kernel。

### 主要优化：算子融合（Operator Fusion）

```
优化前:
  x1 = op1(x)     → 读显存 1 次，写 1 次
  x2 = op2(x1)    → 读显存 1 次，写 1 次
  x3 = op3(x2)    → 读显存 1 次，写 1 次
  总计: 3 次读 + 3 次写

优化后（融合成 1 个 kernel）:
  x3 = fused_op123(x)  → 读显存 1 次，写 1 次
  总计: 1 次读 + 1 次写
```

对于 memory-bound 的 decode 阶段，减少显存访问的效果显著。

### nano-vllm 中的使用

```python
# RMSNorm: @torch.compile
# nanovllm/layers/layernorm.py:17

# Sampler: @torch.compile
# nanovllm/layers/sampler.py:8

# SiluAndMul: @torch.compile
# nanovllm/layers/activation.py:8
```

---

## 7.8 完整的一次推理流程

```python
# nanovllm/engine/model_runner.py:214
def run(self, seqs, is_prefill):
    # ① 准备输入数据
    if is_prefill:
        input_ids, positions = self.prepare_prefill(seqs)
    else:
        input_ids, positions = self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs)

    # ② 模型前向传播
    logits = self.run_model(input_ids, positions, is_prefill)
    # 内部: input_ids → Embedding → N × TransformerBlock → LM Head → logits

    # ③ 采样
    token_ids = self.sampler(logits, temperatures).tolist()
    return token_ids
```

加上外层调度循环（`nanovllm/engine/llm_engine.py`）：

```python
while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()     # 调度
    token_ids = model_runner.run(seqs, is_prefill)  # 执行
    scheduler.postprocess(seqs, token_ids, is_prefill)  # 后处理
```

---

## 小结

| 优化技术 | 解决的问题 | 原理 | 代码位置 |
|---------|-----------|------|---------|
| Prefill/Decode 分离 | 长短序列的计算模式不同 | 分别优化 | `scheduler.py:25` |
| PagedAttention | KV cache 显存管理 | 分块 + block table | `block_manager.py:26` |
| Prefix Caching | 相同前缀重复计算 | 哈希匹配 + block 共享 | `block_manager.py:36` |
| Continuous Batching | 短序列等长序列 | 独立调度，即时回收 | `scheduler.py:81` |
| Tensor Parallelism | 单卡放不下 | 矩阵拆分到多卡 | `layers/linear.py` |
| CUDA Graph | decode 时 kernel launch 开销 | 预录制执行计划 | `model_runner.py:222` |
| torch.compile | 多次显存读写 | 算子融合 | `layers/*.py` @torch.compile |

---

**至此，你已经学完了从张量到推理引擎的全部知识。** 回到 [index.md](index.md) 查看"总"的部分，确认你能串起完整的链路。
