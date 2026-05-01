Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# Phase 0: 前置知识

> 没有这些，后面的一切都是空中楼阁。

---

## 0.1 张量（Tensor）

### 是什么

张量就是多维数组。它是深度学习中所有数据的基本容器。

| 维度 | 名称 | 形状示例 | 含义 |
|------|------|----------|------|
| 0D | 标量 | `()` | 一个数字，如 loss 值 |
| 1D | 向量 | `(4096,)` | 一个 token 的隐藏表示 |
| 2D | 矩阵 | `(4096, 4096)` | 一层线性变换的权重 |
| 3D | 张量 | `(32, 4096, 4096)` | 一个 batch 的隐藏状态 |
| 4D | 张量 | `(2, 64, 2048, 128)` | KV cache（层×块×头×维度） |

### 为什么 GPU 擅长算矩阵乘法

GPU 有几千个小核心（如 A100 有 6912 个 CUDA 核心），每个核心能同时做一次乘加运算。矩阵乘法 `C = A × B` 的每个输出元素独立可并行，正好适合 GPU 的 SIMD（单指令多数据）架构。

### 示例

```python
import torch

# 创建张量
scalar = torch.tensor(3.14)                    # 0D
vector = torch.randn(4096)                      # 1D：一个 token 的隐藏表示
matrix = torch.randn(4096, 4096)                # 2D：一层线性权重
hidden = torch.randn(32, 4096)                  # 3D：batch=32 的隐藏状态
kv_cache = torch.randn(2, 28, 1024, 4, 128)    # 5D：KV cache (K/V, layers, blocks, heads, dim)

# 矩阵乘法 —— 这就是线性变换的本质
x = torch.randn(1, 4096)       # 输入：1个token
W = torch.randn(4096, 4096)    # 权重矩阵
y = x @ W                      # 输出：(1, 4096) @ (4096, 4096) → (1, 4096)
print(f"输入形状: {x.shape}, 权重形状: {W.shape}, 输出形状: {y.shape}")

# 这等价于 nn.Linear
import torch.nn as nn
linear = nn.Linear(4096, 4096, bias=False)
linear.weight.data = W.T  # PyTorch 的权重是转置存储的
y2 = linear(x)
print(f"手动计算和 nn.Linear 结果一致: {torch.allclose(y, y2, atol=1e-5)}")

# GPU 加速
if torch.cuda.is_available():
    x_gpu = x.cuda()
    W_gpu = W.cuda()
    y_gpu = x_gpu @ W_gpu
    print(f"GPU 计算: {y_gpu.device}")
```

---

## 0.2 GPU 并行基础

### CUDA 编程模型

GPU 的计算由三层层次组织：

```
Grid（网格）
 └── Block（块）—— 可以有几千个
      └── Thread（线程）—— 每个块最多 1024 个
```

- **Grid** = 一次 kernel launch 的全部工作
- **Block** = 一组互相可以同步、共享内存的线程
- **Thread** = 执行一个最小计算单元

### 为什么这和深度学习有关

当 nano-vllm 执行一次注意力计算时，GPU 把工作分配给成千上万个线程：
- 每个 thread 处理一个或几个元素的乘加
- 同一个 block 内的 thread 可以共享 fast shared memory
- FlashAttention 就是精心设计 thread 和 shared memory 的使用来减少全局显存访问

### 示例

```python
import torch

if torch.cuda.is_available():
    # 查看 GPU 信息
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"CUDA 核心数: {torch.cuda.get_device_properties(0).multi_processor_count} SM")

    # 多 GPU（Tensor Parallelism）
    print(f"可用 GPU 数: {torch.cuda.device_count()}")

    # 在不同 GPU 上创建张量
    x0 = torch.randn(100, 100, device="cuda:0")
    x1 = torch.randn(100, 100, device="cuda:1") if torch.cuda.device_count() > 1 else None
```

---

## 0.3 Python 装饰器与 torch.compile

### 装饰器是什么

装饰器是一个函数，它接收一个函数，返回一个增强版函数。本质是语法糖：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__} 之前")
        result = func(*args, **kwargs)
        print(f"调用 {func.__name__} 之后")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# 输出：
# 调用 say_hello 之前
# Hello!
# 调用 say_hello 之后
```

### torch.compile 做了什么

`@torch.compile` 是 PyTorch 2.0 引入的 JIT 编译器装饰器。它把 Python 的动态执行图编译成静态的融合内核（fused kernel），减少 Python 开销和 GPU kernel launch 次数。

在 nano-vllm 中的使用：

```python
# nanovllm/layers/sampler.py:8
class Sampler(nn.Module):
    @torch.compile                                    # ← 编译采样逻辑
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

# nanovllm/layers/layernorm.py:17
class RMSNorm(nn.Module):
    @torch.compile                                    # ← 编译归一化逻辑
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x
```

### 示例：感受 torch.compile 的加速

```python
import torch
import torch.nn as nn
import time

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 11008)
        self.fc2 = nn.Linear(11008, 4096)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().cuda()
x = torch.randn(1, 4096, device="cuda")

# 不编译
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    model(x)
torch.cuda.synchronize()
print(f"不编译: {time.time() - t0:.3f}s")

# 编译后
compiled = torch.compile(model)
compiled(x)  # warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    compiled(x)
torch.cuda.synchronize()
print(f"编译后: {time.time() - t0:.3f}s")
```

---

## 0.4 浮点数精度

### 三种常用精度

| 类型 | 位数 | 范围 | 精度 | 每个数占空间 |
|------|------|------|------|-------------|
| FP32 | 32 bit | ±3.4×10³⁸ | ~7 位有效数字 | 4 字节 |
| FP16 | 16 bit | ±6.5×10⁴ | ~3 位有效数字 | 2 字节 |
| BF16 | 16 bit | ±3.4×10³⁸ | ~3 位有效数字 | 2 字节 |

关键区别：
- **FP16** 精度好但范围小，容易溢出（overflow/underflow）
- **BF16** 范围和 FP32 一样大，但精度低一些 → 训练更稳定
- 大模型推理通常用 **FP16 或 BF16**，比 FP32 省一半显存

### 显存计算

一个 7B 参数的模型：
- FP32: `7B × 4 bytes = 28 GB`
- FP16/BF16: `7B × 2 bytes = 14 GB`

### 示例

```python
import torch

value = 3.141592653589793

fp32 = torch.tensor(value, dtype=torch.float32)
fp16 = torch.tensor(value, dtype=torch.float16)
bf16 = torch.tensor(value, dtype=torch.bfloat16)

print(f"原始值:    {value}")
print(f"FP32:      {fp32.item():.15f}")   # ~7 位有效
print(f"FP16:      {fp16.item():.15f}")   # ~3 位有效
print(f"BF16:      {bf16.item():.15f}")   # ~3 位有效

# FP16 的溢出问题
big_number = torch.tensor(70000.0, dtype=torch.float16)
print(f"\nFP16 存 70000: {big_number.item()}")  # inf!

# BF16 不会溢出
big_number_bf16 = torch.tensor(70000.0, dtype=torch.bfloat16)
print(f"BF16 存 70000: {big_number_bf16.item()}")  # 正常

# 在 nano-vllm 中，模型精度由 config.hf_config.dtype 决定
# nanovllm/engine/model_runner.py:29
#   torch.set_default_dtype(hf_config.dtype)  # 通常是 torch.float16 或 torch.bfloat16
```

---

## 小结

| 概念 | 一句话 | 后续在哪用到 |
|------|--------|-------------|
| 张量 | 多维数组，深度学习的基本数据容器 | 所有计算 |
| GPU 并行 | 几千核心同时算，矩阵乘法天然适合 | 线性层、注意力、采样 |
| 装饰器 / torch.compile | 编译 Python 代码成融合内核，减少开销 | RMSNorm、Sampler、SiluAndMul |
| 浮点精度 | FP16/BF16 省显存，BF16 不溢出 | 模型加载、KV cache 分配 |

下一步：[01-parameters.md](01-parameters.md) —— 模型的 70 亿个数字分别是什么？
