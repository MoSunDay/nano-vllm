Commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8

# 大模型 & 推理框架学习路线

> 从"一个数字"到"一整座推理引擎"——总分总结构

## 总：全貌

大模型在做的事，用一句话概括：

**把一段文字的每个字变成数字，经过几十层数学运算，输出下一个字的概率。**

这引出五个阶段，也就是整个学习路线的主干：

```
文字 → Embedding（翻译）→ Transformer Layers（加工）→ LM Head（翻译回来）→ Sampling（选字）
```

以及一个贯穿始终的问题：**怎么让这个过程更快？**——这就是推理引擎要解决的事。

## 分：逐层拆解

建议按以下顺序学习（MoE 放最后，因为它建立在已理解 FFN 的基础上）：

| Phase | 文件 | 主题 | 核心问题 |
|-------|------|------|----------|
| 0 | [00-prerequisites.md](00-prerequisites.md) | 前置知识 | 张量、GPU 并行、装饰器、浮点精度 |
| 1 | [01-parameters.md](01-parameters.md) | 参数 | 模型的 70 亿个数字分别是什么？ |
| 2 | [02-embedding-tokenizer.md](02-embedding-tokenizer.md) | Embedding & Tokenizer | 文字和数字怎么互转？ |
| 3 | [03-attention.md](03-attention.md) | 注意力机制 | 每个字怎么"看到"所有其他字？ |
| 4 | [04-transformer-block.md](04-transformer-block.md) | Transformer Block | 注意力之外还有什么组件？ |
| 5 | [05-moe.md](05-moe.md) | MoE（混合专家） | 不是每字都需要全部参数？ |
| 6 | [06-sampling.md](06-sampling.md) | 采样 | 从概率分布到最终文字？ |
| 7 | [07-inference-engine.md](07-inference-engine.md) | 推理引擎 | 怎么让推理又快又省？ |

## 总：串起来

学完所有 Phase 后，你应该能完整回答：

> **"用户输入一句话，nano-vllm 做了什么才返回结果？"**

```
1. Tokenizer 把文字切成 token → Embedding 层查表变成向量
2. Scheduler 决定这个请求什么时候开始执行
3. BlockManager 分配 KV cache 的显存空间
4. 向量经过 N 层 Transformer Block：
   a. RMSNorm 归一化
   b. 多头注意力（QKV 投影 → RoPE 加位置 → FlashAttention → 输出投影）
   c. 残差连接
   d. RMSNorm 归一化
   e. MLP（gate+up 投影 → SiLU 激活 → down 投影）
   f. 残差连接
5. LM Head 把最终向量投影到词表大小的 logits
6. Sampler 根据 temperature 等参数采样出下一个 token
7. 如果还没结束，回到第 2 步继续（Scheduler 重新调度下一个 iteration）
```

## 学习方式

每个 Phase 的学习步骤：**先读概念 → 再去 nano-vllm 对应模块读源码 → 最后动手写实验代码。**

每个文档都包含：
- **概念讲解**：用最直白的方式解释"这是什么、为什么需要"
- **代码示例**：可独立运行的 Python 示例
- **源码对应**：指向 nano-vllm 中的具体文件和行号
