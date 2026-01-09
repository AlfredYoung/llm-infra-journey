
# LLM Chapter 1 学习记录：从零实现一个 Mini GPT（含 Causal Mask & RoPE）

> 目标：  
> 用 PyTorch **手写一个最小但“现代”的 GPT 模型**，  
> 覆盖 Transformer 的核心组件，并理解：
>
> - Self-Attention 的本质
> - Causal Mask 如何实现自回归
> - RoPE 位置编码的工程实现
> - GPT 训练与推理（生成）的完整闭环

---

## 1. 本章最终实现了什么？

本章不是“只写一个 Attention Demo”，而是完成了一个 **可训练、可生成文本的 Mini GPT**。

### ✅ 已实现模块

#### 模型结构
- `RoPE`：旋转位置编码（Rotary Positional Embedding）
- `MultiHeadAttention`：多头自注意力（支持 causal mask + RoPE）
- `FeedForward`：前馈网络（SwiGLU 变体）
- `Transformer`：Pre-LN Transformer Block
- `MiniGPT`：完整 GPT Decoder-only 模型

#### 训练 & 推理
- 自回归语言模型训练（next-token prediction）
- AdamW + weight decay
- warmup + cosine learning rate decay
- 梯度裁剪（gradient clipping）
- perplexity / accuracy 评估
- greedy decoding 生成文本

---

## 2. GPT / Decoder-only Transformer 总览

GPT 使用的是 **Decoder-only Transformer**，每一层结构如下：

```
x
│
├── LayerNorm
├── Multi-Head Self-Attention (Causal)
├── Residual Add
│
├── LayerNorm
├── FeedForward (SwiGLU)
├── Residual Add
│
└── output
```

---

## 3. 关键张量形状（必须完全掌握）

设：

- B = batch_size
- S = seq_len
- C = d_model
- H = num_heads
- D = head_dim = C / H

---

## 4. Causal Mask 的意义

Causal Mask 确保模型在预测第 t 个 token 时，
**只能看到 0~t 的 token，而不能看到未来信息**。

---

## 5. RoPE：旋转位置编码

RoPE 将位置信息直接编码进 Q/K 中：

```
x_even' = x_even * cos - x_odd * sin
x_odd'  = x_even * sin + x_odd * cos
```

相比传统绝对位置编码，RoPE 更适合长上下文和外推。

---

## 6. FeedForward：SwiGLU 结构

使用现代 LLM 常见的 SwiGLU 结构：

```
FFN(x) = W3( SiLU(W1(x)) ⊙ W2(x) )
```

---

## 7. Pre-LN Transformer

LayerNorm 放在子层之前（Pre-LN），
可显著提升深层模型训练稳定性。

---

## 8. MiniGPT 总体结构

```
Token Embedding
↓
N × Transformer Block
↓
Final LayerNorm
↓
Linear LM Head（Weight Tying）
```

---

## 9. 训练目标

语言模型的训练目标是 **Next Token Prediction**：

```
[t0, t1, t2] → 预测 t3
```

Loss 使用 Cross Entropy。

---

## 10. 推理：Greedy Decoding

每一步选择概率最大的 token 作为下一个输出。

---

## 11. 本章总结

到本章结束，你已经：

- 手写了一个现代 GPT
- 理解了 Attention / Causal Mask / RoPE
- 跑通了训练与生成闭环

**这是从“理解 Transformer”迈向“实现 LLM”的关键一步。**

---

## 12. 下一章展望

- KV Cache
- 增量推理
- Flash Attention
- 更真实的数据集与 Tokenizer
