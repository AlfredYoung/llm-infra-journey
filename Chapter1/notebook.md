# LLM Chapter 1 学习记录：从零实现一个简单 Transformer Block（含 Causal Mask）

> 目标：用 PyTorch 手写一个最小可工作的 Transformer Block，并理解张量形状、Self-Attention、Causal Mask 的意义与实现方式。

---

## 1. 本章实现内容概览

本文件实现了三个核心模块：

- `MultiHeadAttention`：多头自注意力（Multi-Head Self-Attention）
- `FeedForward`：前馈网络（Position-wise FFN）
- `Transformer`：一个标准 Transformer Block（Residual + LayerNorm + Dropout）

当前版本已实现：
- ✅ Q/K/V 三个线性投影
- ✅ Multi-Head reshape 与 head 拼接
- ✅ Scaled Dot-Product Attention
- ✅ Causal Mask（自回归遮罩，禁止看未来 token）
- ✅ Residual connection + LayerNorm + FFN

尚未实现（下一章可加）：
- ⏳ 位置编码（Sinusoidal / RoPE）
- ⏳ Padding mask（变长序列）
- ⏳ Attention dropout
- ⏳ Pre-LN（现代 LLM 常用）结构改造

---

## 2. 关键张量形状（必须掌握）

设：
- `B` = batch_size
- `S` = seq_len
- `d_model` = 模型隐藏维度
- `H` = num_heads
- `D` = head_dim = d_model / H

输入：
- `x`: `[B, S, d_model]`

在注意力中：

1) 线性投影后：
- `Q,K,V`: `[B, S, d_model]`

2) 分头后（view + transpose）：
- `Q,K,V`: `[B, H, S, D]`

3) 注意力分数：
- `scores = Q @ K^T`: `[B, H, S, S]`

4) softmax 得到注意力权重：
- `attn_score`: `[B, H, S, S]`

5) 加权求和输出：
- `attn_output`: `[B, H, S, D]`

6) 拼回 d_model（transpose + contiguous + view）：
- `attn_output`: `[B, S, d_model]`

---

## 3. 为什么 Self-Attention 是 (x, x, x)

在 `Transformer.forward()` 中：

```python
attn_output = self.mha(x, x, x)
