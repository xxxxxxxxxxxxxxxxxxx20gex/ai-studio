# Transformer 理论速查 ⭐⭐⭐⭐⭐

> 💡 本文档为快速复习使用，详细内容请参考 `Transformer原理详解.md`

## 1. 核心架构

- **编码器-解码器结构**
  - Encoder: 6层堆叠，每层包含Multi-Head Attention + FFN
  - Decoder: 6层堆叠，每层包含Masked Multi-Head Attention + Cross Attention + FFN
  
- **关键组件**
  - Self-Attention机制（核心）
  - Position Encoding（位置编码）
  - Multi-Head Attention（多头注意力）
  - Feed Forward Network（前馈网络）
  - Layer Normalization + Residual Connection（层归一化+残差连接）

## 2. 为什么Transformer重要？

- 并行计算能力强（相比RNN）
- 长距离依赖建模能力强
- 成为现代大模型（GPT、BERT等）的基础架构

## 3. 面试重点

- ✅ 能画出Transformer结构图
- ✅ 理解Self-Attention的计算过程
- ✅ 解释为什么需要Multi-Head
- ✅ 说明Position Encoding的作用和实现方式

## 相关文档

- 📖 [Transformer原理详解](./Transformer原理详解.md) - 完整理论
- 📖 [面试答题_Transformer](./面试答题_Transformer.md) - 面试答题示例
- 📄 [transformer.py](../代码实现/transformer.py) - 代码实现

