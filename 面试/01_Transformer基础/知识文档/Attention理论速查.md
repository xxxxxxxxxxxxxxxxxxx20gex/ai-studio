# Attention 机制理论速查 ⭐⭐⭐⭐⭐

> 💡 本文档为快速复习使用，详细内容请参考 `Attention机制详解.md`

## 1. Attention公式（必背）

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**参数说明:**
- Q (Query): 查询向量，shape = (batch_size, seq_len, d_k)
- K (Key): 键向量，shape = (batch_size, seq_len, d_k)
- V (Value): 值向量，shape = (batch_size, seq_len, d_v)
- d_k: Key的维度，用于缩放防止softmax梯度消失
- sqrt(d_k): 缩放因子

## 2. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O

其中 head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
```

## 3. 不同类型的Attention

- **Self-Attention**: Q=K=V（同一序列内部）
- **Cross-Attention**: Q来自decoder，K和V来自encoder
- **Masked Attention**: 解码时遮蔽未来信息

## 4. 面试重点

- ✅ 手写Attention公式
- ✅ 解释为什么要除以sqrt(d_k) ⭐⭐⭐
- ✅ 说明Q、K、V的作用
- ✅ Multi-Head的优势

## 相关文档

- 📖 [Attention机制详解](./Attention机制详解.md) - 完整理论
- 📖 [面试答题_Attention](./面试答题_Attention.md) - 面试答题示例
- 📄 [PE.py](../代码实现/PE.py) - 位置编码实现

