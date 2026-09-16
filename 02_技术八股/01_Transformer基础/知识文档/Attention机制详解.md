# Attention 机制详解（第 2 层知识笔记）

> 💡 **使用指南**：本文件是「第二层：知识笔记」，重点讲清楚 Attention 的公式、Q/K/V 的含义、Multi-Head 的数学形式以及不同类型的 Attention。  
> 建议先读完本文件，再去看 `面试答题_Attention.md`，用里面的口语化模板准备面试。

---

## 一、Scaled Dot-Product Attention 公式

### 1.1 标准公式

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

**符号说明：**
- \(Q \in \mathbb{R}^{L_q \times d_k}\)：Query，查询向量；
- \(K \in \mathbb{R}^{L_k \times d_k}\)：Key，键向量；
- \(V \in \mathbb{R}^{L_k \times d_v}\)：Value，值向量；
- \(d_k\)：Key / Query 的维度，用于缩放；
- \(L_q, L_k\)：query 序列长度、key/value 序列长度。

### 1.2 逐步拆解

1. **相似度计算：\(QK^T\)**

```text
Q: (seq_len, d_k)
K: (seq_len, d_k)

QK^T: (seq_len, seq_len)

例如句子 "我 爱 中国" (3个词)：

QK^T = [
  [s11, s12, s13],  # "我" 对所有词的相似度
  [s21, s22, s23],  # "爱" 对所有词的相似度
  [s31, s32, s33]   # "中国" 对所有词的相似度
]
```

2. **缩放：除以 \(\sqrt{d_k}\)**

原因：防止点积结果随维度增大而变得过大，Softmax 区分度过强、梯度变小。

3. **Softmax 归一化**

```text
Attention_Weights = softmax(QK^T / sqrt(d_k))

每一行和为 1，表示当前 query 对所有 key 的「注意力分布」。
```

4. **对 V 加权求和**

```text
Output = Attention_Weights · V
       = (seq_len, seq_len) × (seq_len, d_v)
       = (seq_len, d_v)
```

---

## 二、Q / K / V 的含义

### 2.1 搜索引擎类比（直观）

```text
Q (Query):   你的搜索词，例如 "如何学习 Transformer"
K (Key):     每篇文档的标题/关键字
V (Value):   每篇文档的实际内容

流程：
1. 用 Query 和每个 Key 计算相似度 → 匹配分数
2. 对匹配分数做 Softmax → 得到权重
3. 用权重对 Value 加权求和 → 得到最终返回的内容表示
```

### 2.2 在 NLP 中的映射

```text
输入句子: "我 爱 中国"

当计算 "爱" 的新表示时：
Q("爱"): "爱" 想要查询什么信息？
K("我", "爱", "中国"): 每个词能提供什么信息？
V("我", "爱", "中国"): 每个词的具体内容表示
```

### 2.3 为什么要分三套矩阵？

如果直接用同一个矩阵 X：

```text
Attention(X, X, X) = softmax(XX^T / sqrt(d)) · X
```

只能在一个空间里做自相似度建模，表达力有限。  
使用独立的 \(W_Q, W_K, W_V\)：

```text
Q = X · W_Q   # 学习“查询的视角”
K = X · W_K   # 学习“被查询的视角”
V = X · W_V   # 学习“输出的表示空间”
```

这样可以：
- 在不同子空间里度量相似度；
- 输出的 V 空间可以和 Q/K 不同，更灵活。

---

## 三、为什么要除以 sqrt(d_k)？

### 3.1 问题：维度越大，点积方差越大

假设 Q、K 的各维是均值 0、方差 1 的独立随机变量：

```text
点积：q · k = Σ_{i=1..d_k} q_i k_i

E[q · k] = 0
Var[q · k] = d_k    # 方差随 d_k 线性增长
```

当 \(d_k\) 很大时，点积的数值范围会很大，经过 Softmax 后：

- 一个位置概率接近 1；
- 其他位置接近 0；
- 梯度非常小，训练不稳定。

### 3.2 解决方案：缩放

```text
Scores = QK^T / sqrt(d_k)

Var[q · k / sqrt(d_k)] = Var[q · k] / d_k = 1
```

即通过除以 \(\sqrt{d_k}\) 把方差归一到 1，使不同维度下分布尺度一致，Softmax 更稳定。

---

## 四、不同类型的 Attention

### 4.1 Self-Attention（自注意力）

- 定义：Q、K、V 都来自**同一序列**；
- 作用：在同一句话内部，让每个 token 根据上下文重新表示自己；
- Transformer 编码器中的 Self-Attention 通常是「双向」的。

### 4.2 Cross-Attention（交叉注意力）

- 定义：Q 来自目标序列（decoder），K/V 来自源序列（encoder）；
- 用途：在翻译 / seq2seq 中，解码器在生成时「查阅」编码器输出。

### 4.3 Masked Self-Attention（掩码自注意力）

- 定义：对未来位置做 Mask，保证每个位置只能看见自己和之前的 token；
- 用途：自回归语言模型（如 GPT）在训练/生成中使用，避免「偷看未来」。

---

## 五、Multi-Head Attention 概述

### 5.1 数学形式

```text
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
```

- 将输入先投影到多个低维子空间；
- 在各子空间分别做 Self-Attention；
- 再把所有 head 拼接起来，过一个线性层。

### 5.2 多头的直觉

- 有的头专注于局部邻近词；
- 有的头专注于长距离依赖；
- 有的头专注于特定词性 / 句法结构；
- 通过这种「多视角」方式，模型对句子结构的理解更全面。

---

## 六、面试高频考点速查

> 这里只列考点标题，具体的口语化答题模板在 `面试答题_Attention.md` 中。

- 手写并解释公式：`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V`；
- 说明 Q/K/V 分别代表什么，为什么需要三套矩阵；
- 解释为什么要除以 `sqrt(d_k)`，能从方差角度推导一遍；
- 区分 Self-Attention / Cross-Attention / Masked Attention；
- 说明 Multi-Head 的优势和数学形式。

---

## 七、配套文档

- 📚 上层大纲：`理论.md` 中「二、Attention 机制」章节  
- 📝 面试答题示例：`面试答题_Attention.md`  
- 📘 相关主题：`Transformer原理详解.md`、`BERT与GPT的架构关系详解.md`


