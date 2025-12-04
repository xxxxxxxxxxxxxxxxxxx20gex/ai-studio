# Transformer 面试答题详解（第 3 层：答题示例）

> 💡 **使用说明**：本文件属于「第三层：面试答题详解」，重点是**口语化答题模板**和**结构化表达**。  
> 在真正背答案之前，强烈建议先阅读配套的第二层知识笔记：`Transformer原理详解.md`，理解原理后再来练习答题会轻松很多。

---

## 目录

- [Q1: 请画出Transformer的结构图并解释](#q1-请画出transformer的结构图并解释)
- [Q2: 详细解释Self-Attention的计算过程](#q2-详细解释self-attention的计算过程)
- [Q3: 为什么需要Multi-Head Attention？](#q3-为什么需要multi-head-attention)
- [Q4: Position Encoding的作用和实现方式？](#q4-position-encoding的作用和实现方式)

---

## Q1: 请画出Transformer的结构图并解释

**答题思路:**

```text
Transformer整体结构（左右并列）：

[输入文本] → Embedding + Position Encoding
                    ↓
┌─────────────────────────────────────┐
│          Encoder (×6层)            │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │  ← 看全文
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
│  ┌─────────────────────────────┐   │
│  │    Feed Forward Network     │   │
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│          Decoder (×6层)            │
│  ┌─────────────────────────────┐   │
│  │ Masked Multi-Head Attention │   │  ← 只看前文
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
│  ┌─────────────────────────────┐   │
│  │  Cross-Attention (K,V来自   │   │  ← 连接编码器
│  │       Encoder输出)          │   │
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
│  ┌─────────────────────────────┐   │
│  │    Feed Forward Network     │   │
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
└─────────────────────────────────────┘
                    ↓
            Linear + Softmax
                    ↓
              [输出概率分布]
```

**口述解释:**

> 「Transformer 由编码器和解码器组成，各 6 层堆叠。编码器每层包含 Multi-Head Self-Attention 和前馈网络，用来**理解输入序列**；解码器在此基础上，多了 Masked Self-Attention（防止看到未来）和 Cross-Attention（从编码器获取信息）。  
> 每个子层后面都有残差连接和 LayerNorm，保证训练稳定；最后通过线性层 + Softmax 得到每个位置的输出概率。」

---

## Q2: 详细解释Self-Attention的计算过程

**答题思路:**

1. 先说输入/输出的形状；  
2. 再按 Q/K/V → Scores → Softmax → 加权求和 这 4 步讲；  
3. 最后用一个中文例子收尾。

**步骤 1：线性变换生成 Q、K、V**

```text
输入: X ∈ R^(seq_len × d_model)  // 例如 (10, 512)

Q = X · W_Q  // (10, 512) × (512, 64) = (10, 64)
K = X · W_K  // (10, 512) × (512, 64) = (10, 64)
V = X · W_V  // (10, 512) × (512, 64) = (10, 64)
```

**步骤 2：计算注意力分数**

```text
Scores = Q · K^T / sqrt(d_k)
       = (10, 64) × (64, 10) / sqrt(64)
       = (10, 10) / 8

这个 (10×10) 矩阵表示每个词对其他词的关注度
```

**步骤 3：Softmax 归一化**

```text
Attention_Weights = softmax(Scores)  // 每行和为 1
```

**步骤 4：加权求和得到输出**

```text
Output = Attention_Weights · V
       = (10, 10) × (10, 64)
       = (10, 64)
```

**举例说明:**

> 「比如输入『我 爱 中国』，在计算『爱』的表示时，会先算出『爱』对『我』『爱』『中国』的注意力分数（比如 0.2, 0.3, 0.5），然后用这些权重加权求和三个词的 V 向量，得到融合了上下文信息的『爱』的新表示。」

---

## Q3: 为什么需要Multi-Head Attention？

**核心答案：** 让模型从不同「子空间」和不同「关系视角」去看同一句话，提升表达能力。

**详细解释:**

1. **单个 Attention 的局限：**
   - 只能学到一种注意力模式；
   - 可能只关注语法，忽略语义，或者只关注局部，忽略远距离依赖。

2. **Multi-Head 的优势：**

```text
8 个头 (h = 8)，每个头可以学习不同的模式：

Head 1: 主谓关系
        "我 爱 中国" → "我" 关注 "爱"

Head 2: 修饰关系
        "美丽的 中国" → "中国" 关注 "美丽的"

Head 3: 长距离依赖
        "开头的词" 关注 "结尾的词"

Head 4-8: 其他语义/句法模式
```

3. **参数与计算量：**

```text
单头: d_model = 512
多头: 8 个头，每头 d_k = 512 / 8 = 64

计算量相近，但表达能力更强
```

4. **类比：**

> 「就像 CNN 用多个卷积核提取不同特征，Multi-Head 用多个注意力头捕捉不同的语义和句法关系。」

**加分点：**

> 「实际上，在分析 BERT/GPT 的注意力头时，人们发现有的头专门关注局部邻近词，有的专门关注长距离依赖，还有的关注句法结构，这说明多头确实学到了不同的模式。」

---

## Q4: Position Encoding 的作用和实现方式？

**问题背景：**

- Attention 本身是 **排列不变（permutation invariant）** 的：
  - 打乱输入顺序，单纯的 Self-Attention 输出不变；
  - 但语言是有顺序的，必须显式加入位置信息。

**作用：**

> 「告诉模型每个 token 的位置信息，让模型区分『我爱你』和『你爱我』。」

### 实现方式 1：正弦位置编码（Sinusoidal PE）

**公式：**

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**代码实现示意：**

```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    """
    生成正弦位置编码矩阵
    seq_len: 序列长度
    d_model: 模型隐层维度
    """
    # 初始化位置编码矩阵 (seq_len, d_model)
    PE = np.zeros((seq_len, d_model))

    # 位置索引 (0, 1, 2, ..., seq_len-1)
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)

    # 维度索引（步长为 2，只算偶数维，再用 sin/cos 组合）
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )  # (d_model/2,)

    # 偶数维: sin
    PE[:, 0::2] = np.sin(position * div_term)

    # 奇数维: cos
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
```

**优点：**

- 不需要训练参数；
- 可以推广到比训练时更长的序列；
- 绝对位置 & 相对位置信息都可以通过三角性质表示出来。

### 实现方式 2：可学习位置编码（Learnable PE）

```python
# 直接学习一个位置嵌入表
position_embeddings = nn.Embedding(max_position, d_model)

# 使用时
pos_ids = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
pos_emb = position_embeddings(pos_ids)
```

**优点：**
- 任务自适应，更灵活；

**缺点：**
- 无法自然外推到未见过的长度（比如训练时最长 512，推理时直接用 4096 会出问题）。

### 实现方式 3：相对位置编码 / RoPE / ALiBi（加分）

> 「更进一步，像 Transformer-XL、T5、DeBERTa 采用的是相对位置编码；LLaMA 等模型使用 RoPE（旋转位置编码），ALiBi 则在注意力得分里直接注入相对偏移，这些方法都更适合长文本场景，是现在大模型里的热点。」

---

## 结语与配套

- 理论回顾：建议结合 `Transformer原理详解.md` 把每一题涉及的公式和图再过一遍；  
- 面试实战：可以对着题目自己先回答一遍，再对照本文件优化自己的表达结构和用词。  
- 相关主题：
  - `Attention机制详解.md` + `面试答题_Attention.md`
  - `BERT与GPT的架构关系详解.md` + `面试答题_BERT_GPT.md`


