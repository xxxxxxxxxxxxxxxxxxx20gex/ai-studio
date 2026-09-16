# Transformer 原理详解（第 2 层知识笔记）

> 💡 **使用指南**：本文件属于「第二层：知识笔记」，主要讲清楚概念、公式和直觉，适合用来打基础、梳理框架。  
> 面试前建议先通读本文件，再去看对应的「第三层：面试答题示例」：`面试答题_Transformer.md`。
>
> **推荐阅读顺序：**
> 1. 总体结构与组件
> 2. Self-Attention 计算过程
> 3. Multi-Head 设计原因
> 4. 位置编码（Position Encoding）
> 5. 与 RNN/CNN 的对比与优缺点
> 6. 面试高频考点速查

---

## 一、Transformer 整体结构

### 1.1 高层结构示意

```text
[输入文本] → Embedding + Position Encoding
                    ↓
┌─────────────────────────────────────┐
│          Encoder (×6层)            │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │  ← 看全文做自注意力
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
│  │ Masked Multi-Head Attention │   │  ← 只看前文的自注意力
│  └─────────────────────────────┘   │
│            ↓ (Add & Norm)         │
│  ┌─────────────────────────────┐   │
│  │  Cross-Attention            │   │  ← K,V 来自 Encoder 输出
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

**核心记忆点：**
- 编码器（Encoder）堆叠多层 `Self-Attention + FFN`，用于「理解输入」。
- 解码器（Decoder）在每层中有：
  - Masked Self-Attention：保证解码时只能看到前文，适合自回归生成。
  - Cross-Attention：从编码器输出中「读信息」。
- 每个子层后都有 **残差连接（Residual）+ LayerNorm**，保证训练稳定。

---

## 二、Self-Attention 计算过程（以单头为例）

### 2.1 线性变换得到 Q、K、V

设输入序列表示为：

```text
X ∈ R^(seq_len × d_model)    # 例如 (10, 512)
```

通过三组可学习的参数矩阵得到查询（Q）、键（K）、值（V）：

```text
Q = X · W_Q    # (10, 512) × (512, d_k) = (10, d_k)
K = X · W_K    # (10, 512) × (512, d_k) = (10, d_k)
V = X · W_V    # (10, 512) × (512, d_v) = (10, d_v)
```

### 2.2 计算注意力分数

```text
Scores = Q · K^T / sqrt(d_k)
       = (seq_len, d_k) × (d_k, seq_len)
       = (seq_len, seq_len)
```

这个 `seq_len × seq_len` 的矩阵表示：  
第 \(i\) 个 token 对第 \(j\) 个 token 的「关注程度」。

### 2.3 Softmax 归一化

对每一行做 Softmax：

```text
Attention_Weights = softmax(Scores)   # 每一行和为 1
```

可以理解为：对于当前词，如何在所有词上分配注意力权重。

### 2.4 加权求和值向量

```text
Output = Attention_Weights · V
       = (seq_len, seq_len) × (seq_len, d_v)
       = (seq_len, d_v)
```

**直觉例子：**
- 句子「我 爱 中国」中，在计算「爱」的新表示时：
  - 可能对「我」「爱」「中国」的权重是 `[0.2, 0.3, 0.5]`；
  - 新表示 = `0.2·V(我) + 0.3·V(爱) + 0.5·V(中国)`。

这样，每个词的表示都「融合了上下文信息」。

---

## 三、为什么需要 Multi-Head Attention？

### 3.1 单头的局限

- 只有一个注意力头，只能学习「一种」注意力模式：
  - 例如只抓到语法关系，而忽略语义关系；
  - 或只关注局部信息，忽略远距离依赖。

### 3.2 多头的优势

```text
假设 d_model = 512, 头数 h = 8

每个头的维度: d_k = 512 / 8 = 64
```

- 不同的头可以学习不同的模式（类比 CNN 中不同卷积核）：
  - Head1：主谓关系 —— 「我 爱 中国」中「我」关注「爱」；
  - Head2：修饰关系 —— 「美丽的 中国」中「中国」关注「美丽的」；
  - Head3：长距离依赖 —— 句首词关注句尾词；
  - 其他头学习更多句法 / 语义结构。
- 计算量与单头类似，但表达能力明显增强。

### 3.3 Multi-Head 数学形式

```text
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

MultiHead(Q, K, V) = Concat(head_1, …, head_h) · W^O
```

其中：
- \(W_i^Q, W_i^K, W_i^V\) 是第 i 个头的线性变换矩阵；
- 最后通过 \(W^O\) 将拼接结果映射回 \(d_model\) 维。

---

## 四、位置编码（Position Encoding）

### 4.1 为什么需要位置编码？

- 自注意力本身是 **排列不变（permutation invariant）** 的：
  - 如果我们打乱输入顺序，单纯的 Attention 输出不变；
  - 但自然语言有严格的序列顺序，如「我爱你」和「你爱我」含义完全不同。
- 因此必须显式注入位置信息，帮助模型区分「谁在前、谁在后」。

### 4.2 正弦位置编码（Sinusoidal PE）

原论文中的公式：

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**实现要点：**
- `pos` 为位置（0, 1, 2, …）；
- `i` 为维度索引（0, 1, …, d_model/2 - 1）；
- 偶数维用 `sin`，奇数维用 `cos`。

**优点：**
- 不需要训练参数；
- 可以自然外推到更长序列；
- 通过三角恒等式可以编码相对位置信息。

### 4.3 可学习位置编码（Learnable PE）

```python
position_embeddings = nn.Embedding(max_position, d_model)
pos_ids = torch.arange(seq_len)  # [0, 1, ..., seq_len-1]
pos_emb = position_embeddings(pos_ids)
```

**特点：**
- 需要训练；
- 对特定任务更灵活；
- 但无法直接外推到训练时未见过的长度。

### 4.4 相对位置编码与新方法

- 相对位置编码（如 Transformer-XL、T5、DeBERTa）：
  - 不编码绝对位置，只编码 token 之间的相对距离。
- 新方法：
  - RoPE（旋转位置编码）、ALiBi 等，更适合长序列建模（LLaMA 等模型使用）。

---

## 五、与 RNN/CNN 的对比

### 5.1 相比 RNN

- **并行性更好**：
  - RNN 逐步处理（t=1,2,...），难以并行；
  - Transformer 对同一层的所有位置可以并行计算。
- **长距离依赖建模能力强**：
  - RNN 依靠链式传递，存在梯度消失 / 爆炸问题；
  - Self-Attention 任意两位置间可直接建立连接。

### 5.2 相比 CNN

- CNN 通过局部卷积 + 多层堆叠扩展感受野；
- Self-Attention 一层就可以「看全局」，建模全局依赖更自然；
- 但 CNN 在图像等局部模式强的场景仍然非常有效。

---

## 六、面试高频考点速查

> **这个小节只列关键点，具体口述模板请看 `面试答题_Transformer.md`。**

- **结构整体：**
  - 能画出 Encoder-Decoder 结构图，并指出每个模块的作用；
  - 说明 Encoder/Decoder 各自包含哪些子层，以及残差 + LayerNorm 的位置。
- **Self-Attention：**
  - 能写出标准公式：`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`；
  - 能解释 Q/K/V 的含义及形状；
  - 能说明为什么要除以 `sqrt(d_k)`。
- **Multi-Head：**
  - 说明多头的数学形式和设计动机；
  - 会用「多个卷积核 / 多个注意力视角」的类比回答。
- **位置编码：**
  - 解释为什么 Attention 需要位置编码；
  - 说出正弦编码与可学习编码的优缺点；
  - 知道 RoPE、ALiBi 是为长文本提出的改进。
- **对比：**
  - 简要对比 Transformer 与 RNN 的并行性、长依赖建模能力；
  - 知道 BERT、GPT 都是基于 Transformer 的变体。

---

## 七、配套文档

- 📚 **上层大纲**：`理论.md` 中的「一、Transformer 原理」章节  
- 📝 **面试答题示例**：`面试答题_Transformer.md`  
- 📘 **相关文档**：`BERT与GPT的架构关系详解.md`（理解 BERT/GPT 与 Transformer 的关系）


