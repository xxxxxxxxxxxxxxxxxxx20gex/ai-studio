"""
BERT (Bidirectional Encoder Representations from Transformers)
只使用 Transformer 的 Encoder 部分

架构特点：
- 只有Encoder（没有Decoder）
- 双向注意力（可以看到全文）
- 预训练任务：MLM（完形填空）+ NSP（句子关系）
- 擅长理解类任务：分类、NER、QA等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 复用 Transformer 的组件
# ============================================
class MultiHeadAttention(nn.Module):
    """多头注意力（和Transformer相同）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分成多头
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        return output


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff=3072):  # BERT-base用3072
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))  # BERT使用GELU激活


# ============================================
# BERT Encoder Layer
# ============================================
class BERTEncoderLayer(nn.Module):
    """
    BERT的Encoder层
    
    和Transformer的Encoder层相同：
    - Self-Attention（双向）
    - Feed Forward Network
    - Layer Normalization + 残差连接
    """
    def __init__(self, d_model, num_heads, d_ff=3072, dropout=0.1):
        super().__init__()
        
        # ⭐ Multi-Head Self-Attention（双向）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward Network
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: (batch_size, seq_len, d_model)
            mask: attention mask（可选）
        """
        # 1. ⭐ Self-Attention（Q=K=V，双向）
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


# ============================================
# BERT Embeddings
# ============================================
class BERTEmbeddings(nn.Module):
    """
    BERT的Embedding层
    
    包含三部分（与GPT不同）：
    1. Token Embedding: 词嵌入
    2. Position Embedding: 位置嵌入（可学习，不是sin/cos）
    3. Segment Embedding: 句子嵌入（区分句子A和句子B）
    """
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        
        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. ⭐ Position Embedding（可学习，不同于Transformer的sin/cos）
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # 3. ⭐ Segment Embedding（用于区分两个句子，NSP任务需要）
        self.segment_embedding = nn.Embedding(2, d_model)  # 0: 句子A, 1: 句子B
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, segment_ids=None):
        """
        参数:
            input_ids: token ids (batch_size, seq_len)
            segment_ids: 句子id (batch_size, seq_len), 0或1
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. Token Embedding
        token_embed = self.token_embedding(input_ids)
        
        # 2. Position Embedding
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embed = self.position_embedding(position_ids)
        
        # 3. Segment Embedding
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        segment_embed = self.segment_embedding(segment_ids)
        
        # ⭐ 三者相加（BERT的特色）
        embeddings = token_embed + position_embed + segment_embed
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# ============================================
# BERT 模型
# ============================================
class BERT(nn.Module):
    """
    ⭐⭐⭐ BERT = 只有Encoder，没有Decoder
    
    架构特点:
    1. 只使用Transformer的Encoder部分
    2. 双向注意力（可以看到全文）
    3. 三种Embedding（Token + Position + Segment）
    
    预训练任务:
    1. MLM (Masked Language Model): 完形填空
       例如: "我爱[MASK]" → 预测"中国"
    2. NSP (Next Sentence Prediction): 判断两句话是否连续
    
    适用任务:
    - 文本分类（情感分析、主题分类）
    - 命名实体识别（NER）
    - 问答系统（抽取式QA）
    - 语义相似度
    """
    def __init__(
        self,
        vocab_size,        # 词汇表大小
        d_model=768,       # BERT-base: 768, BERT-large: 1024
        num_layers=12,     # BERT-base: 12, BERT-large: 24
        num_heads=12,      # BERT-base: 12, BERT-large: 16
        d_ff=3072,         # BERT-base: 3072, BERT-large: 4096
        max_len=512,       # 最大序列长度
        dropout=0.1
    ):
        super().__init__()
        
        # ⭐ BERT特殊的Embedding层
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout)
        
        # ⭐ N个Encoder层（和Transformer的Encoder相同）
        self.encoder_layers = nn.ModuleList([
            BERTEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # ⭐ 预训练任务的输出层
        # 1. MLM任务：预测被mask的词
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # 2. NSP任务：判断两个句子是否连续
        self.nsp_head = nn.Linear(d_model, 2)  # 二分类：是/否
        
        # ⭐ 下游任务的输出层（微调时使用）
        self.classifier = None  # 根据具体任务设置
        
    def forward(self, input_ids, segment_ids=None, mask=None):
        """
        参数:
            input_ids: token ids (batch_size, seq_len)
            segment_ids: 句子id (batch_size, seq_len)
            mask: attention mask
        
        返回:
            sequence_output: 每个token的表示 (batch_size, seq_len, d_model)
            pooled_output: [CLS] token的表示 (batch_size, d_model)
        """
        # 1. Embedding（Token + Position + Segment）
        x = self.embeddings(input_ids, segment_ids)
        
        # 2. ⭐ 通过N个Encoder层（双向注意力）
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 3. 输出
        sequence_output = x  # 每个token的表示
        pooled_output = x[:, 0, :]  # [CLS] token的表示（用于分类）
        
        return sequence_output, pooled_output
    
    def predict_mlm(self, input_ids, segment_ids=None):
        """
        MLM任务：预测被mask的词
        
        例如:
            输入: "我爱[MASK]" → [101, 2769, 4263, 103, 102]
            输出: 词汇表的概率分布
        """
        sequence_output, _ = self.forward(input_ids, segment_ids)
        # 对每个位置预测词汇
        mlm_logits = self.mlm_head(sequence_output)
        return mlm_logits
    
    def predict_nsp(self, input_ids, segment_ids):
        """
        NSP任务：判断两个句子是否连续
        
        例如:
            输入: "[CLS] 我爱中国 [SEP] 中国很美 [SEP]"
            segment_ids: [0, 0, 0, 0, 0, 1, 1, 1, 1]
            输出: [是连续的, 不是连续的] 的概率
        """
        _, pooled_output = self.forward(input_ids, segment_ids)
        # 用[CLS]的表示做二分类
        nsp_logits = self.nsp_head(pooled_output)
        return nsp_logits
    
    def predict_classification(self, input_ids, num_classes):
        """
        下游任务：文本分类
        
        例如：情感分析（正面/负面）
        """
        if self.classifier is None or self.classifier.out_features != num_classes:
            self.classifier = nn.Linear(self.embeddings.token_embedding.embedding_dim, num_classes)
        
        _, pooled_output = self.forward(input_ids)
        logits = self.classifier(pooled_output)
        return logits


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # BERT-base 参数
    vocab_size = 30522  # BERT的词汇表大小
    d_model = 768
    num_layers = 12
    num_heads = 12
    
    # 创建BERT模型
    model = BERT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    print("=" * 60)
    print("BERT 模型示例")
    print("=" * 60)
    
    # ============ 示例1: MLM任务（完形填空）============
    print("\n【示例1: MLM任务 - 完形填空】")
    print("输入: '我爱[MASK]'")
    
    batch_size = 2
    seq_len = 8
    
    # 输入token ids（假设103是[MASK]的id）
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # MLM预测
    mlm_logits = model.predict_mlm(input_ids)
    print(f"输出形状: {mlm_logits.shape}")  # (2, 8, 30522)
    print(f"解释: 对每个位置预测{vocab_size}个词的概率")
    
    # ============ 示例2: NSP任务（句子关系判断）============
    print("\n【示例2: NSP任务 - 判断两句话是否连续】")
    print("输入: '[CLS] 句子A [SEP] 句子B [SEP]'")
    
    # segment_ids标记句子A和句子B
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 0, 0, 1, 1, 1, 1]])
    
    nsp_logits = model.predict_nsp(input_ids, segment_ids)
    print(f"输出形状: {nsp_logits.shape}")  # (2, 2)
    print(f"解释: [是连续的概率, 不连续的概率]")
    
    # ============ 示例3: 文本分类（下游任务）============
    print("\n【示例3: 文本分类 - 情感分析】")
    print("输入: '[CLS] 这部电影很好看 [SEP]'")
    
    num_classes = 2  # 正面/负面
    classification_logits = model.predict_classification(input_ids, num_classes)
    print(f"输出形状: {classification_logits.shape}")  # (2, 2)
    print(f"解释: [负面概率, 正面概率]")
    
    # ============ 关键对比 ============
    print("\n" + "=" * 60)
    print("⭐⭐⭐ BERT vs Transformer 的关键区别:")
    print("=" * 60)
    print("1. 架构: BERT只有Encoder，Transformer有Encoder+Decoder")
    print("2. 注意力: BERT是双向的（可以看全文），适合理解任务")
    print("3. Embedding: BERT有3种（Token+Position+Segment）")
    print("4. 预训练: BERT用MLM+NSP，Transformer用翻译任务")
    print("5. 应用: BERT擅长分类、NER等理解任务")
    print("   Transformer擅长翻译等Seq2Seq任务")
    
    print("\n⭐ BERT的核心创新:")
    print("- 双向预训练: 通过[MASK]学习上下文")
    print("- 统一架构: 预训练+微调范式")
    print("- 简化设计: 去掉Decoder，更轻量")

