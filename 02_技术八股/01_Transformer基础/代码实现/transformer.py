"""
Transformer 完整架构实现（简化版）
用于机器翻译等 Seq2Seq 任务

架构特点：
- Encoder + Decoder 完整结构
- Encoder: 双向注意力（可以看到全文）
- Decoder: 单向注意力 + Cross-Attention（连接Encoder和Decoder）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Multi-Head Attention（多头注意力）
# ============================================
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q、K、V 的线性变换
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 输出的线性变换
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: Query (batch_size, seq_len_q, d_model)
            K: Key   (batch_size, seq_len_k, d_model)
            V: Value (batch_size, seq_len_v, d_model)
            mask: 注意力mask (可选)
        """
        batch_size = Q.size(0)
        
        # 1. 线性变换并分成多头
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. 加权求和
        # output: (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attention_weights, V)
        
        # 6. 拼接多头并通过输出层
        # (batch_size, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        return output


# ============================================
# 2. Position-wise Feed Forward Network
# ============================================
class FeedForward(nn.Module):
    """
    前馈网络：两层全连接 + ReLU
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ============================================
# 3. Positional Encoding（位置编码）
# ============================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        # 偶数维用sin，奇数维用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pe[:, :seq_len, :]
        return x


# ============================================
# 4. Encoder Layer（编码器层）
# ============================================
class EncoderLayer(nn.Module):
    """
    编码器层 = Self-Attention + Feed Forward
    特点：双向注意力（可以看到全文）
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
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
            mask: padding mask（可选）
        """
        # 1. Self-Attention + 残差连接
        # ⭐ Q=K=V=x，双向注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed Forward + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


# ============================================
# 5. Decoder Layer（解码器层）
# ============================================
class DecoderLayer(nn.Module):
    """
    解码器层 = Masked Self-Attention + Cross-Attention + Feed Forward
    特点：
    1. Masked Self-Attention（单向，只看前文）
    2. Cross-Attention（从Encoder获取信息）
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # ⭐ Cross-Attention（连接Encoder和Decoder）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward Network
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        参数:
            x: Decoder输入 (batch_size, tgt_len, d_model)
            encoder_output: Encoder输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列的因果mask（下三角）
            src_mask: 源序列的padding mask
        """
        # 1. Masked Self-Attention（只看前文）
        # ⭐ 使用因果mask，保证只能看到前面的词
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. ⭐⭐⭐ Cross-Attention（从Encoder获取信息）
        # Q来自Decoder，K和V来自Encoder
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


# ============================================
# 6. Encoder（编码器）
# ============================================
class Encoder(nn.Module):
    """
    完整的Encoder = N个EncoderLayer堆叠
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # N个Encoder层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入token ids (batch_size, seq_len)
        """
        # 1. Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. 通过N个Encoder层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ============================================
# 7. Decoder（解码器）
# ============================================
class Decoder(nn.Module):
    """
    完整的Decoder = N个DecoderLayer堆叠
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # N个Decoder层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        参数:
            x: 目标序列 (batch_size, tgt_len)
            encoder_output: Encoder的输出
        """
        # 1. Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. 通过N个Decoder层
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return x


# ============================================
# 8. 完整的Transformer模型
# ============================================
class Transformer(nn.Module):
    """
    ⭐⭐⭐ 完整的Transformer = Encoder + Decoder
    
    用于Seq2Seq任务（如机器翻译）
    
    架构特点:
    - Encoder: 双向理解源序列
    - Decoder: 单向生成目标序列
    - Cross-Attention: 连接Encoder和Decoder
    
    示例:
        翻译任务:
        输入(英语): "I love AI"
        Encoder理解英语 → Decoder生成中文: "我爱人工智能"
    """
    def __init__(
        self, 
        src_vocab_size,    # 源语言词汇表大小
        tgt_vocab_size,    # 目标语言词汇表大小
        d_model=512,       # 模型维度
        num_layers=6,      # Encoder/Decoder层数
        num_heads=8,       # 注意力头数
        d_ff=2048,         # FFN维度
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # ⭐ Encoder（编码器）
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        
        # ⭐ Decoder（解码器）
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        
        # 输出层（生成词汇表的概率分布）
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        参数:
            src: 源序列 (batch_size, src_len)
            tgt: 目标序列 (batch_size, tgt_len)
        
        返回:
            输出概率分布 (batch_size, tgt_len, tgt_vocab_size)
        """
        # 1. ⭐ Encoder处理源序列
        encoder_output = self.encoder(src, src_mask)
        
        # 2. ⭐ Decoder生成目标序列（同时参考Encoder输出）
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # 3. 输出层
        output = self.output_layer(decoder_output)
        
        return output
    
    @staticmethod
    def create_causal_mask(seq_len):
        """
        创建因果mask（下三角矩阵）
        用于Decoder的Masked Self-Attention
        
        例如 seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
         
        保证第i个位置只能看到前i个位置的信息
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 模型参数
    src_vocab_size = 10000  # 源语言词汇表大小
    tgt_vocab_size = 8000   # 目标语言词汇表大小
    d_model = 512
    num_layers = 6
    num_heads = 8
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 示例输入
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # 源序列和目标序列（token ids）
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 创建因果mask（Decoder需要）
    tgt_mask = Transformer.create_causal_mask(tgt_len)
    
    # 前向传播
    output = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"源序列形状: {src.shape}")          # (2, 10)
    print(f"目标序列形状: {tgt.shape}")        # (2, 8)
    print(f"输出形状: {output.shape}")          # (2, 8, 8000)
    print(f"\n输出解释: (batch_size={batch_size}, tgt_len={tgt_len}, tgt_vocab_size={tgt_vocab_size})")
    print(f"对于每个位置，输出{tgt_vocab_size}个词的概率分布")
    
    print("\n⭐⭐⭐ Transformer架构特点:")
    print("1. 完整的Encoder-Decoder结构")
    print("2. Encoder使用双向注意力（可以看到全文）")
    print("3. Decoder使用单向注意力（只能看前文）+ Cross-Attention")
    print("4. 适用于Seq2Seq任务（翻译、摘要等）")

