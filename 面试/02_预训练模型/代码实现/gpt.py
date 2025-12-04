"""
GPT (Generative Pre-trained Transformer)
只使用 Transformer 的 Decoder 部分（去掉Cross-Attention）

架构特点：
- 只有Decoder（没有Encoder）
- 单向/因果注意力（只能看前文）
- 去掉了Cross-Attention（因为没有Encoder）
- 预训练任务：预测下一个词（Causal Language Modeling）
- 擅长生成类任务：文本生成、对话、代码生成等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 复用的基础组件
# ============================================
class MultiHeadAttention(nn.Module):
    """多头注意力"""
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
        
        # ⭐ 应用因果mask（只能看前文）
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
    def __init__(self, d_model, d_ff=3072):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))  # GPT使用GELU


class PositionalEncoding(nn.Module):
    """
    位置编码（可学习版本）
    GPT使用可学习的位置编码，不同于Transformer的sin/cos
    """
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        # ⭐ 可学习的位置编码（和BERT类似）
        self.position_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # 生成位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 添加位置编码
        pos_embed = self.position_embedding(positions)
        
        return x + pos_embed


# ============================================
# GPT Decoder Layer
# ============================================
class GPTDecoderLayer(nn.Module):
    """
    GPT的Decoder层
    
    ⭐⭐⭐ 关键特点：
    1. 只有Masked Self-Attention（单向，只看前文）
    2. 没有Cross-Attention（因为没有Encoder）
    3. 这是和Transformer Decoder的最大区别！
    
    对比Transformer的Decoder:
    - Transformer Decoder: Masked Self-Attn + Cross-Attn + FFN
    - GPT Decoder:         Masked Self-Attn + FFN  ← 去掉了Cross-Attn
    """
    def __init__(self, d_model, num_heads, d_ff=3072, dropout=0.1):
        super().__init__()
        
        # ⭐ Masked Multi-Head Self-Attention（因果注意力，只看前文）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # ⭐⭐⭐ 注意：没有Cross-Attention！
        # Transformer的Decoder有这一层，GPT没有
        
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
            mask: 因果mask（下三角矩阵）
        """
        # 1. ⭐ Masked Self-Attention（只能看前文）
        # Q=K=V=x，但使用因果mask
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


# ============================================
# GPT 模型
# ============================================
class GPT(nn.Module):
    """
    ⭐⭐⭐ GPT = 只有Decoder（去掉Cross-Attention）
    
    架构特点:
    1. 只使用Transformer的Decoder部分
    2. 单向/因果注意力（只能看前文，不能看未来）
    3. ⭐ 去掉了Cross-Attention（因为没有独立的Encoder）
    
    预训练任务:
    - CLM (Causal Language Modeling): 预测下一个词
      例如: "我爱自然语言" → 预测"处理"
    
    适用任务:
    - 文本生成（写文章、写代码）
    - 对话系统（ChatGPT）
    - 续写任务
    - 翻译（也能做）
    
    为什么去掉Cross-Attention？
    - Transformer: 翻译任务，有独立的源序列和目标序列
      → Decoder通过Cross-Attn从Encoder获取源序列信息
    - GPT: 生成任务，输入和输出是同一个序列
      → 只需要看之前生成的内容，不需要额外的源序列
      → Cross-Attention没用，直接去掉！
    """
    def __init__(
        self,
        vocab_size,        # 词汇表大小
        d_model=768,       # GPT-2 small: 768, medium: 1024, large: 1280, xl: 1600
        num_layers=12,     # GPT-2 small: 12, medium: 24, large: 36, xl: 48
        num_heads=12,      # GPT-2 small: 12, medium: 16, large: 20, xl: 25
        d_ff=3072,         # 通常是 d_model * 4
        max_len=1024,      # GPT-2: 1024, GPT-3: 2048
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. ⭐ Position Embedding（可学习）
        self.position_embedding = PositionalEncoding(d_model, max_len)
        
        # 3. ⭐ N个Decoder层（去掉了Cross-Attention）
        self.decoder_layers = nn.ModuleList([
            GPTDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. 输出层（预测下一个词）
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Layer Norm（GPT在最后加一个）
        self.final_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        """
        参数:
            input_ids: token ids (batch_size, seq_len)
        
        返回:
            logits: 每个位置预测下一个词的概率分布 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. Token Embedding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # 2. Position Embedding
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        # 3. ⭐ 创建因果mask（下三角矩阵）
        # 保证第i个位置只能看到前i个位置
        causal_mask = self.create_causal_mask(seq_len, device=input_ids.device)
        
        # 4. ⭐ 通过N个Decoder层（只有Self-Attention，没有Cross-Attention）
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        
        # 5. Final Layer Norm
        x = self.final_norm(x)
        
        # 6. 输出层（预测词汇）
        logits = self.output_layer(x)
        
        return logits
    
    @staticmethod
    def create_causal_mask(seq_len, device='cpu'):
        """
        创建因果mask（下三角矩阵）
        
        例如 seq_len=4:
        [[1, 0, 0, 0],    第0个位置只能看到第0个
         [1, 1, 0, 0],    第1个位置只能看到第0-1个
         [1, 1, 1, 0],    第2个位置只能看到第0-2个
         [1, 1, 1, 1]]    第3个位置只能看到第0-3个
         
        这就是"因果"/"单向"的含义：只能看过去，不能看未来
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        ⭐ 自回归生成文本
        
        这是GPT的核心应用！
        
        参数:
            input_ids: 初始序列 (batch_size, seq_len)
            max_new_tokens: 最多生成多少个新token
            temperature: 温度系数（控制随机性）
            top_k: Top-K采样
        
        返回:
            生成的完整序列
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 1. 获取当前序列的logits
                logits = self.forward(input_ids)
                
                # 2. 只关注最后一个位置（预测下一个词）
                next_token_logits = logits[:, -1, :] / temperature
                
                # 3. Top-K采样（可选）
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -1e9)
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # 4. Softmax + 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 5. 拼接到序列后面
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # GPT-2 small 参数
    vocab_size = 50257  # GPT-2的词汇表大小
    d_model = 768
    num_layers = 12
    num_heads = 12
    
    # 创建GPT模型
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    print("=" * 60)
    print("GPT 模型示例")
    print("=" * 60)
    
    # ============ 示例1: 预测下一个词 ============
    print("\n【示例1: 预测下一个词（训练时）】")
    print("输入: '我 爱 自然 语言'")
    print("目标: 预测每个位置的下一个词")
    
    batch_size = 2
    seq_len = 10
    
    # 输入token ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    logits = model(input_ids)
    print(f"输出形状: {logits.shape}")  # (2, 10, 50257)
    print(f"解释: 对每个位置，预测下一个词的概率分布")
    
    # 具体来说：
    print("\n位置对应关系:")
    print("  位置0的输出 → 预测位置1的词")
    print("  位置1的输出 → 预测位置2的词")
    print("  ...")
    print("  位置9的输出 → 预测位置10的词（新生成）")
    
    # ============ 示例2: 因果mask可视化 ============
    print("\n【示例2: 因果Mask可视化】")
    mask = GPT.create_causal_mask(5)
    print("seq_len=5的因果mask:")
    print(mask.squeeze().tolist())  # 使用 tolist() 代替 numpy()
    print("\n解释:")
    print("  第0行: 位置0只能看到位置0")
    print("  第1行: 位置1只能看到位置0-1")
    print("  第2行: 位置2只能看到位置0-2")
    print("  ...")
    print("  这就是'单向'/'因果'的含义！")
    
    # ============ 示例3: 自回归生成 ============
    print("\n【示例3: 自回归生成文本】")
    print("输入: '今天天气'")
    print("生成过程:")
    
    # 初始序列（假设）
    initial_ids = torch.tensor([[1, 2, 3, 4]])  # "今天天气"
    
    print("  步骤1: '今天天气' → 预测 → '很'")
    print("  步骤2: '今天天气很' → 预测 → '好'")
    print("  步骤3: '今天天气很好' → 预测 → '，'")
    print("  ...")
    print("  逐词生成，每次只预测一个词")
    
    # 实际生成（演示）
    generated = model.generate(initial_ids, max_new_tokens=5)
    print(f"\n生成的序列形状: {generated.shape}")  # (1, 9) = 4(初始) + 5(新生成)
    
    # ============ 关键对比 ============
    print("\n" + "=" * 60)
    print("⭐⭐⭐ GPT vs Transformer Decoder 的关键区别:")
    print("=" * 60)
    print("1. Cross-Attention:")
    print("   Transformer Decoder: 有（从Encoder获取信息）")
    print("   GPT Decoder:         没有（因为没有Encoder）")
    print()
    print("2. 输入来源:")
    print("   Transformer: 目标序列 + Encoder输出")
    print("   GPT:         只有目标序列本身")
    print()
    print("3. 应用场景:")
    print("   Transformer: 翻译（源语言 → 目标语言）")
    print("   GPT:         生成（根据前文 → 续写）")
    
    print("\n" + "=" * 60)
    print("⭐⭐⭐ GPT vs BERT 的关键区别:")
    print("=" * 60)
    print("1. 架构:")
    print("   BERT: Encoder-only（双向）")
    print("   GPT:  Decoder-only（单向）")
    print()
    print("2. 注意力:")
    print("   BERT: 可以看到全文（适合理解）")
    print("   GPT:  只能看前文（适合生成）")
    print()
    print("3. 预训练:")
    print("   BERT: MLM（完形填空）")
    print("   GPT:  CLM（预测下一词）")
    print()
    print("4. 擅长任务:")
    print("   BERT: 分类、NER、QA")
    print("   GPT:  生成、对话、续写")
    
    print("\n⭐ GPT的核心优势:")
    print("- 自回归生成天然支持文本生成")
    print("- 简化架构（去掉Cross-Attention）")
    print("- 规模化后能力惊人（GPT-3/4）")
    print("- 统一范式（Few-shot Learning）")

