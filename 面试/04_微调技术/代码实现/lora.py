"""
LoRA (Low-Rank Adaptation) 低秩适配微调技术

【核心概念】
LoRA是什么？
- 一种参数高效的微调方法（PEFT）
- 冻结预训练权重，只训练低秩分解矩阵
- 大幅减少可训练参数量（通常<1%）

【核心原理】
假设原始权重矩阵 W ∈ R^(d×k)
LoRA将权重更新分解为两个低秩矩阵：
    W' = W + ΔW = W + B*A
其中：
    - W: 预训练权重（冻结，不更新）
    - A ∈ R^(r×k): 下投影矩阵（可训练）
    - B ∈ R^(d×r): 上投影矩阵（可训练）
    - r: 秩（rank），通常 r << min(d, k)

【参数量对比】
原始参数量: d × k
LoRA参数量: d × r + r × k = r × (d + k)
当r=8, d=k=1024时：
    原始: 1,048,576 参数
    LoRA: 16,384 参数（减少98.4%）
"""

import torch
import torch.nn as nn
import math


# ============================================
# 1. LoRA核心层实现
# ============================================

class LoRALayer(nn.Module):
    """
    LoRA层的核心实现
    
    在原始线性层基础上添加低秩分解的旁路
    
    参数：
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: 低秩分解的秩（越小参数越少，但表达能力越弱）
        alpha: 缩放因子，控制LoRA的影响程度
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA的两个可训练矩阵
        # A: (rank, in_features) - 下投影，使用高斯初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # B: (out_features, rank) - 上投影，初始化为0（保证初始ΔW=0）
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 缩放因子（归一化，使得不同rank的效果相对稳定）
        self.scaling = self.alpha / self.rank
        
        # 初始化A矩阵（使用Kaiming初始化）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        """
        前向传播：计算 ΔW * x = B * A * x
        
        参数：
            x: 输入特征，shape=(batch_size, seq_len, in_features)
        
        返回：
            LoRA的输出，shape=(batch_size, seq_len, out_features)
        """
        # 计算流程：x -> A -> B -> output
        # (batch, seq, in_features) @ (in_features, rank) = (batch, seq, rank)
        h = x @ self.lora_A.T
        # (batch, seq, rank) @ (rank, out_features) = (batch, seq, out_features)
        output = h @ self.lora_B.T
        # 应用缩放
        return output * self.scaling


# ============================================
# 2. 带LoRA的线性层
# ============================================

class LinearWithLoRA(nn.Module):
    """
    集成LoRA的线性层
    
    计算公式：output = (W + α/r * B*A) * x = W*x + LoRA(x)
    
    参数：
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA的秩
        alpha: LoRA的缩放因子
        use_lora: 是否启用LoRA（方便对比实验）
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, use_lora=True):
        super(LinearWithLoRA, self).__init__()
        
        # 原始的线性层（预训练权重）
        self.linear = nn.Linear(in_features, out_features)
        
        # 冻结原始权重
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA层
        self.use_lora = use_lora
        if use_lora:
            self.lora = LoRALayer(in_features, out_features, rank, alpha)
    
    def forward(self, x):
        """
        前向传播：原始输出 + LoRA输出
        """
        # 基础输出（使用冻结的预训练权重）
        output = self.linear(x)
        
        # 加上LoRA的调整
        if self.use_lora:
            output = output + self.lora(x)
        
        return output
    
    def merge_weights(self):
        """
        【推理优化】合并LoRA权重到原始权重
        
        推理时可以将 W' = W + B*A 合并，避免额外计算
        """
        if self.use_lora:
            # 计算 ΔW = α/r * B * A
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            # 更新原始权重 W' = W + ΔW
            self.linear.weight.data += delta_w
            # 删除LoRA层以节省内存
            del self.lora
            self.use_lora = False


# ============================================
# 3. 带LoRA的Transformer层（简化版）
# ============================================

class TransformerBlockWithLoRA(nn.Module):
    """
    集成LoRA的Transformer块（简化版）
    
    在自注意力机制的Q、K、V投影层应用LoRA
    """
    def __init__(self, hidden_dim=512, num_heads=8, lora_rank=8, lora_alpha=16):
        super(TransformerBlockWithLoRA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Q、K、V投影层（带LoRA）
        self.q_proj = LinearWithLoRA(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.k_proj = LinearWithLoRA(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        self.v_proj = LinearWithLoRA(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        
        # 输出投影层（也可以应用LoRA）
        self.out_proj = LinearWithLoRA(hidden_dim, hidden_dim, lora_rank, lora_alpha)
        
        # LayerNorm和前馈网络（简化，不使用LoRA）
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        """
        参数：
            x: shape=(batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 自注意力（带LoRA）
        residual = x
        x = self.norm1(x)
        
        # 计算Q、K、V（通过带LoRA的投影层）
        q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算（简化版）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.out_proj(attn_output)
        x = x + residual
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


# ============================================
# 4. 参数量统计工具
# ============================================

def count_parameters(model):
    """
    统计模型的参数量
    
    返回：
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ============================================
# 5. 示例与对比
# ============================================

def lora_comparison_example():
    """对比全量微调和LoRA微调的参数量"""
    print("=" * 60)
    print("LoRA参数量对比示例")
    print("=" * 60)
    
    hidden_dim = 768  # BERT-base的隐藏层维度
    
    # 1. 原始线性层（全量微调）
    linear_full = nn.Linear(hidden_dim, hidden_dim)
    total_full, trainable_full = count_parameters(linear_full)
    
    print(f"\n【全量微调】")
    print(f"输入维度: {hidden_dim}, 输出维度: {hidden_dim}")
    print(f"总参数量: {total_full:,}")
    print(f"可训练参数: {trainable_full:,}")
    
    # 2. 带LoRA的线性层
    ranks = [4, 8, 16, 32]
    print(f"\n【LoRA微调】")
    for rank in ranks:
        linear_lora = LinearWithLoRA(hidden_dim, hidden_dim, rank=rank)
        total_lora, trainable_lora = count_parameters(linear_lora)
        
        reduction = (1 - trainable_lora / trainable_full) * 100
        print(f"\nrank={rank}:")
        print(f"  总参数量: {total_lora:,}")
        print(f"  可训练参数: {trainable_lora:,}")
        print(f"  参数减少: {reduction:.2f}%")
    
    print()


def lora_forward_example():
    """LoRA前向传播示例"""
    print("=" * 60)
    print("LoRA前向传播示例")
    print("=" * 60)
    
    # 创建模拟数据
    batch_size, seq_len, hidden_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建带LoRA的线性层
    layer = LinearWithLoRA(hidden_dim, hidden_dim, rank=8, alpha=16)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output = layer(x)
    print(f"输出形状: {output.shape}")
    
    # 统计参数
    total, trainable = count_parameters(layer)
    print(f"\n总参数量: {total:,}")
    print(f"可训练参数: {trainable:,} (只有LoRA的A和B)")
    
    print()


def lora_transformer_example():
    """带LoRA的Transformer示例"""
    print("=" * 60)
    print("带LoRA的Transformer块示例")
    print("=" * 60)
    
    # 创建模拟数据
    batch_size, seq_len, hidden_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建Transformer块
    transformer = TransformerBlockWithLoRA(
        hidden_dim=hidden_dim,
        num_heads=8,
        lora_rank=8,
        lora_alpha=16
    )
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output = transformer(x)
    print(f"输出形状: {output.shape}")
    
    # 统计参数
    total, trainable = count_parameters(transformer)
    print(f"\n总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"可训练参数占比: {trainable/total*100:.2f}%")
    
    print("\n【LoRA应用位置】")
    print("✓ Q投影层")
    print("✓ K投影层")
    print("✓ V投影层")
    print("✓ 输出投影层")
    print("✗ LayerNorm（不需要微调）")
    print("✗ FFN（可选，这里未使用LoRA）")
    
    print()


def lora_merge_example():
    """LoRA权重合并示例（推理优化）"""
    print("=" * 60)
    print("LoRA权重合并示例（推理优化）")
    print("=" * 60)
    
    # 创建测试数据
    x = torch.randn(1, 5, 256)
    
    # 创建带LoRA的层
    layer = LinearWithLoRA(256, 256, rank=8)
    
    # 合并前
    print("\n【合并前】")
    output_before = layer(x)
    total_before, trainable_before = count_parameters(layer)
    print(f"总参数量: {total_before:,}")
    print(f"可训练参数: {trainable_before:,}")
    
    # 合并权重
    layer.merge_weights()
    
    # 合并后
    print("\n【合并后】")
    output_after = layer(x)
    total_after, trainable_after = count_parameters(layer)
    print(f"总参数量: {total_after:,}")
    print(f"可训练参数: {trainable_after:,}")
    
    # 验证输出一致性
    diff = torch.abs(output_before - output_after).max().item()
    print(f"\n输出差异: {diff:.6f} (应该接近0)")
    
    print("\n【合并的好处】")
    print("✓ 推理时无额外计算开销")
    print("✓ 可以删除LoRA层，节省内存")
    print("✓ 与原始模型架构完全兼容")
    
    print()


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LoRA (Low-Rank Adaptation) 完整示例")
    print("="*60 + "\n")
    
    # 1. 参数量对比
    lora_comparison_example()
    
    # 2. 前向传播
    lora_forward_example()
    
    # 3. Transformer应用
    lora_transformer_example()
    
    # 4. 权重合并
    lora_merge_example()
    
    print("=" * 60)
    print("LoRA示例运行完毕！")
    print("\n【核心要点总结】")
    print("1. LoRA通过低秩分解减少可训练参数（通常<1%）")
    print("2. 只训练A、B矩阵，冻结原始权重W")
    print("3. 推理时可合并权重，无额外开销")
    print("4. 适用于大模型微调（LLaMA、GPT等）")
    print("=" * 60)

