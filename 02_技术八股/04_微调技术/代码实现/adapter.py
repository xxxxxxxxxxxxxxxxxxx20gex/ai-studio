"""
Adapter（适配器）微调技术

【核心概念】
Adapter是什么？
- 一种参数高效的微调方法（PEFT）
- 在预训练模型的层之间插入小型"适配器"模块
- 只训练适配器参数，冻结原始模型

【核心原理】
Adapter模块结构（瓶颈结构）：
    输入(d维) -> 下投影(d->m) -> 非线性激活 -> 上投影(m->d) -> 残差连接 -> 输出(d维)
    
其中：
    - d: 原始隐藏层维度（如768）
    - m: 瓶颈维度（如64），m << d
    - 残差连接：保证初始时Adapter不影响原始输出

【参数量】
每个Adapter模块参数量: 2 × d × m + bias
典型配置：d=768, m=64
    参数量: 2 × 768 × 64 ≈ 98K （相比原始层的数百万参数大幅减少）
"""

import torch
import torch.nn as nn


# ============================================
# 1. Adapter核心模块
# ============================================

class AdapterModule(nn.Module):
    """
    Adapter模块的核心实现（瓶颈架构）
    
    架构：Down-Project -> Activation -> Up-Project -> Residual
    
    参数：
        input_dim: 输入维度（通常是Transformer的隐藏层维度）
        bottleneck_dim: 瓶颈维度（越小参数越少，通常为input_dim的1/8到1/16）
        activation: 激活函数（默认ReLU）
    """
    def __init__(self, input_dim, bottleneck_dim=64, activation='relu'):
        super(AdapterModule, self).__init__()
        
        # 下投影层：降维到瓶颈维度
        # (input_dim) -> (bottleneck_dim)
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # 非线性激活
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 上投影层：恢复到原始维度
        # (bottleneck_dim) -> (input_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # 初始化：使上投影初始输出接近0，保证初始时Adapter影响很小
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        """
        前向传播：bottleneck变换 + 残差连接
        
        参数：
            x: 输入特征，shape=(batch_size, seq_len, input_dim)
        
        返回：
            输出特征，shape与输入相同
        """
        # 保存输入用于残差连接
        residual = x
        
        # Adapter变换
        # 步骤1: 降维
        x = self.down_project(x)  # (batch, seq, bottleneck_dim)
        
        # 步骤2: 非线性激活
        x = self.activation(x)
        
        # 步骤3: 升维回原始维度
        x = self.up_project(x)  # (batch, seq, input_dim)
        
        # 步骤4: 残差连接（关键！保证初始时输出≈输入）
        output = x + residual
        
        return output


# ============================================
# 2. 带Adapter的Transformer层
# ============================================

class TransformerLayerWithAdapter(nn.Module):
    """
    集成Adapter的Transformer层
    
    Adapter插入位置（两处）：
    1. 自注意力之后、残差连接之前
    2. 前馈网络之后、残差连接之前
    
    架构：
        x -> LayerNorm -> Self-Attention -> [Adapter] -> Residual
          -> LayerNorm -> Feed-Forward  -> [Adapter] -> Residual
    """
    def __init__(self, hidden_dim=768, num_heads=12, adapter_dim=64, use_adapter=True):
        super(TransformerLayerWithAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_adapter = use_adapter
        
        # 自注意力机制（简化实现）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Adapter模块（两个）
        if use_adapter:
            # Adapter 1: 插入在attention之后
            self.adapter_attn = AdapterModule(hidden_dim, adapter_dim)
            # Adapter 2: 插入在FFN之后
            self.adapter_ffn = AdapterModule(hidden_dim, adapter_dim)
        
        # 冻结原始Transformer参数（只训练Adapter）
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
    
    def forward(self, x, mask=None):
        """
        参数：
            x: 输入特征，shape=(batch_size, seq_len, hidden_dim)
            mask: 注意力掩码（可选）
        """
        # ===== 自注意力块 =====
        residual = x
        x = self.norm1(x)
        
        # 自注意力
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # 【插入点1】Adapter after attention
        if self.use_adapter:
            attn_output = self.adapter_attn(attn_output)
        
        # 残差连接
        x = residual + attn_output
        
        # ===== 前馈网络块 =====
        residual = x
        x = self.norm2(x)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        
        # 【插入点2】Adapter after FFN
        if self.use_adapter:
            ffn_output = self.adapter_ffn(ffn_output)
        
        # 残差连接
        x = residual + ffn_output
        
        return x


# ============================================
# 3. 完整的带Adapter的模型
# ============================================

class BERTWithAdapter(nn.Module):
    """
    带Adapter的BERT模型（简化版）
    
    在每个Transformer层插入Adapter模块
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        adapter_dim=64,
        use_adapter=True
    ):
        super(BERTWithAdapter, self).__init__()
        
        # Token嵌入
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置嵌入（简化：使用可学习的位置嵌入）
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        # 多层Transformer（每层都带Adapter）
        self.layers = nn.ModuleList([
            TransformerLayerWithAdapter(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                adapter_dim=adapter_dim,
                use_adapter=use_adapter
            )
            for _ in range(num_layers)
        ])
        
        # 输出层（用于分类任务）
        self.classifier = nn.Linear(hidden_dim, 2)  # 二分类示例
        
        # 冻结嵌入层和分类器（可选）
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.position_embedding.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids):
        """
        参数：
            input_ids: token IDs，shape=(batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        x = self.embedding(input_ids)  # (batch, seq, hidden_dim)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        
        # 通过所有Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 取[CLS] token的输出用于分类（第一个token）
        cls_output = x[:, 0, :]  # (batch, hidden_dim)
        
        # 分类
        logits = self.classifier(cls_output)  # (batch, num_classes)
        
        return logits


# ============================================
# 4. 参数量统计
# ============================================

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ============================================
# 5. 示例与对比
# ============================================

def adapter_module_example():
    """Adapter模块基础示例"""
    print("=" * 60)
    print("Adapter模块基础示例")
    print("=" * 60)
    
    # 创建Adapter模块
    input_dim = 768
    bottleneck_dim = 64
    adapter = AdapterModule(input_dim, bottleneck_dim)
    
    # 模拟输入
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输入维度: {input_dim}")
    print(f"瓶颈维度: {bottleneck_dim}")
    
    # 前向传播
    output = adapter(x)
    print(f"输出形状: {output.shape}")
    
    # 参数量
    total, trainable = count_parameters(adapter)
    print(f"\nAdapter参数量: {total:,}")
    print(f"压缩比: {bottleneck_dim}/{input_dim} = 1/{input_dim//bottleneck_dim}")
    
    print()


def adapter_comparison_example():
    """对比全量微调和Adapter微调"""
    print("=" * 60)
    print("Adapter参数量对比（BERT-base配置）")
    print("=" * 60)
    
    hidden_dim = 768
    num_layers = 12
    
    # 1. 不使用Adapter（全量微调）
    model_full = BERTWithAdapter(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_adapter=False
    )
    # 解冻所有参数
    for param in model_full.parameters():
        param.requires_grad = True
    
    total_full, trainable_full = count_parameters(model_full)
    
    print(f"\n【全量微调】")
    print(f"总参数量: {total_full:,}")
    print(f"可训练参数: {trainable_full:,}")
    
    # 2. 使用Adapter
    adapter_dims = [32, 64, 128, 256]
    print(f"\n【Adapter微调】")
    
    for adapter_dim in adapter_dims:
        model_adapter = BERTWithAdapter(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            adapter_dim=adapter_dim,
            use_adapter=True
        )
        
        total_adapter, trainable_adapter = count_parameters(model_adapter)
        reduction = (1 - trainable_adapter / trainable_full) * 100
        
        print(f"\nAdapter维度={adapter_dim}:")
        print(f"  总参数量: {total_adapter:,}")
        print(f"  可训练参数: {trainable_adapter:,}")
        print(f"  参数减少: {reduction:.2f}%")
        print(f"  可训练比例: {trainable_adapter/total_adapter*100:.2f}%")
    
    print()


def adapter_forward_example():
    """Adapter前向传播示例"""
    print("=" * 60)
    print("带Adapter的Transformer层前向传播")
    print("=" * 60)
    
    # 创建模型
    layer = TransformerLayerWithAdapter(
        hidden_dim=512,
        num_heads=8,
        adapter_dim=64
    )
    
    # 模拟输入
    batch_size, seq_len, hidden_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output = layer(x)
    print(f"输出形状: {output.shape}")
    
    # 参数统计
    total, trainable = count_parameters(layer)
    print(f"\n总参数量: {total:,}")
    print(f"可训练参数: {trainable:,} (仅Adapter)")
    print(f"可训练比例: {trainable/total*100:.2f}%")
    
    print("\n【Adapter插入位置】")
    print("✓ 自注意力之后")
    print("✓ 前馈网络之后")
    print("✗ LayerNorm（不插入）")
    print("✗ 嵌入层（不插入）")
    
    print()


def adapter_initialization_test():
    """测试Adapter初始化（应该接近恒等映射）"""
    print("=" * 60)
    print("Adapter初始化测试（应该接近恒等映射）")
    print("=" * 60)
    
    # 创建Adapter
    adapter = AdapterModule(input_dim=768, bottleneck_dim=64)
    
    # 随机输入
    x = torch.randn(1, 10, 768)
    
    # 前向传播
    output = adapter(x)
    
    # 计算差异（初始化时应该很小）
    diff = torch.abs(output - x).mean().item()
    max_diff = torch.abs(output - x).max().item()
    
    print(f"\n输入与输出的平均差异: {diff:.6f}")
    print(f"输入与输出的最大差异: {max_diff:.6f}")
    
    print("\n【初始化策略】")
    print("✓ 上投影层权重初始化为0")
    print("✓ 上投影层偏置初始化为0")
    print("✓ 保证初始时 Adapter(x) ≈ x （恒等映射）")
    print("✓ 训练过程中逐渐学习任务相关的调整")
    
    print()


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Adapter（适配器）微调技术完整示例")
    print("="*60 + "\n")
    
    # 1. Adapter模块示例
    adapter_module_example()
    
    # 2. 参数量对比
    adapter_comparison_example()
    
    # 3. 前向传播
    adapter_forward_example()
    
    # 4. 初始化测试
    adapter_initialization_test()
    
    print("=" * 60)
    print("Adapter示例运行完毕！")
    print("\n【核心要点总结】")
    print("1. Adapter通过瓶颈结构减少参数（通常2-4%）")
    print("2. 插入在Transformer层的关键位置（attention后、FFN后）")
    print("3. 初始化为接近恒等映射，不破坏预训练知识")
    print("4. 相比LoRA，Adapter参数稍多但更稳定")
    print("=" * 60)

