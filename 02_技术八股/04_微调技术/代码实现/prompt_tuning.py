"""
提示学习（Prompt Tuning）微调技术

【核心概念】
Prompt Tuning是什么？
- 一种极致参数高效的微调方法
- 只训练连续的"软提示"（soft prompt）向量
- 冻结整个预训练模型，只优化提示嵌入

【三种主要方法】
1. Prompt Tuning: 在输入前添加可学习的软提示token
2. P-Tuning v1: 使用LSTM/MLP编码提示，可插入任意位置
3. Prefix Tuning: 在每层的K、V前添加可学习的前缀

【核心原理】
传统方法：[CLS] I love this movie [SEP]
            ↓
Prompt Tuning: [P1] [P2] [P3] [P4] I love this movie [SEP]
                ↑    ↑    ↑    ↑
            可学习的软提示（连续向量，非离散token）

【参数量】
假设提示长度=20，隐藏层维度=768
参数量: 20 × 768 = 15,360（极少！）
相比BERT-base（110M参数）：只有0.014%
"""

import torch
import torch.nn as nn
import math


# ============================================
# 1. Prompt Tuning（基础版）
# ============================================

class PromptTuning(nn.Module):
    """
    Prompt Tuning: 最简单的提示学习方法
    
    在输入序列前添加可学习的软提示向量
    
    参数：
        num_prompts: 提示token的数量（如10、20等）
        hidden_dim: 隐藏层维度（需与预训练模型匹配）
        init_method: 初始化方法
            - 'random': 随机初始化
            - 'vocab': 从词表中随机采样token的嵌入
    """
    def __init__(self, num_prompts=20, hidden_dim=768, init_method='random'):
        super(PromptTuning, self).__init__()
        self.num_prompts = num_prompts
        self.hidden_dim = hidden_dim
        
        # 可学习的软提示嵌入
        # shape: (num_prompts, hidden_dim)
        self.soft_prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))
        
        # 初始化
        if init_method == 'random':
            # 使用标准正态分布初始化
            nn.init.normal_(self.soft_prompts, mean=0.0, std=0.02)
        elif init_method == 'vocab':
            # 实际应用中，可以从词表嵌入中采样
            # 这里简化为随机初始化
            nn.init.uniform_(self.soft_prompts, -0.5, 0.5)
    
    def forward(self, input_embeds):
        """
        将软提示拼接到输入嵌入前面
        
        参数：
            input_embeds: 输入的token嵌入，shape=(batch_size, seq_len, hidden_dim)
        
        返回：
            拼接后的嵌入，shape=(batch_size, num_prompts + seq_len, hidden_dim)
        """
        batch_size = input_embeds.size(0)
        
        # 扩展软提示到batch维度
        # (num_prompts, hidden_dim) -> (batch_size, num_prompts, hidden_dim)
        prompts_batch = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接：[软提示] + [原始输入]
        output_embeds = torch.cat([prompts_batch, input_embeds], dim=1)
        
        return output_embeds


# ============================================
# 2. P-Tuning v1（使用编码器）
# ============================================

class PTuningV1(nn.Module):
    """
    P-Tuning v1: 使用LSTM编码提示
    
    与Prompt Tuning的区别：
    - 不直接优化提示向量
    - 使用BiLSTM编码器生成提示
    - 提示可以插入到输入序列的任意位置
    
    参数：
        num_prompts: 提示数量
        hidden_dim: 隐藏层维度
        encoder_hidden_dim: LSTM编码器的隐藏层维度
    """
    def __init__(self, num_prompts=20, hidden_dim=768, encoder_hidden_dim=256):
        super(PTuningV1, self).__init__()
        self.num_prompts = num_prompts
        self.hidden_dim = hidden_dim
        
        # 提示的初始嵌入（输入到LSTM）
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, encoder_hidden_dim))
        
        # BiLSTM编码器：将初始嵌入编码为实际的提示向量
        self.lstm_encoder = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=encoder_hidden_dim // 2,  # 双向，所以除以2
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 投影层：将LSTM输出投影到模型的隐藏维度
        self.projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        
        # 初始化
        nn.init.uniform_(self.prompt_embeddings, -0.5, 0.5)
    
    def forward(self, input_embeds):
        """
        生成并拼接软提示
        
        参数：
            input_embeds: 输入嵌入，shape=(batch_size, seq_len, hidden_dim)
        """
        batch_size = input_embeds.size(0)
        
        # 步骤1: LSTM编码提示嵌入
        # (num_prompts, encoder_hidden_dim) -> (1, num_prompts, encoder_hidden_dim)
        prompt_input = self.prompt_embeddings.unsqueeze(0)
        
        # LSTM编码
        lstm_output, _ = self.lstm_encoder(prompt_input)  # (1, num_prompts, encoder_hidden_dim)
        
        # 步骤2: 投影到目标维度
        soft_prompts = self.projection(lstm_output)  # (1, num_prompts, hidden_dim)
        
        # 步骤3: 扩展到batch维度
        prompts_batch = soft_prompts.expand(batch_size, -1, -1)
        
        # 步骤4: 拼接
        output_embeds = torch.cat([prompts_batch, input_embeds], dim=1)
        
        return output_embeds


# ============================================
# 3. Prefix Tuning（在每层添加前缀）
# ============================================

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: 在Transformer每层的K、V前添加可学习前缀
    
    与Prompt Tuning的区别：
    - Prompt Tuning只在输入层添加提示
    - Prefix Tuning在每个Transformer层都添加前缀
    
    参数：
        num_layers: Transformer层数
        num_heads: 注意力头数
        num_prefix: 前缀长度
        head_dim: 每个注意力头的维度
    """
    def __init__(self, num_layers=12, num_heads=12, num_prefix=20, head_dim=64):
        super(PrefixTuning, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_prefix = num_prefix
        self.head_dim = head_dim
        
        # 为每层的K和V分别创建前缀
        # shape: (num_layers, 2, num_heads, num_prefix, head_dim)
        # 第二个维度2代表K和V
        self.prefix_params = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, num_prefix, head_dim)
        )
        
        # 初始化
        nn.init.normal_(self.prefix_params, mean=0.0, std=0.02)
    
    def get_prefix(self, layer_idx):
        """
        获取指定层的K、V前缀
        
        参数：
            layer_idx: 层索引（0到num_layers-1）
        
        返回：
            prefix_k: K的前缀，shape=(num_heads, num_prefix, head_dim)
            prefix_v: V的前缀，shape=(num_heads, num_prefix, head_dim)
        """
        # 获取该层的前缀
        layer_prefix = self.prefix_params[layer_idx]  # (2, num_heads, num_prefix, head_dim)
        
        prefix_k = layer_prefix[0]  # (num_heads, num_prefix, head_dim)
        prefix_v = layer_prefix[1]  # (num_heads, num_prefix, head_dim)
        
        return prefix_k, prefix_v
    
    def forward(self, k, v, layer_idx):
        """
        在K、V前添加前缀
        
        参数：
            k: Key张量，shape=(batch_size, num_heads, seq_len, head_dim)
            v: Value张量，shape=(batch_size, num_heads, seq_len, head_dim)
            layer_idx: 当前层索引
        
        返回：
            k_with_prefix: 添加前缀后的K
            v_with_prefix: 添加前缀后的V
        """
        batch_size = k.size(0)
        
        # 获取该层的前缀
        prefix_k, prefix_v = self.get_prefix(layer_idx)
        
        # 扩展到batch维度
        # (num_heads, num_prefix, head_dim) -> (batch_size, num_heads, num_prefix, head_dim)
        prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
        prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 拼接：[前缀] + [原始K/V]
        k_with_prefix = torch.cat([prefix_k, k], dim=2)  # 在seq_len维度拼接
        v_with_prefix = torch.cat([prefix_v, v], dim=2)
        
        return k_with_prefix, v_with_prefix


# ============================================
# 4. 带Prompt Tuning的完整模型
# ============================================

class ModelWithPromptTuning(nn.Module):
    """
    集成Prompt Tuning的模型（简化版）
    
    模拟BERT + Prompt Tuning的组合
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        num_prompts=20,
        prompt_method='prompt_tuning'
    ):
        super(ModelWithPromptTuning, self).__init__()
        self.hidden_dim = hidden_dim
        self.prompt_method = prompt_method
        
        # Token嵌入（冻结）
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # 提示学习模块（只有这部分可训练）
        if prompt_method == 'prompt_tuning':
            self.prompt_module = PromptTuning(num_prompts, hidden_dim)
        elif prompt_method == 'p_tuning_v1':
            self.prompt_module = PTuningV1(num_prompts, hidden_dim)
        
        # Transformer层（冻结，这里用一个简化的MLP代替）
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # 分类头（可选训练）
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, input_ids):
        """
        参数：
            input_ids: token IDs，shape=(batch_size, seq_len)
        """
        # 步骤1: 获取token嵌入
        input_embeds = self.embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # 步骤2: 添加软提示
        embeds_with_prompt = self.prompt_module(input_embeds)
        # shape: (batch_size, num_prompts + seq_len, hidden_dim)
        
        # 步骤3: 通过Transformer
        hidden_states = self.transformer(embeds_with_prompt)
        
        # 步骤4: 取[CLS]位置的输出（这里是第一个提示token的位置）
        cls_output = hidden_states[:, 0, :]
        
        # 步骤5: 分类
        logits = self.classifier(cls_output)
        
        return logits


# ============================================
# 5. 参数量统计
# ============================================

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ============================================
# 6. 示例与对比
# ============================================

def prompt_tuning_basic_example():
    """Prompt Tuning基础示例"""
    print("=" * 60)
    print("Prompt Tuning基础示例")
    print("=" * 60)
    
    # 创建Prompt Tuning模块
    prompt_tuning = PromptTuning(num_prompts=20, hidden_dim=768)
    
    # 模拟输入（已经是嵌入向量）
    batch_size, seq_len, hidden_dim = 2, 10, 768
    input_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\n原始输入形状: {input_embeds.shape}")
    print(f"提示数量: {prompt_tuning.num_prompts}")
    
    # 添加软提示
    output_embeds = prompt_tuning(input_embeds)
    print(f"添加提示后形状: {output_embeds.shape}")
    
    # 参数量
    total, trainable = count_parameters(prompt_tuning)
    print(f"\n可训练参数量: {total:,}")
    print(f"参数明细: {prompt_tuning.num_prompts} × {hidden_dim} = {total:,}")
    
    print()


def p_tuning_v1_example():
    """P-Tuning v1示例"""
    print("=" * 60)
    print("P-Tuning v1示例（使用LSTM编码器）")
    print("=" * 60)
    
    # 创建P-Tuning v1模块
    p_tuning = PTuningV1(num_prompts=20, hidden_dim=768, encoder_hidden_dim=256)
    
    # 模拟输入
    batch_size, seq_len, hidden_dim = 2, 10, 768
    input_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\n原始输入形状: {input_embeds.shape}")
    
    # 添加编码后的提示
    output_embeds = p_tuning(input_embeds)
    print(f"添加提示后形状: {output_embeds.shape}")
    
    # 参数量
    total, trainable = count_parameters(p_tuning)
    print(f"\n可训练参数量: {total:,}")
    print(f"包括: 提示嵌入 + BiLSTM + 投影层")
    
    print("\n【P-Tuning v1的优势】")
    print("✓ LSTM可以捕获提示之间的依赖关系")
    print("✓ 相比直接优化向量，训练更稳定")
    print("✓ 可以插入到序列的任意位置（这里简化为前缀）")
    
    print()


def prefix_tuning_example():
    """Prefix Tuning示例"""
    print("=" * 60)
    print("Prefix Tuning示例（每层添加K、V前缀）")
    print("=" * 60)
    
    # 创建Prefix Tuning模块
    num_layers, num_heads, num_prefix, head_dim = 12, 12, 20, 64
    prefix_tuning = PrefixTuning(num_layers, num_heads, num_prefix, head_dim)
    
    # 模拟某一层的K、V
    batch_size, seq_len = 2, 10
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"\n原始K形状: {k.shape}")
    print(f"原始V形状: {v.shape}")
    print(f"前缀长度: {num_prefix}")
    
    # 添加前缀（以第0层为例）
    k_with_prefix, v_with_prefix = prefix_tuning(k, v, layer_idx=0)
    
    print(f"\n添加前缀后K形状: {k_with_prefix.shape}")
    print(f"添加前缀后V形状: {v_with_prefix.shape}")
    
    # 参数量
    total, trainable = count_parameters(prefix_tuning)
    print(f"\n可训练参数量: {total:,}")
    print(f"参数明细: {num_layers} × 2 × {num_heads} × {num_prefix} × {head_dim}")
    
    print("\n【Prefix Tuning的特点】")
    print("✓ 在每层都添加前缀，影响更深层的表示")
    print("✓ 只修改K、V，不修改Q（Query来自输入）")
    print("✓ 参数量稍多，但效果通常更好")
    
    print()


def comparison_example():
    """对比三种方法的参数量"""
    print("=" * 60)
    print("提示学习方法参数量对比")
    print("=" * 60)
    
    num_prompts = 20
    hidden_dim = 768
    num_heads = 12
    head_dim = 64
    num_layers = 12
    
    # 1. Prompt Tuning
    pt = PromptTuning(num_prompts, hidden_dim)
    pt_total, pt_train = count_parameters(pt)
    
    # 2. P-Tuning v1
    ptv1 = PTuningV1(num_prompts, hidden_dim, encoder_hidden_dim=256)
    ptv1_total, ptv1_train = count_parameters(ptv1)
    
    # 3. Prefix Tuning
    prefix = PrefixTuning(num_layers, num_heads, num_prompts, head_dim)
    prefix_total, prefix_train = count_parameters(prefix)
    
    print(f"\n提示长度: {num_prompts}")
    print(f"隐藏层维度: {hidden_dim}")
    
    print(f"\n1. Prompt Tuning:")
    print(f"   参数量: {pt_train:,}")
    
    print(f"\n2. P-Tuning v1:")
    print(f"   参数量: {ptv1_train:,}")
    print(f"   比Prompt Tuning多: {ptv1_train/pt_train:.2f}x")
    
    print(f"\n3. Prefix Tuning:")
    print(f"   参数量: {prefix_train:,}")
    print(f"   比Prompt Tuning多: {prefix_train/pt_train:.2f}x")
    
    # 假设BERT-base有110M参数
    bert_params = 110_000_000
    print(f"\n相比BERT-base（{bert_params:,}参数）:")
    print(f"  Prompt Tuning: {pt_train/bert_params*100:.4f}%")
    print(f"  P-Tuning v1: {ptv1_train/bert_params*100:.4f}%")
    print(f"  Prefix Tuning: {prefix_train/bert_params*100:.4f}%")
    
    print()


def full_model_example():
    """完整模型示例"""
    print("=" * 60)
    print("带Prompt Tuning的完整模型示例")
    print("=" * 60)
    
    # 创建模型
    model = ModelWithPromptTuning(
        num_prompts=20,
        prompt_method='prompt_tuning'
    )
    
    # 模拟输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    
    print(f"\n输入token IDs形状: {input_ids.shape}")
    
    # 前向传播
    logits = model(input_ids)
    print(f"输出logits形状: {logits.shape}")
    
    # 参数统计
    total, trainable = count_parameters(model)
    print(f"\n总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"可训练比例: {trainable/total*100:.4f}%")
    
    print("\n【冻结的部分】")
    print("✗ Token嵌入层")
    print("✗ Transformer层")
    
    print("\n【可训练的部分】")
    print("✓ 软提示向量")
    print("✓ 分类头（可选）")
    
    print()


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("提示学习（Prompt Tuning）完整示例")
    print("="*60 + "\n")
    
    # 1. Prompt Tuning基础
    prompt_tuning_basic_example()
    
    # 2. P-Tuning v1
    p_tuning_v1_example()
    
    # 3. Prefix Tuning
    prefix_tuning_example()
    
    # 4. 方法对比
    comparison_example()
    
    # 5. 完整模型
    full_model_example()
    
    print("=" * 60)
    print("提示学习示例运行完毕！")
    print("\n【核心要点总结】")
    print("1. Prompt Tuning: 最简单，参数最少（<0.1%）")
    print("2. P-Tuning v1: 使用编码器，训练更稳定")
    print("3. Prefix Tuning: 每层添加前缀，效果更好但参数稍多")
    print("4. 三种方法都远少于全量微调和LoRA的参数量")
    print("5. 适用于大规模模型的轻量级定制化")
    print("=" * 60)

