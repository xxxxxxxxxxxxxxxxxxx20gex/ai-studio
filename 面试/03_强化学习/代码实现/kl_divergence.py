"""
KL散度（Kullback-Leibler Divergence）简明教程

【核心概念】
KL散度是什么？
- 衡量两个概率分布之间的差异程度
- 数学公式：KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
- 性质：非对称、非负、P=Q时为0

【在PPO中的作用】
- 限制新旧策略的差距，防止更新步子太大
- 保证训练稳定性
"""

import torch


# ============================================
# 1. 基础KL散度计算
# ============================================

def kl_divergence(p, q):
    """
    计算KL散度：KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
    
    参数：
        p: 真实分布 (形状任意，最后一维是概率)
        q: 近似分布 (形状与p相同)
    
    返回：
        KL散度值
    """
    # 防止log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # KL散度公式
    kl = torch.sum(p * torch.log(p / q), dim=-1)
    return kl


# ============================================
# 2. 简单示例：理解KL散度
# ============================================

def basic_example():
    """基础示例：对比不同分布的KL散度"""
    
    print("=" * 60)
    print("示例1：基础KL散度计算")
    print("=" * 60)
    
    # 分布1：均匀分布 [0.5, 0.5]
    p1 = torch.tensor([0.5, 0.5])
    
    # 分布2：偏向第一类 [0.8, 0.2]
    p2 = torch.tensor([0.8, 0.2])
    
    # 分布3：偏向第二类 [0.2, 0.8]
    p3 = torch.tensor([0.2, 0.8])
    
    print("\n分布定义：")
    print(f"P1 (均匀): {p1.numpy()}")
    print(f"P2 (偏1):  {p2.numpy()}")
    print(f"P3 (偏2):  {p3.numpy()}")
    
    # 计算KL散度
    print("\nKL散度计算：")
    print(f"KL(P1 || P1) = {kl_divergence(p1, p1).item():.4f}  ← 相同分布，为0")
    print(f"KL(P1 || P2) = {kl_divergence(p1, p2).item():.4f}")
    print(f"KL(P2 || P1) = {kl_divergence(p2, p1).item():.4f}  ← 非对称性")
    print(f"KL(P2 || P3) = {kl_divergence(p2, p3).item():.4f}  ← 差异大，值大")
    
    print("\n关键点：")
    print("✓ 分布越相似，KL散度越小")
    print("✓ KL(P||Q) ≠ KL(Q||P)  (非对称)")
    print("✓ 相同分布时KL散度为0")


# ============================================
# 3. PPO中的应用
# ============================================

def ppo_example():
    """PPO中的KL散度示例"""
    
    print("\n" + "=" * 60)
    print("示例2：PPO中的KL散度")
    print("=" * 60)
    
    # 假设有3个动作
    num_actions = 3
    batch_size = 4
    
    # 旧策略的动作概率（固定）
    old_policy_probs = torch.tensor([
        [0.6, 0.3, 0.1],  # 状态1
        [0.5, 0.4, 0.1],  # 状态2
        [0.7, 0.2, 0.1],  # 状态3
        [0.4, 0.4, 0.2],  # 状态4
    ])
    
    # 新策略的动作概率（训练中更新）
    # 情况1：几乎没变
    new_policy_similar = torch.tensor([
        [0.61, 0.29, 0.10],
        [0.51, 0.39, 0.10],
        [0.69, 0.21, 0.10],
        [0.41, 0.39, 0.20],
    ])
    
    # 情况2：变化很大
    new_policy_different = torch.tensor([
        [0.2, 0.3, 0.5],  # 完全颠倒
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
        [0.8, 0.1, 0.1],
    ])
    
    print("\n计算新旧策略之间的KL散度：")
    
    # 计算KL散度
    kl_similar = kl_divergence(old_policy_probs, new_policy_similar).mean()
    kl_different = kl_divergence(old_policy_probs, new_policy_different).mean()
    
    print(f"\n策略微调（几乎不变）：")
    print(f"  平均KL散度 = {kl_similar.item():.6f}  ← 很小，可以接受")
    
    print(f"\n策略大变：")
    print(f"  平均KL散度 = {kl_different.item():.4f}  ← 很大，需要惩罚！")
    
    print("\nPPO的做法：")
    print("  如果 KL散度 > 阈值:")
    print("    → 增加惩罚系数β")
    print("    → 限制策略更新幅度")
    print("    → 保证训练稳定")


# ============================================
# 4. 高斯分布的KL散度（连续动作）
# ============================================

def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """
    两个高斯分布的KL散度（解析解）
    
    公式：KL(N(μ1,σ1²) || N(μ2,σ2²)) 
        = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 1/2
    """
    kl = torch.log(sigma2 / sigma1) + \
         (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
    return kl


def gaussian_example():
    """高斯分布示例（用于连续动作空间）"""
    
    print("\n" + "=" * 60)
    print("示例3：高斯分布的KL散度（连续动作）")
    print("=" * 60)
    
    # 两个高斯分布
    mu1, sigma1 = torch.tensor(0.0), torch.tensor(1.0)   # N(0, 1)
    mu2, sigma2 = torch.tensor(0.0), torch.tensor(1.0)   # N(0, 1)
    mu3, sigma3 = torch.tensor(2.0), torch.tensor(1.5)   # N(2, 1.5)
    
    print("\n分布定义：")
    print(f"P1: N(均值={mu1.item()}, 标准差={sigma1.item()})")
    print(f"P2: N(均值={mu2.item()}, 标准差={sigma2.item()})")
    print(f"P3: N(均值={mu3.item()}, 标准差={sigma3.item()})")
    
    # 计算KL散度
    kl1 = gaussian_kl(mu1, sigma1, mu2, sigma2)
    kl2 = gaussian_kl(mu1, sigma1, mu3, sigma3)
    
    print("\nKL散度：")
    print(f"KL(P1 || P2) = {kl1.item():.6f}  ← 相同分布")
    print(f"KL(P1 || P3) = {kl2.item():.4f}  ← 不同分布")
    
    print("\n应用：连续动作PPO（如机器人控制）")


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    
    print("\n" + "🎯" * 30)
    print("KL散度（Kullback-Leibler Divergence）简明教程")
    print("🎯" * 30 + "\n")
    
    # 运行示例
    basic_example()      # 基础概念
    ppo_example()        # PPO应用
    gaussian_example()   # 连续动作
    
    # 总结
    print("\n" + "=" * 60)
    print("📚 核心要点")
    print("=" * 60)
    print("\n1. KL散度 = 衡量两个分布的差异")
    print("   公式：KL(P||Q) = Σ P(i) * log(P(i) / Q(i))")
    
    print("\n2. 性质：")
    print("   • 非对称：KL(P||Q) ≠ KL(Q||P)")
    print("   • 非负：KL ≥ 0")
    print("   • P=Q时等于0")
    
    print("\n3. PPO中的作用：")
    print("   • 限制新旧策略差距")
    print("   • KL散度大 → 惩罚加重")
    print("   • 保证训练稳定")
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60 + "\n")

