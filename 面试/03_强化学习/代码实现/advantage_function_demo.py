"""
优势函数（Advantage Function）演示

这个脚本演示 PPO 中的优势函数如何工作：
A(s,a) = Q(s,a) - V(s)

通过具体例子理解：
- Q(s,a): 状态-动作价值（采取某个动作的期望回报）
- V(s): 状态价值（当前状态的平均期望回报）
- A(s,a): 优势（该动作相比平均水平好多少）
"""

import numpy as np


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def calculate_advantage_simple():
    """示例1: 简单的优势函数计算"""
    print_section("示例1: 优势函数基础计算")
    
    # 场景：LLM生成下一个词
    state = "请解释什么是"
    print(f"\n当前状态: '{state}'")
    
    # 可能的动作（下一个词）及其价值
    actions = {
        "人工智能": 9.5,
        "AI": 9.2,
        "这个": 6.0,
        "嗯嗯嗯": 2.0,
        "机器学习": 8.8
    }
    
    # 计算平均价值（状态价值）
    V_s = np.mean(list(actions.values()))
    
    print(f"\n可能的动作及其 Q(s,a) 值:")
    for action, q_value in actions.items():
        print(f"  '{action}': Q(s,a) = {q_value:.1f}")
    
    print(f"\n状态价值 V(s) = {V_s:.1f} (平均水平)")
    
    # 计算优势函数
    print("\n优势函数 A(s,a) = Q(s,a) - V(s):")
    advantages = {}
    for action, q_value in actions.items():
        advantage = q_value - V_s
        advantages[action] = advantage
        
        # 根据优势值给出评价
        if advantage > 2:
            evaluation = "✓✓ 很好！大力鼓励"
        elif advantage > 0:
            evaluation = "✓  不错，鼓励"
        elif advantage > -2:
            evaluation = "✗  不好，惩罚"
        else:
            evaluation = "✗✗ 很差！大力惩罚"
        
        print(f"  '{action}': A = {q_value:.1f} - {V_s:.1f} = {advantage:+.1f}  {evaluation}")
    
    return advantages


def simulate_ppo_update():
    """示例2: 模拟PPO如何使用优势函数更新策略"""
    print_section("示例2: PPO使用优势函数更新策略")
    
    state = "写一首关于春天的诗，第一句是"
    print(f"\n场景: '{state}'")
    
    # 旧策略的概率分布
    print("\n【旧策略】的概率分布:")
    old_policy = {
        "春眠不觉晓": 0.25,
        "春风又绿": 0.20,
        "啊啊啊": 0.15,
        "呃呃呃": 0.10,
        "春江水暖": 0.30
    }
    
    for action, prob in old_policy.items():
        print(f"  '{action}': π_old = {prob:.2f}")
    
    # 奖励（Q值）
    print("\n【实际获得的奖励】Q(s,a):")
    q_values = {
        "春眠不觉晓": 9.0,
        "春风又绿": 8.5,
        "啊啊啊": 2.0,
        "呃呃呃": 1.5,
        "春江水暖": 9.2
    }
    
    for action, q in q_values.items():
        print(f"  '{action}': Q = {q:.1f}")
    
    # 计算状态价值（平均奖励）
    V_s = np.mean(list(q_values.values()))
    print(f"\n【状态价值】V(s) = {V_s:.1f}")
    
    # 计算优势函数
    print("\n【优势函数】A(s,a) = Q(s,a) - V(s):")
    advantages = {}
    for action in q_values:
        advantage = q_values[action] - V_s
        advantages[action] = advantage
        print(f"  '{action}': A = {advantage:+.1f}")
    
    # PPO更新策略
    print("\n【PPO策略更新】:")
    print("规则: A > 0 → 增加概率, A < 0 → 减少概率")
    
    new_policy = {}
    learning_rate = 0.1  # 学习率
    
    for action in old_policy:
        advantage = advantages[action]
        
        # 简化的策略更新（实际PPO更复杂）
        if advantage > 0:
            # 好动作，增加概率
            change = learning_rate * advantage / 10
            new_prob = old_policy[action] + change
            direction = "↑ 增加"
        else:
            # 差动作，减少概率
            change = learning_rate * abs(advantage) / 10
            new_prob = max(0.05, old_policy[action] - change)  # 保持最小概率
            direction = "↓ 减少"
        
        # 归一化
        new_policy[action] = new_prob
        
        print(f"  '{action}':")
        print(f"    π_old = {old_policy[action]:.2f} → π_new = {new_prob:.2f} {direction}")
        print(f"    原因: A = {advantage:+.1f}")
    
    # 归一化新策略
    total = sum(new_policy.values())
    for action in new_policy:
        new_policy[action] /= total
    
    print("\n【归一化后的新策略】:")
    for action, prob in new_policy.items():
        old_prob = old_policy[action]
        change = prob - old_prob
        arrow = "↑" if change > 0 else "↓"
        print(f"  '{action}': {prob:.3f} ({arrow} {abs(change):.3f})")


def compare_with_without_baseline():
    """示例3: 对比有无基准线（V(s)）的区别"""
    print_section("示例3: 为什么需要基准线V(s)？")
    
    # 场景1: 所有奖励都很高
    print("\n【场景1: 所有动作奖励都很高】")
    state = "回答技术问题"
    q_high = {
        "详细解释": 95,
        "简短回答": 92,
        "举例说明": 97,
        "引用论文": 90
    }
    
    print(f"Q值: {list(q_high.values())}")
    
    # 不使用基准线
    print("\n❌ 不使用基准线 (只看绝对Q值):")
    print("  所有动作奖励都很高 (90-97)")
    print("  → 难以区分哪个更好")
    print("  → 可能都会增加概率，导致过度自信")
    
    # 使用基准线
    V_s = np.mean(list(q_high.values()))
    print(f"\n✓ 使用基准线 V(s) = {V_s:.1f}:")
    for action, q in q_high.items():
        advantage = q - V_s
        print(f"  '{action}': A = {q} - {V_s:.1f} = {advantage:+.1f}")
    
    print("\n  → 清楚看出相对好坏")
    print("  → '举例说明' (A=+3.5) 比 '引用论文' (A=-3.5) 更好")
    
    # 场景2: 所有奖励都很低
    print("\n【场景2: 所有动作奖励都很低】")
    state = "回答超纲问题"
    q_low = {
        "坦诚不知": 25,
        "瞎编乱造": 15,
        "推荐资源": 30,
        "回避问题": 18
    }
    
    print(f"Q值: {list(q_low.values())}")
    
    V_s = np.mean(list(q_low.values()))
    print(f"\n✓ 使用基准线 V(s) = {V_s:.1f}:")
    for action, q in q_low.items():
        advantage = q - V_s
        print(f"  '{action}': A = {q} - {V_s:.1f} = {advantage:+.1f}")
    
    print("\n  → 即使绝对值都很低，仍能区分相对好坏")
    print("  → '推荐资源' 是相对最好的选择")


def advantage_in_training():
    """示例4: 训练过程中优势函数的变化"""
    print_section("示例4: 训练过程中优势函数的变化")
    
    print("\n模拟一个动作在训练过程中的变化:")
    print("动作: 生成词 '人工智能'")
    print("状态: '请解释什么是'")
    
    epochs = [0, 100, 500, 1000, 2000]
    
    print("\n训练轮次 | Q(s,a) | V(s) | A(s,a) | 策略概率 | 说明")
    print("-" * 70)
    
    # 模拟训练过程
    for epoch in epochs:
        if epoch == 0:
            q_value = 6.5
            v_value = 5.0
            prob = 0.15
            note = "初始状态"
        elif epoch == 100:
            q_value = 7.8
            v_value = 6.5
            prob = 0.22
            note = "开始学习，A>0 → 概率增加"
        elif epoch == 500:
            q_value = 8.9
            v_value = 7.8
            prob = 0.35
            note = "继续优化"
        elif epoch == 1000:
            q_value = 9.2
            v_value = 8.5
            prob = 0.45
            note = "接近最优"
        else:
            q_value = 9.5
            v_value = 9.0
            prob = 0.48
            note = "收敛，A接近0"
        
        advantage = q_value - v_value
        
        print(f"{epoch:>4} | {q_value:>6.1f} | {v_value:>4.1f} | {advantage:>+6.1f} | {prob:>8.2f} | {note}")
    
    print("\n观察:")
    print("1. 优势函数 A > 0 时，策略概率持续增加")
    print("2. 随着训练，Q(s,a) 和 V(s) 都在提升")
    print("3. 最终 A 接近 0，说明该动作已经是平均水平")
    print("4. 如果其他动作更好，A 可能变成负数，概率会下降")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "优势函数 (Advantage Function) 演示" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # 示例1: 基础计算
    calculate_advantage_simple()
    
    # 示例2: PPO更新
    simulate_ppo_update()
    
    # 示例3: 基准线的重要性
    compare_with_without_baseline()
    
    # 示例4: 训练过程
    advantage_in_training()
    
    # 总结
    print_section("核心要点总结")
    print("""
┌─────────────────────────────────────────────────────────────┐
│  优势函数: A(s,a) = Q(s,a) - V(s)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Q(s,a): 采取动作a的期望回报（绝对价值）                    │
│  V(s):   当前状态的平均回报（基准线）                        │
│  A(s,a): 动作a相对于平均水平的优势                           │
│                                                             │
│  作用:                                                       │
│    • A > 0: 比平均好 → 增加概率 ✓                          │
│    • A < 0: 比平均差 → 减少概率 ✗                          │
│    • A ≈ 0: 接近平均 → 保持不变                            │
│                                                             │
│  为什么需要优势函数？                                        │
│    1. 消除绝对奖励值的影响（可能都很高或都很低）             │
│    2. 关注相对好坏，而非绝对好坏                            │
│    3. 减少方差，稳定训练                                    │
│    4. 让模型知道每个动作相对于平均水平的表现                 │
│                                                             │
│  Critic的作用:                                               │
│    • 估计 V(s) 提供基准线                                   │
│    • 帮助计算优势函数                                        │
│    • 这就是 PPO 需要 Critic 的原因！                        │
│                                                             │
│  DPO为什么不需要？                                           │
│    • DPO直接从偏好对学习，不需要估计价值                    │
│    • 这是DPO简化的地方之一                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

💡 记忆口诀:
   Q - V = A
   绝对 - 平均 = 优势
   做得好不好？ 看和平均比！
""")


if __name__ == "__main__":
    main()

