# PPO vs DPO 理论速查 ⭐⭐⭐⭐

> 💡 本文档为快速复习使用，详细内容请参考 `PPO核心概念详解.md`

## 1. PPO (Proximal Policy Optimization)

**背景:** RLHF (人类反馈强化学习) 的经典算法

**核心思想:**
- 使用Reward Model指导模型优化
- 限制策略更新幅度，防止模型崩溃
- 通过KL散度约束保持与原模型的接近

**优化目标:**
```
maximize: E[r(x,y)] - β * KL(π_θ || π_ref)
```
- r(x,y): 奖励模型打分
- β: KL惩罚系数
- π_θ: 当前策略
- π_ref: 参考策略（SFT模型）

**流程:**
1. SFT (监督微调)
2. 训练Reward Model
3. PPO强化学习优化
4. 多轮迭代

**需要的4个模型:**
- **Actor** (策略模型): 要优化的主模型，生成回答
- **Critic** (价值模型): 估计状态价值，稳定训练
- **Reference** (参考模型): SFT模型副本（冻结），用于KL约束
- **Reward Model** (奖励模型): 给回答打分（冻结）

**优点:**
- 稳定性好
- 效果经过验证（ChatGPT使用）

**缺点:**
- 训练复杂，需要4个模型（显存消耗大）
- 计算资源消耗大
- 超参数敏感

---

## 2. DPO (Direct Preference Optimization)

**背景:** 2023年提出，简化RLHF流程

**核心思想:**
- 直接从偏好数据优化策略
- 不需要显式的Reward Model
- 数学上等价于RLHF，但更简单

**优化目标:**
```
L_DPO = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
```
- y_w: 更优回答
- y_l: 较差回答
- σ: sigmoid函数

**流程:**
1. SFT (监督微调)
2. 收集偏好数据对 (y_w, y_l)
3. 直接优化DPO损失

**只需要2个模型:**
- **Policy** (策略模型): 要优化的主模型
- **Reference** (参考模型): SFT模型副本（冻结）

> ⭐ 对比PPO: 不需要Reward Model和Critic，显存节省50%+

**优点:**
- 训练简单，只需2个模型（策略+参考）
- 更稳定，不会出现reward hacking
- 计算效率高（显存占用小）

**缺点:**
- 对数据质量要求高
- 理论研究还不够深入

---

## 3. 对比总结

| 维度 | PPO | DPO |
|------|-----|-----|
| 复杂度 | 高（4模型） | 低（2模型） |
| 训练稳定性 | 中等 | 高 |
| 资源消耗 | 大 | 小 |
| 数据需求 | Reward标注 | 偏好对比 |
| 理论成熟度 | 成熟 | 较新 |
| 工业应用 | 广泛 | 增长中 |

## 4. 面试重点

- ✅ 说明RLHF的完整流程
- ✅ 解释PPO的核心约束（KL散度）⭐⭐⭐
- ✅ 理解DPO为什么更简单
- ✅ 知道何时选择PPO/DPO

## 相关文档

- 📖 [PPO核心概念详解](./PPO核心概念详解.md) - 完整理论
- 📖 [面试答题_PPO_DPO](./面试答题_PPO_DPO.md) - 面试答题示例
- 📄 [advantage_function_demo.py](../代码实现/advantage_function_demo.py) - 优势函数演示
- 📄 [kl_divergence.py](../代码实现/kl_divergence.py) - KL散度演示

