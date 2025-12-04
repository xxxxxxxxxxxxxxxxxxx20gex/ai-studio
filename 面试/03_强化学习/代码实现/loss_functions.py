"""
经典损失函数（Loss Functions）详解

【核心概念】
损失函数是什么？
- 衡量模型预测值与真实值之间的差距
- 训练过程中通过最小化损失函数来优化模型
- 不同任务需要选择合适的损失函数

【常见损失函数分类】
1. 分类任务：交叉熵损失、Focal Loss
2. 回归任务：MSE、MAE
3. 对比学习：Contrastive Loss、Triplet Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 1. 交叉熵损失（Cross Entropy Loss）
# ============================================

def cross_entropy_loss_manual(predictions, targets):
    """
    手动实现交叉熵损失（用于分类任务）
    
    公式：L = -Σ y_true * log(y_pred)
    
    参数：
        predictions: 模型输出的logits，shape=(batch_size, num_classes)
        targets: 真实标签，shape=(batch_size,)，值为类别索引
    
    返回：
        交叉熵损失值（标量）
    """
    # 步骤1: 对logits应用softmax，得到概率分布
    # softmax(x_i) = exp(x_i) / Σ exp(x_j)
    probs = F.softmax(predictions, dim=1)
    
    # 步骤2: 取出目标类别的概率
    # 使用gather选取每个样本对应类别的概率
    batch_size = predictions.size(0)
    target_probs = probs[range(batch_size), targets]
    
    # 步骤3: 计算负对数似然
    # L = -log(p_target)
    loss = -torch.log(target_probs + 1e-10)  # 加小数防止log(0)
    
    # 步骤4: 返回平均损失
    return loss.mean()


def cross_entropy_example():
    """交叉熵损失使用示例"""
    print("=" * 60)
    print("交叉熵损失（Cross Entropy Loss）示例")
    print("=" * 60)
    
    # 模拟数据：3个样本，4个类别
    batch_size, num_classes = 3, 4
    
    # 模型输出的logits（未归一化的分数）
    logits = torch.randn(batch_size, num_classes)
    # 真实标签（类别索引）
    targets = torch.tensor([0, 2, 1])
    
    print(f"模型输出logits:\n{logits}")
    print(f"真实标签: {targets}")
    
    # 方法1: 手动实现
    loss_manual = cross_entropy_loss_manual(logits, targets)
    print(f"\n手动计算的交叉熵损失: {loss_manual.item():.4f}")
    
    # 方法2: PyTorch内置
    loss_builtin = F.cross_entropy(logits, targets)
    print(f"PyTorch内置交叉熵损失: {loss_builtin.item():.4f}")
    
    print("\n【应用场景】")
    print("- 多分类任务（如图像分类、文本分类）")
    print("- 语言模型的下一个token预测")
    print("- 序列标注任务")
    print()


# ============================================
# 2. 均方误差损失（Mean Squared Error, MSE）
# ============================================

def mse_loss_manual(predictions, targets):
    """
    手动实现MSE损失（用于回归任务）
    
    公式：L = (1/n) * Σ (y_pred - y_true)²
    
    参数：
        predictions: 模型预测值，shape任意
        targets: 真实值，shape与predictions相同
    
    返回：
        MSE损失值（标量）
    """
    # 步骤1: 计算差值的平方
    squared_diff = (predictions - targets) ** 2
    
    # 步骤2: 求平均
    loss = squared_diff.mean()
    
    return loss


def mse_example():
    """MSE损失使用示例"""
    print("=" * 60)
    print("均方误差损失（MSE Loss）示例")
    print("=" * 60)
    
    # 模拟回归任务：预测房价
    predictions = torch.tensor([100.0, 150.0, 200.0])  # 预测的房价
    targets = torch.tensor([110.0, 145.0, 195.0])      # 真实的房价
    
    print(f"预测值: {predictions}")
    print(f"真实值: {targets}")
    
    # 方法1: 手动实现
    loss_manual = mse_loss_manual(predictions, targets)
    print(f"\n手动计算的MSE损失: {loss_manual.item():.4f}")
    
    # 方法2: PyTorch内置
    loss_builtin = F.mse_loss(predictions, targets)
    print(f"PyTorch内置MSE损失: {loss_builtin.item():.4f}")
    
    print("\n【应用场景】")
    print("- 回归任务（如房价预测、温度预测）")
    print("- 图像重建（如自编码器）")
    print("- 时间序列预测")
    print()


# ============================================
# 3. Focal Loss（处理类别不平衡）
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss实现（用于类别不平衡的分类任务）
    
    核心思想：
    - 降低易分类样本的权重
    - 聚焦于难分类样本
    
    公式：FL = -α * (1 - p_t)^γ * log(p_t)
    
    参数：
        alpha: 平衡因子，调节正负样本权重，默认0.25
        gamma: 聚焦参数，控制难易样本的权重差异，默认2.0
               gamma越大，易分类样本权重越低
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        参数：
            predictions: 模型输出的logits，shape=(batch_size, num_classes)
            targets: 真实标签，shape=(batch_size,)
        """
        # 步骤1: 计算交叉熵（包含softmax）
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # 步骤2: 计算预测概率
        probs = F.softmax(predictions, dim=1)
        batch_size = predictions.size(0)
        pt = probs[range(batch_size), targets]  # 目标类别的概率
        
        # 步骤3: 应用Focal Loss公式
        # (1 - pt)^gamma：难样本权重高（pt小），易样本权重低（pt大）
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


def focal_loss_example():
    """Focal Loss使用示例"""
    print("=" * 60)
    print("Focal Loss示例（处理类别不平衡）")
    print("=" * 60)
    
    # 模拟类别不平衡场景
    batch_size, num_classes = 6, 3
    logits = torch.randn(batch_size, num_classes)
    
    # 类别0出现频率高（4次），类别1和2出现少（各1次）
    targets = torch.tensor([0, 0, 0, 0, 1, 2])
    
    print(f"真实标签分布: {targets}")
    print(f"类别0: 4次, 类别1: 1次, 类别2: 1次（不平衡）")
    
    # 对比普通交叉熵和Focal Loss
    ce_loss = F.cross_entropy(logits, targets)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
    
    print(f"\n普通交叉熵损失: {ce_loss.item():.4f}")
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    print("\n【Focal Loss优势】")
    print("- 自动降低易分类样本（多数类）的权重")
    print("- 让模型更关注难分类样本（少数类）")
    print("- 参数gamma控制聚焦程度：gamma=0退化为普通交叉熵")
    
    print("\n【应用场景】")
    print("- 目标检测（RetinaNet首次提出）")
    print("- 医疗诊断（罕见病检测）")
    print("- 欺诈检测（异常样本少）")
    print()


# ============================================
# 4. 对比损失（Contrastive Loss）
# ============================================

class ContrastiveLoss(nn.Module):
    """
    对比损失（用于对比学习）
    
    核心思想：
    - 拉近相似样本的距离
    - 推远不相似样本的距离
    
    公式：
    L = (1-y) * 0.5 * D² + y * 0.5 * max(0, margin - D)²
    
    参数：
        margin: 边界值，不相似样本的最小距离
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        参数：
            embedding1: 第一个样本的嵌入向量，shape=(batch_size, embedding_dim)
            embedding2: 第二个样本的嵌入向量，shape=(batch_size, embedding_dim)
            label: 相似性标签，1表示相似，0表示不相似
        """
        # 步骤1: 计算欧氏距离
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # 步骤2: 对比损失公式
        # 相似对（label=1）：距离越小越好
        loss_similar = label * torch.pow(distance, 2)
        
        # 不相似对（label=0）：距离超过margin就不惩罚
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        # 步骤3: 综合损失
        loss = 0.5 * (loss_similar + loss_dissimilar)
        
        return loss.mean()


def contrastive_loss_example():
    """对比损失使用示例"""
    print("=" * 60)
    print("对比损失（Contrastive Loss）示例")
    print("=" * 60)
    
    # 模拟人脸识别场景
    batch_size, embedding_dim = 4, 128
    
    # 生成嵌入向量
    embedding1 = torch.randn(batch_size, embedding_dim)
    embedding2 = torch.randn(batch_size, embedding_dim)
    
    # 标签：前2对是同一个人（相似），后2对是不同人（不相似）
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    
    print(f"样本对数量: {batch_size}")
    print(f"嵌入向量维度: {embedding_dim}")
    print(f"相似性标签: {labels} (1=相似, 0=不相似)")
    
    # 计算对比损失
    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(embedding1, embedding2, labels)
    
    print(f"\n对比损失: {loss.item():.4f}")
    
    print("\n【对比损失作用】")
    print("- 相似对：拉近距离（如同一人的不同照片）")
    print("- 不相似对：推远距离至少margin（如不同人的照片）")
    
    print("\n【应用场景】")
    print("- 人脸识别/人脸验证")
    print("- 图像检索")
    print("- 签名验证")
    print()


# ============================================
# 5. 三元组损失（Triplet Loss）
# ============================================

class TripletLoss(nn.Module):
    """
    三元组损失（用于度量学习）
    
    核心思想：
    - Anchor（锚点）应该更接近Positive（正样本）
    - 而远离Negative（负样本）
    
    公式：L = max(0, D(a,p) - D(a,n) + margin)
    
    参数：
        margin: 边界值，正负样本距离的最小差距
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        参数：
            anchor: 锚点样本，shape=(batch_size, embedding_dim)
            positive: 正样本（与anchor同类），shape=(batch_size, embedding_dim)
            negative: 负样本（与anchor不同类），shape=(batch_size, embedding_dim)
        """
        # 步骤1: 计算距离
        # 锚点到正样本的距离（希望小）
        distance_positive = F.pairwise_distance(anchor, positive)
        # 锚点到负样本的距离（希望大）
        distance_negative = F.pairwise_distance(anchor, negative)
        
        # 步骤2: 三元组损失公式
        # 要求：distance_positive + margin < distance_negative
        # 即：正样本距离 比 负样本距离 至少小margin
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()


def triplet_loss_example():
    """三元组损失使用示例"""
    print("=" * 60)
    print("三元组损失（Triplet Loss）示例")
    print("=" * 60)
    
    # 模拟人脸识别场景
    batch_size, embedding_dim = 4, 128
    
    # 生成三元组
    anchor = torch.randn(batch_size, embedding_dim)      # 参考照片
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # 同一人的另一张照片（接近）
    negative = torch.randn(batch_size, embedding_dim)    # 不同人的照片（远离）
    
    print(f"三元组数量: {batch_size}")
    print(f"嵌入向量维度: {embedding_dim}")
    
    # 计算三元组损失
    criterion = TripletLoss(margin=1.0)
    loss = criterion(anchor, positive, negative)
    
    # 显示距离关系
    dist_ap = F.pairwise_distance(anchor, positive).mean()
    dist_an = F.pairwise_distance(anchor, negative).mean()
    
    print(f"\nAnchor到Positive平均距离: {dist_ap.item():.4f}")
    print(f"Anchor到Negative平均距离: {dist_an.item():.4f}")
    print(f"三元组损失: {loss.item():.4f}")
    
    print("\n【三元组损失目标】")
    print("让 D(anchor, positive) + margin < D(anchor, negative)")
    print("即：同类样本距离 比 异类样本距离 至少小margin")
    
    print("\n【应用场景】")
    print("- FaceNet人脸识别")
    print("- 图像检索排序")
    print("- 推荐系统（用户-物品相似度）")
    print()


# ============================================
# 主函数：运行所有示例
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("经典损失函数完整示例")
    print("="*60 + "\n")
    
    # 1. 交叉熵损失
    cross_entropy_example()
    
    # 2. MSE损失
    mse_example()
    
    # 3. Focal Loss
    focal_loss_example()
    
    # 4. 对比损失
    contrastive_loss_example()
    
    # 5. 三元组损失
    triplet_loss_example()
    
    print("=" * 60)
    print("所有损失函数示例运行完毕！")
    print("=" * 60)

