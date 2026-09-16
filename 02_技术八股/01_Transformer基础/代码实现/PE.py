import numpy as np
# import torch
# import torch.nn as nn

def get_positional_encoding(seq_len, d_model):
    """
    生成位置编码
    """
    # 初始化位置编码矩阵
    PE = np.zeros((seq_len, d_model))
    
    # 位置索引 (0, 1, 2, ..., seq_len-1)
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
    
    # 维度索引
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)
    
    # 偶数维: sin
    PE[:, 0::2] = np.sin(position * div_term)
    
    # 奇数维: cos
    PE[:, 1::2] = np.cos(position * div_term)
    
    return PE

# 示例
PE = get_positional_encoding(seq_len=100, d_model=512)
print(PE.shape)  # (100, 512)


# position_embeddings = nn.Embedding(max_position, d_model)

# 使用时
# pos_ids = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
# pos_emb = position_embeddings(pos_ids)