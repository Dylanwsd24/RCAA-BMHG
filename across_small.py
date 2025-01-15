import torch
import torch.nn as nn
import numpy as np
from Attention1 import GCN, MeanPooling,MaxPooling


class PageRank2(nn.Module):
    def __init__(self,hid_dim):
        super(PageRank2, self).__init__()
        self.hid_dim = hid_dim
        self.gcn = GCN(hid_dim)
        self.project = nn.Linear(hid_dim,hid_dim)
        # self.dropout = nn.Dropout(p=0.5)
        self.non_linear = nn.ReLU()
        self.non_linear1 = nn.LeakyReLU()
        # 定义一个线性层用于维度调整到 [9089, 64]
        self.dim_transform = nn.Linear(hid_dim, 64)
        # self.mean_pooling = MeanPooling()




    def forward(self, h, adj, simlar):
        # 从DGLGraph中提取邻接矩阵（通常是CSR格式）
        adj_matrix = adj.adj().to_dense()  # 转换为密集矩阵
        similarity_matrix = self.non_linear(adj_matrix * simlar)

        # Step 2: 仅考虑adj矩阵中存在的连接
        filtered_similarity_matrix = similarity_matrix * (adj_matrix > 0).float()
        # 初始化新的连接矩阵
        new_connection_matrix = torch.zeros_like(filtered_similarity_matrix)
        t = 500 # 设定t值，根据需要调整
        for i in range(filtered_similarity_matrix.size(0)):
            # Step 3: 对于每一行，找到相似性最高的前t个节点的值和索引
            top_values, top_indices = torch.topk(filtered_similarity_matrix[i],
                                                 min(t, filtered_similarity_matrix[i].nonzero().size(0)))
            # Step 4: 将这些最高值填充到新的连接矩阵的对应位置
            new_connection_matrix[i, top_indices] = top_values
        # 这里可以继续后续处理，例如利用new_connection_matrix更新节点特征
        # 示例：使用新的连接矩阵进行特征聚合
        h = self.non_linear(self.project(h))
        feat = self.gcn(h, new_connection_matrix)
        feat = self.dim_transform(feat)
        feat = self.non_linear(feat)
        return feat
