import torch
import torch.nn as nn
import numpy as np
from Attention1 import GCN, MeanPooling,MaxPooling


class PageRank1(nn.Module):
    def __init__(self,hid_dim):
        super(PageRank1, self).__init__()
        self.hid_dim = hid_dim
        # self.gcn = GCN(hid_dim)
        self.project = nn.Linear(hid_dim,hid_dim)
        # self.dropout = nn.Dropout(p=0.5)
        self.non_linear = nn.ReLU()
        self.non_linear1 = nn.LeakyReLU()
        # 定义一个线性层用于维度调整到 [9089, 64]
        self.dim_transform = nn.Linear(hid_dim, 64)
        self.mean_pooling = MeanPooling()


    def forward(self, h, adj):
        # 从DGLGraph中提取邻接矩阵（通常是CSR格式）
        adj_matrix = adj.adj().to_dense()  # 转换为密集矩阵
        # 示例：使用新的连接矩阵进行特征聚合
        h = self.non_linear(self.project(h))
        feat = self.mean_pooling(h, adj_matrix)
        feat = self.dim_transform(feat)
        feat = self.non_linear(feat)

        return feat
