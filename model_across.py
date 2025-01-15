import torch
import torch.nn as nn
from across_big import HAN2
from across_small import PageRank2
import numpy as np
import copy
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):  #64,128
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))# 64,128  -->  128,1
    def forward(self, z):
        w = self.project(z).mean(0)     #w.shape(2,1)  经过project变成（4019,2,1） 在经过mean变成（2,1）
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)
class models(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim,drop=0.5): #输入特征维度 隐藏层的大小
        super(models, self).__init__()
        self.input_dim = input_dim
        self.no_linear = nn.ReLU()
        self.fc_list = nn.ModuleList([
             nn.Linear(feat_dim, hid_dim) for feat_dim in input_dim
        ])
        self.page_rank= PageRank2(hid_dim)
        self.big_agg = HAN2(4,2,8,8,3,[8],0.5)
        self.semantic_attention = SemanticAttention(in_size=64)
        self.dropout = nn.Dropout(p=drop)
        self.project = nn.Linear(64,out_dim)


    def degree(self,adj): #计算邻接矩阵里每个图的总度值
        adj_degree = []
        for graph in adj:
            # 获取图的邻接矩阵
            adj_matrix = graph.adjacency_matrix().to_dense()
            # 计算每个节点的度
            degree_vector = adj_matrix.sum(dim=1)
            # 计算总度值
            total_degree = degree_vector.sum()
            adj_degree.append(total_degree)
        return adj_degree
    def degree_analyze(self,degree): #分析每个图的总度值的相对差异
        min_value = np.min(degree)
        epsilon = 1e-6
        if min_value == 0:
            min_value = epsilon
        relative_diffs = [(value - min_value) / (min_value + epsilon)* 100 for value in degree]
        return relative_diffs
    def forward(self,features,adj,simlar):
        h_all = [self.no_linear(self.fc_list[i](features[i])) for i in range(len(features))] #通过应用线性变换和激活函数(增加非线性)计算所有隐藏表示 h_all
        adj_degree = self.degree(adj)   #度统计
        #对总度值进行分析，选择应用策略--->比较不同值之间的差异情况
        relative_diffs = self.degree_analyze(adj_degree)
        big_degree_adj = []
        small_degree_adj = copy.deepcopy(adj)
        j=0
        for i in range(len(relative_diffs)):
            if relative_diffs[i] >= 30:
                big_degree_adj.append(adj[i])
                small_degree_adj.pop(i-j)
                j+=1
        #邻接矩阵分类完毕  对大度矩阵与小度矩阵分别运算
        small_degree_feat = [self.dropout(self.page_rank(h_all[0], small_degree_adj[i], simlar)) for i in
                           range(len(small_degree_adj))]
        #使用 PageRank 方法处理 h_all[0]（第一个特征的隐藏表示），结合 PRO[i] 和 simlar，得到大度特征 big_degree_feat。
        big_degree_feat = [self.big_agg(big_degree_adj[i],h_all[0]) for i in range(len(big_degree_adj))]
        feat = []
        feat.extend(big_degree_feat)
        feat.extend(small_degree_feat)
        h = self.semantic_attention(torch.stack(feat, dim=1)) #将 feat 列表中的特征沿第二个维度堆叠（形成一个新的张量），并通过注意力机制（semantic_attention）计算最终的聚合特征 h。
        # h = self.no_linear(self.project1(h))
        a = self.project(h)
        return a,h