import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from Mix import Mix
import dgl
import numpy as np

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):  #64,128
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))# 64,128  -->  128,1
    def forward(self, z):
        #z.shape（4019,2,64）
        w = self.project(z).mean(0)     #w.shape(2,1)  经过project变成（4019,2,1） 在经过mean变成（2,1）
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)     #8*8
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):#gs:(g1,g2)  h(4019*8)
        semantic_embeddings = []
        for i, g in enumerate(gs):
        #两个GAT层对两个元路径进行卷积 每个元路径分别用不同的卷积层  gat_layers[i]   聚合完成后semantic_embeddings里有两个4019*64的矩阵
            # semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))     #两个元路径聚合，每个得到(4019，64)  通过注意力层得到节点基于元路径的嵌入。 return部分再将这些嵌入传输到语义聚合模块进行聚合
            # 确保每一步都返回张量
            semantic_embedding = self.gat_layers[i](g, h).flatten(1)
            semantic_embeddings.append(semantic_embedding)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)           #得到（4019,2,64）其中2是因为有两个元路径,已经不再是一个列表了，变成了torch
        # 通过注意力机制聚合
        h = self.semantic_attention(semantic_embeddings)
        return h

class HAN1(nn.Module):
    def __init__(self, num_meta_paths, ns_num,in_size, hidden_size, out_size, num_heads, dropout):#2，4000，8,3,[8]，0.5
        super(HAN1, self).__init__()
        self.non_linear = nn.LeakyReLU()
        # self.non_linear1 = nn.ReLU()
        self.fc_trans = nn.Linear(in_size, hidden_size, bias=False)   #(4000,8)
        self.dropout = nn.Dropout(p=dropout)
        self.attribute_mix = Mix(hidden_size, ns_num, [5,5], dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, hidden_size, hidden_size, num_heads[0], dropout))       #(2,8,8,8,0.5)
        for l in range(1, len(num_heads)):      #多头，所以使用多个注意力层进行计算      本行不执行
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)     #（8*8,3）
        # self.semantic_attention = SemanticAttention(in_size=hidden_size * num_heads[-1])  # 初始化 semantic_attention
    def forward(self, g, h, simlar):        #h:4019,4000
        # 从DGLGraph中提取邻接矩阵（通常是CSR格式）
        adj_matrix = g.adj().to_dense()  # 转换为密集矩阵
        similarity_matrix = self.non_linear(adj_matrix * simlar)

        # Step 2: 仅考虑adj矩阵中存在的连接
        filtered_similarity_matrix = similarity_matrix * (adj_matrix > 0).float()
        # 初始化新的连接矩阵
        new_connection_matrix = torch.zeros_like(filtered_similarity_matrix)
        t = 29  # 设定t值，根据需要调整
        for i in range(filtered_similarity_matrix.size(0)):
            # Step 3: 对于每一行，找到相似性最高的前t个节点的值和索引
            top_values, top_indices = torch.topk(filtered_similarity_matrix[i],
                                                 min(t, filtered_similarity_matrix[i].nonzero().size(0)))
            # Step 4: 将这些最高值填充到新的连接矩阵的对应位置
            new_connection_matrix[i, top_indices] = top_values
        h = self.dropout(self.fc_trans(h))    #h:4019,8
        h = self.non_linear(h)

        # 将 new_connection_matrix 转换为 DGL 图对象
        src, dst = torch.nonzero(new_connection_matrix, as_tuple=True)
        g_new = dgl.graph((src, dst))
        for gnn in self.layers:
            h = gnn([g_new], h)
        return h