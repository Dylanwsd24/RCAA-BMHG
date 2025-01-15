import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch


class Intra_order(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(Intra_order, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)
        if self.Bias is not None:
            self.Bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        output = torch.spmm(adj, torch.spmm(inputs, self.Weight))
        if self.Bias is not None:
            out = output + self.Bias
        else:
            out = output
        return out


class GCN(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(GCN, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', True)
        self.reset_parameters()
        self.non_linear = nn.Tanh()
    def reset_parameters(self,):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)
        if self.Bias is not None:
            self.Bias.data.uniform_(-stdv, stdv)
    def forward(self, inputs, adj):
        output = self.non_linear(torch.spmm(adj, self.non_linear(torch.spmm(inputs, self.Weight))))
        if self.Bias is not None:
            output = output + self.Bias
        return output


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, inputs, adj):
        # 计算每个节点的邻居数量
        degrees = adj.sum(dim=1, keepdim=True)
        # 避免除以零
        degrees = degrees + (degrees == 0).float()
        # 平均池化操作
        pooled_h = torch.spmm(adj, inputs) / degrees
        return pooled_h


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, inputs, adj):
        # 扩展邻接矩阵以适应广播
        adj_expanded = adj.unsqueeze(-1)

        # 使用邻接矩阵掩码来选择邻居特征
        masked_inputs = inputs.unsqueeze(1) * adj_expanded

        # 对每个节点的邻居特征进行最大池化
        pooled_h = masked_inputs.max(dim=1)[0]

        return pooled_h





