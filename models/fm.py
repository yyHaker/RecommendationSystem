#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fm.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/06 22:37:46
'''

# here put the import lib
import torch
import torch.nn as nn

from .layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class FM(nn.Module):
    """Pure Factorization Machine."""
    def __init__(self, n, k):
        super(FM, self).__init__()
        self.n = n  # num_fileds (all items and features)
        self.k = k  # feature emb dim
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def forward(self, x):
        # x [batch, n]
        linear_part = self.linear(x)
        # matmul [batch, n] * [n, k]
        inter_part1 = torch.mm(x, self.v.t())  
        square_of_sum = torch.sum(torch.pow(inter_part1, 2))

        # matmul [batch, n]^2 * [n, k]^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())
        sum_of_square = torch.sum(inter_part2)

        # out_size = [batch, 1]
        output = linear_part + 0.5 * (square_of_sum - sum_of_square)
        return torch.sigmoid(x.squeeze(1))