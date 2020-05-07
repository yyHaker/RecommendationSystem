#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/06 22:42:30
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeaturesLinear(nn.Module):
    """Features Linear Layer"""
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.emb = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        # calc offset to index emb
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.emb(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """Features Embedding layer"""
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        # calc offset to index emb
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        -----
        output: (batch_size, num_fields, embed_dim)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    """Factorization Machine Layer"""
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix