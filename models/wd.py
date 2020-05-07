#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wd.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/09 11:07:49
'''

# here put the import lib
import torch
import torch.nn as nn
from .layers import FeaturesLinear, FeaturesEmbedding, MultilayerPerception

class WideAndDeepModel(nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self, filed_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(filed_dims)
        self.embedding = FeaturesEmbedding(filed_dims, embed_dim)
        self.embed_output_dim = len(filed_dims) * embed_dim
        self.mlp = MultilayerPerception(self.embed_output_dim, mlp_dims, dropout)
    
    def forward(self, x):
        """
        :param x : Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        # Wide part + Deep part
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))