#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dfm.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/10 16:28:58
'''

# here put the import lib
import torch
import torch.nn as nn
from .layers import FeaturesEmbedding, FeaturesLinear, MultilayerPerception, FactorizationMachine

class DeepFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultilayerPerception(self.embed_output_dim, mlp_dims, dropout)
    
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

