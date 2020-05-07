#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nffm.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/16 22:23:39
'''

# here put the import lib
import torch
import torch.nn as nn
from .layers import FeaturesEmbedding, FeaturesLinear, MultilayerPerception, FieldAwareFactorizationMachine

class NeuralFieldAwareFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of NeuralFFM.
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        # self.fm = FactorizationMachine(reduce_sum=True)
        # self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)
        self.ffm_output_dim = embed_dim
        self.mlp = MultilayerPerception(self.ffm_output_dim, mlp_dims, dropout)
    
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_xs = self.ffm(x) # (batch_size, num_combines, embed_dim)
        # ffm term
        ffm_term = torch.sum(torch.sum(embed_xs, dim=1), dim=1, keepdim=True) # (batch_size, 1)
        # feature interaction
        feature_interaction = self.mlp(embed_xs.view(-1, self.ffm_output_dim)) # (batch_size * num_combines, 1)
        x = self.linear(x) + ffm_term + torch.sum(feature_interaction.view(-1, embed_xs.size(1)), dim=1)
        return torch.sigmoid(x.squeeze(1))
