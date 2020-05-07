#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ffm.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/07 17:21:16
'''

# here put the import lib
import torch
import torch.nn as nn

from .layers import FeaturesLinear, FieldAwareFactorizationMachine


class FieldAwareFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.
    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))