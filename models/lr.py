#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lr.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/06 23:19:04
'''

# here put the import lib
import torch

from .layers import FeaturesLinear


class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))