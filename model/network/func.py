#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
"""
@Project: Python Program
@File: func.py
@Author: Yuan
@Date: 2022/12/10 20:49
"""
import torch.nn as nn


def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)


def mlp_sigmoid(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, in_planes // 16, kernel_size, **kwargs),
                         nn.BatchNorm1d(in_planes // 16),
                         nn.LeakyReLU(inplace=True),
                         conv1d(in_planes // 16, out_planes, kernel_size, **kwargs),
                         nn.Sigmoid())


def conv_bn(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, out_planes, kernel_size, **kwargs),
                         nn.BatchNorm1d(out_planes))