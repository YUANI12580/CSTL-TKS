#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
"""
@Project: Python Program
@File: bicnet_tks.py
@Author: Yuan
@Date: 2022/12/11 19:52
"""
import torch
from .func import *
import torch.nn as nn


class TKS_Two(nn.Module):
    def __init__(self, in_channel, out_channel, part_num):
        super(TKS_Two, self).__init__()
        # ====================卷积方式====================
        # 基本卷积
        # self.conv1 = nn.Conv1d(in_channel * part_num, out_channel * part_num, kernel_size=3,
        #                        padding=1, groups=part_num, bias=False)
        # self.conv2 = nn.Conv1d(in_channel * part_num, out_channel * part_num, kernel_size=5,
        #                        padding=2, groups=part_num, bias=False)
        # 空洞卷积
        self.conv1 = nn.Conv1d(in_channel * part_num, out_channel * part_num, kernel_size=3,
                               padding=1, groups=part_num, bias=False)
        self.conv2 = nn.Conv1d(in_channel * part_num, out_channel * part_num, kernel_size=3,
                               padding=2, dilation=2, groups=part_num, bias=False)
        # self.conv3 = nn.Conv1d(in_channel * part_num, out_channel * part_num, kernel_size=3,
        #                        padding=3, dilation=3, groups=part_num, bias=False)

        # ====================时间池化方式====================
        # GMP
        self.TP_GMP = torch.max
        # GAP
        # self.TP_GAP = torch.mean

        hidden_dim = int((in_channel * part_num) // 4)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel * part_num, hidden_dim, kernel_size=1, groups=part_num, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channel * part_num * 2, kernel_size=1, groups=part_num, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm1d(out_channel * part_num)

    def forward(self, x):
        n, p, c, s = x.size()

        y1 = self.conv1(x.view(n, -1, s)).view(n, p, c, s)
        y2 = self.conv2(x.view(n, -1, s)).view(n, p, c, s)
        # y3 = self.conv3(x.view(n, -1, s)).view(n, p, c, s)

        # GMP
        u = self.TP_GMP((y1 + y2), dim=-1)[0].unsqueeze(-1).view(n, -1, 1)
        # u = self.TP_GMP(y3, dim=-1)[0].unsqueeze(-1).view(n, -1, 1)
        # GAP
        # u = self.TP_GAP((y1 + y2), dim=-1).unsqueeze(-1).view(n, -1, 1)

        w = self.mlp(u)
        w = w.view(n, p, 2, c, 1)
        w = self.sigmoid(w)

        y = torch.stack((y1, y2), 2)
        # y = y3.unsqueeze(2)
        z = (y * w).sum(2)
        # ====================是否有激活方式====================
        z = self.BN(z.view(n, -1, s)).view(n, p, c, s) + x
        # z = self.BN(z.view(n, -1, s)).view(n, p, c, s)

        return z


class MSTE_TKS(nn.Module):
    def __init__(self, in_planes, out_planes, part_num):
        super(MSTE_TKS, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num

        self.short_term = TKS_Two(in_planes, out_planes, part_num)
        self.score = mlp_sigmoid(in_planes * part_num, in_planes * part_num, 1, groups=part_num)

    def get_frame_level(self, x):
        return x

    def get_short_term(self, x):
        return self.short_term(x)

    def get_long_term(self, x):
        n, p, c, s = x.size()
        pred_score = self.score(x.view(n, -1, s)).view(n, p, c, s)
        long_term_feature = x.mul(pred_score).sum(-1).div(pred_score.sum(-1))
        long_term_feature = long_term_feature.unsqueeze(3).repeat(1, 1, 1, s)

        return long_term_feature

    def forward(self, x):
        return self.get_frame_level(x), self.get_short_term(x), self.get_long_term(x)
