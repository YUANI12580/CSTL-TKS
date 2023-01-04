import torch
import torch.nn as nn
import torch.nn.init as init

from .basic_blocks import BasicConv2d, SetBlock
from .cstl import ATA, SSFL
from .tks import MSTE_TKS


class GaitNet(nn.Module):
    def __init__(self, hidden_dim, class_num, part_num, div):   # 256, 73, 64, 16
        super(GaitNet, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]

        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))

        self.multi_scale = MSTE_TKS(_channels[2], _channels[2], part_num)

        for name, module in self.multi_scale._modules.items():
            print(name, " : ", module)

        self.adaptive_aggregation = ATA(_channels[2], _channels[2], part_num, depth=1, num_head=4,\
                                                                 decay=div, kernel_size=3, stride=1)
        self.salient_learning = SSFL(_channels[2], _channels[2], 4, part_num, class_num, topk_num=1)

        self.fc_bin = nn.Parameter(init.xavier_uniform_(torch.zeros(part_num, _channels[2]*3, hidden_dim)))

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho):
        n = silho.size(0)
        if len(silho.size()) == 3:
            silho.unsqueeze(0)

        x = silho.unsqueeze(2)
        del silho

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        x = x.max(-1)[0] + x.mean(-1)

        t_f, t_s, t_l = self.multi_scale(x.permute(0, 3, 2, 1).contiguous())

        aggregated_feature = self.adaptive_aggregation(t_f, t_s, t_l)

        part_classification, weighted_part_feature, selected_part_feature = self.salient_learning(t_f, t_s, t_l)

        feature = torch.cat([aggregated_feature, weighted_part_feature, selected_part_feature], -1)
        feature = feature.matmul(self.fc_bin)
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, part_classification
