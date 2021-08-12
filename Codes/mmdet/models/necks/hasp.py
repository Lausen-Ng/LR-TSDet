import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, constant_init
from mmcv.runner import auto_fp16
import torch


class HASP(nn.Module):
    def __init__(self,
                 in_channel=256,
                 mid_channel=128,
                 dila_rates=(2, 4, 6, 8),  # 1,2.3.4
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
                 ):
        super(HASP, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.dila_rates = dila_rates

        self.conv_1 = ConvModule(
            in_channels=self.in_channel,
            out_channels=self.mid_channel,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            inplace=True)

        self.group_blocks = nn.ModuleList()
        for i, dila_rate in enumerate(self.dila_rates):
            group_block = HACB(in_channel=self.mid_channel,
                                     split_num=4,
                                     dila_rate=dila_rate)
            self.group_blocks.append(group_block)

        self.conv_2 = ConvModule(
            in_channels=4 * mid_channel,
            out_channels=in_channel,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=None,
            inplace=False)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1.0)

    def forward(self, x):

        identity = x
        inputs = self.conv_1(x)

        mid0 = self.group_blocks[0](inputs)
        mid1 = self.group_blocks[1](inputs)
        mid2 = self.group_blocks[2](inputs)
        mid3 = self.group_blocks[3](inputs)

        hasp_out = torch.cat([mid0, mid1, mid2, mid3], dim=1)
        hasp_out = self.conv_2(hasp_out)
        out = hasp_out + identity
        out = self.relu(out)

        return out


class HACB(nn.Module):
    def __init__(self,
                 in_channel=128,
                 split_num=4,
                 dila_rate=1,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
                 ):
        super(HACB, self).__init__()

        self.in_channel = in_channel
        self.split_num = split_num
        self.dila_rate = dila_rate

        self.split_channel = self.in_channel // self.split_num
        self.group_convs = nn.ModuleList()
        for i in range(self.split_num):
            group_conv = ConvModule(
                in_channels=self.split_channel,
                out_channels=self.split_channel,
                kernel_size=3,
                padding=dila_rate,
                dilation=dila_rate,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=True)
            self.group_convs.append(group_conv)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1.0)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        spx = torch.split(x, self.split_channel, dim=1)
        sp = self.group_convs[0](spx[0].contiguous())
        out = sp

        for i in range(1, self.split_num):
            sp = sp + spx[i]
            sp = self.group_convs[i](sp.contiguous())
            out = torch.cat([out, sp], dim=1)

        out = self.channel_shuffle(out, 4)

        return out

