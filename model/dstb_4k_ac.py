import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class DenseSblock(nn.Module):

    def __init__(self, in_channel, reduce_ratio=4) -> None:
        super().__init__()

        # 简化连接
        self.convp1 = nn.Conv2d(in_channel, in_channel // 2, 1)
        self.convp2 = nn.Conv2d(in_channel // reduce_ratio * 1 + in_channel, in_channel // 2, 1)
        self.convp3 = nn.Conv2d(in_channel // reduce_ratio * 2, in_channel // 2, 1)
        self.convp4 = nn.Conv2d(in_channel // reduce_ratio * 2, in_channel // 2, 1)
        self.convp5 = nn.Conv2d(in_channel // reduce_ratio * 2, in_channel, 1)

        self.conv_3 = nn.Conv2d(in_channel // 2, in_channel // reduce_ratio, 3, dilation=1, padding='same')
        self.conv_5 = nn.Conv2d(in_channel // 2, in_channel // reduce_ratio, 3, dilation=2, padding='same')
        self.conv_7 = nn.Conv2d(in_channel // 2, in_channel // reduce_ratio, 4, dilation=2, padding='same')
        self.conv_9 = nn.Conv2d(in_channel // 2, in_channel // reduce_ratio, 5, dilation=2, padding='same')

        self.cab = SE_Block(in_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        # 简化连接
        x1 = self.act(self.conv_3(self.act(self.convp1(x))))
        x2 = torch.cat((x, x1), dim=1)
        x3 = self.act(self.conv_5(self.act(self.convp2(x2))))
        x4 = torch.cat((x1, x3), dim=1)
        x5 = self.act(self.conv_7(self.act(self.convp3(x4))))
        x6 = torch.cat((x3, x5), dim=1)
        x7 = self.act(self.conv_9(self.act(self.convp4(x6))))
        x8 = torch.cat((x5, x7), dim=1)
        x_end = self.act(self.convp5(x8))
        return self.cab(x_end)


class DenseTblock(nn.Module):

    def __init__(self, in_channel, reduce_ratio=4) -> None:
        super().__init__()

        # 简化连接
        self.convp1 = nn.Conv3d(in_channel, in_channel // 2, (1, 1, 1))
        self.convp2 = nn.Conv3d(in_channel // reduce_ratio * 1 + in_channel, in_channel // 2, (1, 1, 1))
        self.convp3 = nn.Conv3d(in_channel // reduce_ratio * 2, in_channel // 2, (1, 1, 1))
        self.convp4 = nn.Conv3d(in_channel // reduce_ratio * 2, in_channel // 2, (1, 1, 1))
        self.convp5 = nn.Conv3d(in_channel // reduce_ratio * 2, in_channel, (1, 1, 1))

        self.conv_3 = nn.Conv3d(in_channel // 2, in_channel // reduce_ratio, (3, 1, 1), dilation=1, padding='same')
        self.conv_5 = nn.Conv3d(in_channel // 2, in_channel // reduce_ratio, (3, 1, 1), dilation=2, padding='same')
        self.conv_7 = nn.Conv3d(in_channel // 2, in_channel // reduce_ratio, (4, 1, 1), dilation=2, padding='same')
        self.conv_9 = nn.Conv3d(in_channel // 2, in_channel // reduce_ratio, (5, 1, 1), dilation=2, padding='same')

        self.cab = SE_Block3D(in_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)
        # 简化连接
        x1 = self.act(self.conv_3(self.act(self.convp1(x))))
        x2 = torch.cat((x, x1), dim=1)
        x3 = self.act(self.conv_5(self.act(self.convp2(x2))))
        x4 = torch.cat((x1, x3), dim=1)
        x5 = self.act(self.conv_7(self.act(self.convp3(x4))))
        x6 = torch.cat((x3, x5), dim=1)
        x7 = self.act(self.conv_9(self.act(self.convp4(x6))))
        x8 = torch.cat((x5, x7), dim=1)
        x_end = self.act(self.convp5(x8))
        x_end = self.cab(x_end)
        return x_end.squeeze(0).permute(1, 0, 2, 3)


class DenseSTblock(nn.Module):

    def __init__(self, in_channel, reduce_ratio) -> None:
        super().__init__()
        self.dsb = DenseSblock(in_channel, reduce_ratio)
        self.dtb = DenseTblock(in_channel, reduce_ratio)

    def forward(self, x):
        x = self.dsb(x)
        x = self.dtb(x)
        return x


class DensenSTBlayer(nn.Module):

    def __init__(self, in_channel, reduce_ratio, dstb_num=2) -> None:
        super().__init__()
        self.block_layer = nn.ModuleList()
        for i_layer in range(dstb_num):
            layer = DenseSTblock(in_channel, reduce_ratio)
            self.block_layer.append(layer)

        self.apply(self._init_weights)

    def forward(self, fm):
        for layer in self.block_layer:
            fm = layer(fm)
        return fm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_Block3D(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block3D, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(5, 768, 44, 88)
    dstb = DensenSTBlayer(768, 8, 2)
    out = dstb(x)
    print(out.shape)
