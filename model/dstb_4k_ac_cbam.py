import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class DenseSblock(nn.Module):

    def __init__(self, in_channel, reduce_ratio=8) -> None:
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

        self.cab = CBAM(in_channel, 2, 5)
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

    def __init__(self, in_channel, reduce_ratio=8) -> None:
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

        self.cab = CBAM3D(in_channel, 2, 5)
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


class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度上的平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度上的最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将平均池化和最大池化结果在通道维度上拼接
        x = self.conv1(x)  # 通过卷积层
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = self.n_channels_in // self.reduction_ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.n_channels_in, self.middle_layer_size, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.middle_layer_size, self.n_channels_in, 1, bias=False)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool(x)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = self.act(pool_sum)

        return sig_pool


class CBAM3D(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention3D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention3D(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention3D, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = n_channels_in // reduction_ratio
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.n_channels_in, self.middle_layer_size, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(self.middle_layer_size, self.n_channels_in, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool(x)
        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)
        pool_sum = avg_pool_bck + max_pool_bck
        sig_pool = self.act(pool_sum)
        return sig_pool


if __name__ == '__main__':
    x = torch.randn(5, 768, 44, 88)
    dstb = DensenSTBlayer(768, 8, 2)
    out = dstb(x)
    print(out.shape)