from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        """ We use GroupNorm (group = 1) to approximate LayerNorm
        for [N, C, H, W] layout"""
        super(GroupNorm, self).__init__(num_groups, num_channels)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Atcm(nn.Module):

    def __init__(self, dim, seb_ratio=16, norm_layer=partial(GroupNorm, num_groups=1), shift_beta=1 / 3):
        super(Atcm, self).__init__()
        self.shift_beta = shift_beta
        self.dim = dim
        self.seb_ratio = seb_ratio
        self.norm2 = norm_layer(dim)
        self.se_block = SE_Block(inchannel=dim, ratio=seb_ratio)

    def forward(self, x):
        shortcut = x
        x = self.pct(x, self.shift_beta)
        x = self.norm2(x)
        x = self.se_block(shortcut + x)
        return x

    @staticmethod
    def pct(x, beta):
        n, c, h, w = x.shape
        dp = int(c * beta) // n
        y = torch.zeros_like(x)
        for i in range(n):
            for j in range(n):
                y[j:j + 1][:, i * dp:(i + 1) * dp, :, :] = x[i:i + 1][:, j * dp:(j + 1) * dp, :, :]
        y[:, dp * n:, :, :] = x[:, dp * n:, :, :]
        return y

    @staticmethod
    def shift_tsm_t6(x, fold_div=18):
        n, c, h, w = x.size()
        x = x.view(1, n, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left 1
        out[:, :-2, fold: 2 * fold] = x[:, 2:, fold: 2 * fold]  # # shift left 2
        out[:, :-3, 2 * fold: 3 * fold] = x[:, 3:, 2 * fold: 3 * fold]  # # shift left 3
        out[:, 1:, 3 * fold: 4 * fold] = x[:, :-1, 3 * fold: 4 * fold]  # shift right 1
        out[:, 2:, 4 * fold: 5 * fold] = x[:, :-2, 4 * fold: 5 * fold]  # shift right 2
        out[:, 3:, 5 * fold: 6 * fold] = x[:, :-3, 5 * fold: 6 * fold]  # shift right 3
        out[:, :, 6 * fold:] = x[:, :, 6 * fold:]  # not shift
        return out.view(n, c, h, w)


class BasicLayer(nn.Module):

    def __init__(self, dim, depth, seb_ratio=16, norm_layer='GN1', shift_beta=1 / 3):
        super(BasicLayer, self).__init__()
        assert norm_layer in ('GN1', 'BN')
        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'GN1':
            norm_layer = partial(GroupNorm, num_groups=1)
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList([
            Atcm(dim=dim, seb_ratio=seb_ratio, norm_layer=norm_layer, shift_beta=shift_beta)
            for i in range(depth)
        ])
        self.apply(self._init_weights)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FusionNet(nn.Module):
    def __init__(self, dim, depth=4, seb_ratio=16, norm_layer='GN1', shift_beta=1 / 3) -> None:
        super().__init__()
        self.shiftfuse = BasicLayer(dim, depth, seb_ratio, norm_layer, shift_beta)

    def forward(self, d8_fm):
        fusion_fm = self.shiftfuse(d8_fm)
        return fusion_fm
