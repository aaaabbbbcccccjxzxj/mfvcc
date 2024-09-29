import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MixUpsampleLayer(BaseModule):
    def __init__(self, input_chans=768, output_chans=192):
        super().__init__()
        self.interpolation = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels=input_chans, out_channels=output_chans, kernel_size=1)
        self.trans_conv = nn.ConvTranspose2d(in_channels=input_chans, out_channels=output_chans,
                                             kernel_size=2, stride=2)
        self.apply(self.init_weights)

    def forward(self, x):
        return self.trans_conv(x) + self.conv(self.interpolation(x))


class MixUpsampleLayerConvFirst(BaseModule):
    def __init__(self, input_chans=768, output_chans=192):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_chans, out_channels=output_chans, kernel_size=1)
        self.interpolation = nn.UpsamplingBilinear2d(scale_factor=2)
        self.trans_conv = nn.ConvTranspose2d(in_channels=output_chans, out_channels=output_chans,
                                             kernel_size=2, stride=2)
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        return self.trans_conv(x) + self.interpolation(x)


class UpsampleModule(BaseModule):
    def __init__(self, input_chans=768, c_down_ratio=2, mode='mix_conv_first'):
        super().__init__()
        assert mode in ['normal', 'mix', 'mix_conv_first']
        upsample_layer = {
            'normal': lambda: nn.ConvTranspose2d(in_channels=input_chans, out_channels=input_chans // c_down_ratio,
                                                 kernel_size=2, stride=2),
            'mix': lambda: MixUpsampleLayer(input_chans=input_chans, output_chans=input_chans // c_down_ratio),
            'mix_conv_first': lambda: MixUpsampleLayerConvFirst(input_chans=input_chans,
                                                                output_chans=input_chans // c_down_ratio),
        }[mode]()

        self.integrated_layer = nn.Sequential(
            LayerNorm(input_chans, eps=1e-6, data_format='channels_first'),
            upsample_layer,
            nn.GELU(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.integrated_layer(x)
