from .utils import ModelConfig

import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm


class SD(torch.nn.Module):
    def __init__(self, config: ModelConfig, use_spectral=False):
        super(SD, self).__init__()
        if use_spectral:
            norm_func = spectral_norm
        else:
            norm_func = weight_norm
        in_ch = [1, 128, 128, 256, 512, 1024, 1024]
        out_ch = in_ch[1:] + [1024]
        kernels = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1]
        paddings = [7, 20, 20, 20, 20, 20, 2]
        self.list = nn.ModuleList([])
        for i in range(7):
            self.list.append(nn.Sequential(
                norm_func(
                    nn.Conv1d(in_channels=in_ch[i], out_channels=out_ch[i], kernel_size=kernels[i], stride=strides[i],
                              groups=groups[i], padding=paddings[i])),
                nn.LeakyReLU(config.leaky)
            ))
        self.list.append(norm_func(nn.Conv1d(1024, 1, 3, 1, padding=1)))

    def forward(self, x):
        res = []
        for conv in self.list:
            x = conv(x)
            res.append(x)
        x = torch.flatten(x, 1, -1)

        return x, res


class MSD(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(MSD, self).__init__()
        self.discriminators = nn.ModuleList([
            SD(config, True),
            SD(config),
            SD(config),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        fins = []
        inters = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            fin, intermediate = d(x)
            fins.append(fin)
            inters.append(intermediate)

        return fins, inters
