from utils import ModelConfig

import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F


class PD(nn.Module):
    def __init__(self, period, config: ModelConfig):
        super(PD, self).__init__()
        self.period = period
        in_ch = [1, 32, 128, 512, 1024]
        out_ch = in_ch[1:] + [1024]
        self.list = nn.ModuleList([])
        for i in range(5):
            self.list.append(nn.Sequential(
                weight_norm(nn.Conv2d(in_ch[i], out_ch[i], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(config.leaky)
            ))
        self.list.append(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))

    def forward(self, x):

        pad = (-x.shape[-1]) % self.period
        if pad != 0:
            # pad if needed
            x = F.pad(x, (0, pad), 'reflect')

        # transform to 2d
        x = x.view(x.shape[0], x.shape[1], -1, self.period)

        res = []
        for conv in self.list:
            x = conv(x)
            res.append(x)

        # flatten score
        x = torch.flatten(x, 1, -1)
        # final, intermediate feats
        return x, res


class MPD(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(MPD, self).__init__()
        self.discriminators = nn.ModuleList([])
        for p in config.MPD_periods:
            self.discriminators.append(PD(p, config))

    def forward(self, x):
        fins = []
        inters = []
        for d in self.discriminators:
            fin, intermediate = d(x)
            fins.append(fin)
            inters.append(intermediate)

        return fins, inters
