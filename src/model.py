import torch
from torch import nn, Tensor
import math
from dataclasses import dataclass
from torch.nn.utils import weight_norm


@dataclass
class ModelConfig:
    mel_ch: int = 80
    hidden_ch: int = 512
    kernel_u = [16, 16, 4, 4]
    num_blocks: int = 3
    kernel_r = [3, 7, 11]
    dilation_r = [[[1, 1], [3, 1], [5, 1]]] * 3
    leaky: float = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations, config: ModelConfig):
        super(ResBlock, self).__init__()

        self.list = nn.ModuleList([])
        for dilation in dilations:
            self.list.append(nn.Sequential(
                nn.LeakyReLU(config.leaky),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, dilation=dilation[0], padding='same')),
                nn.LeakyReLU(config.leaky),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, dilation=dilation[1], padding='same'))
            ))

    def forward(self, x):
        for conv in self.list:
            x = x + conv(x)
        return x


class MRF(nn.Module):
    def __init__(self, ch, config: ModelConfig):
        super(MRF, self).__init__()
        self.list = nn.ModuleList([ResBlock(ch, k, d, config) for k, d in zip(config.kernel_r, config.dilation_r)])

    def forward(self, x):
        res = 0
        for block in self.list:
            res = res + block(x)
        return res / len(self.list)


class Generator(nn.Module):

    def __init__(self, config: ModelConfig):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv1d(config.mel_ch, config.hidden_ch, kernel_size=7, padding='same')

        block_list = []
        cur_ch = config.hidden_ch
        for k in config.kernel_u:
            block_list.append(nn.Sequential(
                nn.LeakyReLU(config.leaky),
                nn.ConvTranspose1d(cur_ch, cur_ch // 2, kernel_size=k, stride=k // 2),
                MRF(cur_ch // 2, config)
            ))
            cur_ch //= 2

        self.net = nn.Sequential(*block_list)

        self.leaky = nn.LeakyReLU(config.leaky)
        self.conv2 = nn.Conv1d(cur_ch, 1, kernel_size=7, padding='same')
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.net(x)
        x = self.leaky(x)
        x = self.conv2(x)
        return self.tan(x).squeeze(1)

