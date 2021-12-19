from dataclasses import dataclass


@dataclass
class ModelConfig:
    mel_ch: int = 80
    hidden_ch: int = 512
    kernel_u = [16, 16, 4, 4]
    num_blocks: int = 3
    kernel_r = [3, 7, 11]
    dilation_r = [[[1, 1], [3, 1], [5, 1]]] * 3
    leaky: float = 0.1
    MPD_periods = [2, 3, 5, 7, 11]


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
