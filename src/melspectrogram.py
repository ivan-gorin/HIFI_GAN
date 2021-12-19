import torch
from torch import nn
import torchaudio
from torch.nn import functional as F
from typing import Dict, Union

MelConfig = {
    'sample_rate': 22050,
    'win_length': 1024,
    'hop_length': 256,
    'n_fft': 1024,
    'f_min': 0,
    'f_max': 8000,
    'n_mels': 80,
    'power': 1.0,
    'center': False,
    'pad_mode': 'reflect',
    'pad': -11.5129251
}


class MelSpectrogram(nn.Module):

    def __init__(self, config: Dict[str, Union[int, float]]):
        """
        sample_rate, win_length, hop_length
        n_fft
        f_min
        f_max
        n_mels
        """
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(**config)

        self.pad_size = (config['n_fft'] - config['hop_length']) // 2

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        audio = F.pad(audio, (self.pad_size, self.pad_size), mode="reflect")
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
