import torchaudio
import torch


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, cut=8192):
        super().__init__(root=root)
        self.cut = cut

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        if waveform.shape[1] <= self.cut:
            return waveform, waveform_length
        start = torch.randint(low=0, high=waveform.shape[1] - self.cut, size=(1,))
        return waveform[:, start:start + self.cut], waveform_length
