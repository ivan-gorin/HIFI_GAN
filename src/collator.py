from typing import Tuple, Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return {"waveform": waveform, "waveform_length": waveform_length}
