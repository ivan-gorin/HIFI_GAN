from torch import nn
from torch.nn import functional as F


class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_spec, true_spec):
        if pred_spec.shape[-1] > true_spec.shape[-1]:
            pred_spec = F.pad(pred_spec, (0, true_spec.shape[-1] - pred_spec.shape[-1]))
        else:
            true_spec = F.pad(pred_spec, (0, pred_spec.shape[-1] - true_spec.shape[-1]))
        return self.l1(pred_spec, true_spec)

