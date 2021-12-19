from torch import nn
from torch.nn import functional as F
import torch


def spectrogram_loss(spec, pred_spec):
    return F.l1_loss(spec, pred_spec) * 45


def feature_loss(true_feats, pred_feats):
    loss = 0
    for t_feat, p_feat in zip(true_feats, pred_feats):
        for tb, pb in zip(t_feat, p_feat):
            loss += (tb - pb).abs().mean()

    return loss * 2


class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, spec, pred_spec, mpd_pred_feats, mpd_true_feats, msd_pred_feats, msd_true_feats, mpd_pred,
                msd_pred):
        loss = spectrogram_loss(spec, pred_spec) + feature_loss(mpd_true_feats, mpd_pred_feats) + \
               feature_loss(msd_true_feats, msd_pred_feats)

        for i in mpd_pred:
            loss += (1 - i).square().mean()
        for i in msd_pred:
            loss += (1 - i).square().mean()

        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, mpd_pred, mpd_true, msd_pred, msd_true):
        loss = 0

        for i in range(len(mpd_true)):
            loss += torch.mean(torch.square(mpd_true[i] - 1)) + torch.mean(torch.square(mpd_pred[i]))
        for i in range(len(msd_true)):
            loss += torch.mean(torch.square(msd_true[i] - 1)) + torch.mean(torch.square(msd_pred[i]))

        return loss
