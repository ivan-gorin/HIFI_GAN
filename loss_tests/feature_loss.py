import torch
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser


class ModuleIter(nn.Module):

    def __init__(self):
        super(ModuleIter, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, true_feats, pred_feats):
        loss = 0
        for i in range(len(true_feats)):
            for j in range(len(true_feats[i])):
                loss += self.l1(true_feats[i][j], pred_feats[i][j])

        return loss * 2


class ModuleZip(nn.Module):

    def __init__(self):
        super(ModuleZip, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, true_feats, pred_feats):
        loss = 0
        for dt, dp in zip(true_feats, pred_feats):
            for lt, lp in zip(dt, dp):
                loss += self.l1(lt, lp)

        return loss * 2


def feature_loss(true_feats, pred_feats):
    loss = 0
    for i in range(len(true_feats)):
        for j in range(len(true_feats[i])):
            loss += torch.mean(torch.abs(true_feats[i][j] - pred_feats[i][j]))

    return loss * 2


def feature_loss2(true_feats, pred_feats):
    loss = 0
    for dt, dp in zip(true_feats, pred_feats):
        for lt, lp in zip(dt, dp):
            loss += torch.mean(torch.abs(lt - lp))

    return loss * 2


def main(length):
    x = [torch.randn((1000, 1000)).to('cuda') for i in range(10)]
    y = [torch.randn((1000, 1000)).to('cuda') for i in range(10)]
    l1 = ModuleIter().to('cuda')
    l2 = ModuleZip().to('cuda')
    loss = 0
    print('Module loss, indexing.')
    for _ in tqdm(range(length)):
        loss = l1(x, y)
    print('Module loss, zip.')
    for _ in tqdm(range(length)):
        loss = l2(x, y)
    print('function loss, indexing.')
    for _ in tqdm(range(length)):
        loss = feature_loss(x, y)
    print('function loss, zip.')
    for _ in tqdm(range(length)):
        loss = feature_loss2(x, y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('length', type=int, default=15)
    args = parser.parse_args()
    main(args.length)
