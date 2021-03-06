import torch.nn as nn
import torch

Huber = nn.SmoothL1Loss().cuda()

def delta_loss(pred_azi, pred_ele, target, bin):
    # compute the ground truth delta value according to angle value and bin size
    target_delta = ((target % bin) / bin) - 0.5

    # compute the delta prediction in the ground truth bin
    target_label = (target // bin).long()
    delta_azi = pred_azi[torch.arange(pred_azi.size(0)), target_label[:, 0]].tanh() / 2
    delta_ele = pred_ele[torch.arange(pred_ele.size(0)), target_label[:, 1]].tanh() / 2
    pred_delta = torch.cat((delta_azi.unsqueeze(1), delta_ele.unsqueeze(1)), 1)

    return Huber(5. * pred_delta, 5. * target_delta)


class DeltaLoss(nn.Module):
    def __init__(self, bin):
        super(DeltaLoss, self).__init__()
        self.__bin__ = bin
        return

    def forward(self, pred_azi, pred_ele, target):
        return delta_loss(pred_azi, pred_ele, target, self.__bin__)
