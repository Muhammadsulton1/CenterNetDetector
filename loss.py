import torch


def _focal_loss(pred, gt, alpha=2.0, beta=4.0):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    pos_loss = -torch.log(pred.clamp(1e-6)) * (1 - pred) ** alpha * pos
    neg_loss = -torch.log((1 - pred).clamp(1e-6)) * pred ** alpha * (1 - gt) ** beta * neg
    return (pos_loss.sum() + neg_loss.sum()) / torch.clamp(pos.sum(), min=1.0)
