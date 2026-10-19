import torch
import torch.nn.functional as F


def _focal_loss(logits, gt, alpha=2.0, beta=4.0):
    """Сумма focal loss CenterNet по всем ячейкам (нормировка на число объектов - на стороне вызывающего).

    Принимает логиты и считает во float32 через logsigmoid: sigmoid во float16 / bfloat16 округляется до 1.0
    уже при логите ~9, и log(1 - p) с clamp терял градиент у уверенных ложных пиков.
    """
    logits = logits.float()
    p = logits.sigmoid()
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = -F.logsigmoid(logits) * (1 - p) ** alpha * pos
    neg_loss = -F.logsigmoid(-logits) * p ** alpha * (1 - gt) ** beta * neg
    return pos_loss.sum() + neg_loss.sum()
