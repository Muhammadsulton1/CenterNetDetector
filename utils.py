import math, torch

def _gaussian_radius(box, min_overlap=0.7):
    w, h = box
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))) / 2
    return int(max(0, min(r1, r2, r3)))


def _draw_gaussian(heatmap, center, radius):
    x, y = center
    diam = 2 * radius + 1
    sigma = diam / 6
    gauss = torch.exp(
        -((torch.arange(diam) - radius)[:, None] ** 2 +
          (torch.arange(diam) - radius)[None, :] ** 2) / (2 * sigma ** 2)
    ).to(device=heatmap.device, dtype=heatmap.dtype)
    x0, y0 = int(x), int(y)
    h, w = heatmap.shape[-2:]
    left, right = max(0, x0 - radius), min(w, x0 + radius + 1)
    top, bottom = max(0, y0 - radius), min(h, y0 + radius + 1)
    heatmap[..., top:bottom, left:right] = torch.maximum(
        heatmap[..., top:bottom, left:right], gauss[(top - y0 + radius):(bottom - y0 + radius),
        (left - x0 + radius):(right - x0 + radius)]
    )


def _focal_loss(pred, gt, alpha=2.0, beta=4.0):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    pos_loss = -torch.log(pred.clamp(1e-6)) * (1 - pred) ** alpha * pos
    neg_loss = -torch.log((1 - pred).clamp(1e-6)) * pred ** alpha * (1 - gt) ** beta * neg
    return (pos_loss.sum() + neg_loss.sum()) / torch.clamp(pos.sum(), min=1.0)
