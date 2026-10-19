import math

import numpy as np


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
    """Гаусс с пиком 1 в center (sigma = диаметр / 6), перекрытия объединяются через max."""
    x0, y0 = center
    diam = 2 * radius + 1
    sigma = diam / 6
    d = np.arange(diam) - radius
    gauss = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma ** 2)).astype(heatmap.dtype)
    h, w = heatmap.shape
    left, right = max(0, x0 - radius), min(w, x0 + radius + 1)
    top, bottom = max(0, y0 - radius), min(h, y0 + radius + 1)
    np.maximum(heatmap[top:bottom, left:right],
               gauss[top - y0 + radius:bottom - y0 + radius, left - x0 + radius:right - x0 + radius],
               out=heatmap[top:bottom, left:right])


def encode_targets(boxes, labels, num_classes, out_h, out_w, stride):
    """Плотные цели CenterNet для одной картинки (считаются на CPU в DataLoader).

    boxes: (N, 4) xyxy в пикселях входа модели, labels: (N,). Возвращает float32 карты:
    hm (C, out_h, out_w) - гауссы в центрах, wh / off (2, out_h, out_w) - размер в ячейках и дробная часть центра,
    mask (out_h, out_w) - 1 в ячейках центров.
    """
    hm = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    wh = np.zeros((2, out_h, out_w), dtype=np.float32)
    off = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)
    for (x1, y1, x2, y2), cls in zip(np.asarray(boxes, dtype=np.float64), np.asarray(labels)):
        cls = int(cls)
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        if not (0 <= cls < num_classes and 0 <= cx < out_w and 0 <= cy < out_h):
            continue
        bw, bh = (x2 - x1) / stride, (y2 - y1) / stride
        ix, iy = int(cx), int(cy)
        _draw_gaussian(hm[cls], (ix, iy), _gaussian_radius((bh, bw)))
        mask[iy, ix] = 1
        wh[:, iy, ix] = bw, bh
        off[:, iy, ix] = cx - ix, cy - iy
    return {"hm": hm, "wh": wh, "off": off, "mask": mask}
