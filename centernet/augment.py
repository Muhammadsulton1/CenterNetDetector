"""Аугментации детекции в стиле Ultralytics.

Все функции работают с RGB uint8 картинкой (H, W, 3), боксами float32 (N, 4) в пикселях xyxy и метками int64 (N,).
Случайность берётся из np.random: DataLoader сам даёт каждому воркеру свой seed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FILL = (114, 114, 114)


def random_perspective(img, boxes, labels, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                       border=0):
    """Поворот, сдвиг, масштаб, скос и перспектива одной матрицей. border < 0 обрезает (после mosaic 2s -> s)."""
    h, w = img.shape[:2]
    H, W = h + 2 * border, w + 2 * border

    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2  # центр картинки в начало координат
    P = np.eye(3)
    P[2, 0], P[2, 1] = np.random.uniform(-perspective, perspective, 2)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((0, 0), np.random.uniform(-degrees, degrees),
                                    np.random.uniform(1 - scale, 1 + scale))
    S = np.eye(3)
    S[0, 1], S[1, 0] = np.tan(np.radians(np.random.uniform(-shear, shear, 2)))
    T = np.eye(3)
    T[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * W
    T[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * H
    M = T @ S @ R @ P @ C

    if border == 0 and np.allclose(M, np.eye(3)):
        return img, boxes, labels
    if perspective:
        img = cv2.warpPerspective(img, M, (W, H), borderValue=FILL)
    else:
        img = cv2.warpAffine(img, M[:2], (W, H), borderValue=FILL)
    if not len(boxes):
        return img, boxes, labels

    # 4 угла каждого бокса -> новые габариты (прямые переходят в прямые, крайние точки - углы)
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
    xy = np.c_[corners, np.ones(len(corners))] @ M.T
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(-1, 4, 2)
    new = np.c_[xy[..., 0].min(1), xy[..., 1].min(1), xy[..., 0].max(1), xy[..., 1].max(1)]
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, W)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, H)

    keep = box_candidates(boxes, new, R[0, 0] ** 2 + R[0, 1] ** 2)
    return img, new[keep].astype(np.float32), labels[keep]


def box_candidates(old, new, scale2, min_side=2, min_area_ratio=0.1, max_ratio=100):
    """Отбрасывает боксы, от которых после трансформации осталась щель или почти ничего не видно."""
    w0, h0 = old[:, 2] - old[:, 0], old[:, 3] - old[:, 1]
    w1, h1 = new[:, 2] - new[:, 0], new[:, 3] - new[:, 1]
    ar = np.maximum(w1 / (h1 + 1e-16), h1 / (w1 + 1e-16))
    return (w1 > min_side) & (h1 > min_side) & (w1 * h1 / (w0 * h0 * scale2 + 1e-16) > min_area_ratio) & (ar < max_ratio)


def augment_hsv(img, hgain=0.0, sgain=0.0, vgain=0.0):
    """Случайные множители оттенка, насыщенности и яркости в [1 - gain, 1 + gain]."""
    if not (hgain or sgain or vgain):
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flip_lr(img, boxes):
    w = img.shape[1]
    boxes = boxes.copy()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return np.ascontiguousarray(img[:, ::-1]), boxes


def flip_ud(img, boxes):
    h = img.shape[0]
    boxes = boxes.copy()
    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return np.ascontiguousarray(img[::-1]), boxes


def mosaic(images, boxes_list, labels_list, s):
    """4 картинки (длинная сторона ~s) вокруг случайного центра на холсте 2s x 2s."""
    canvas = np.full((2 * s, 2 * s, 3), FILL[0], dtype=np.uint8)
    yc, xc = (int(np.random.uniform(s / 2, 3 * s / 2)) for _ in range(2))
    out_boxes, out_labels = [], []
    for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
        h, w = img.shape[:2]
        if i == 0:  # слева сверху: правый нижний угол картинки в центре
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        elif i == 1:  # справа сверху
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b = 0, h - (y2a - y1a)
        elif i == 2:  # слева снизу
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b = w - (x2a - x1a), 0
        else:  # справа снизу
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b = 0, 0
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
        shifted = boxes + np.array([x1a - x1b, y1a - y1b] * 2, dtype=np.float32)
        out_boxes.append(shifted)
        out_labels.append(labels)

    boxes = np.concatenate(out_boxes).reshape(-1, 4)
    labels = np.concatenate(out_labels).astype(np.int64)
    boxes = boxes.clip(0, 2 * s)
    keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    return canvas, boxes[keep].astype(np.float32), labels[keep]


def mixup(img1, boxes1, labels1, img2, boxes2, labels2):
    r = np.random.beta(32.0, 32.0)  # около 0.5
    img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
    return img, np.concatenate([boxes1, boxes2]).reshape(-1, 4), np.concatenate([labels1, labels2])


def copy_paste(img, boxes, labels, src, src_boxes, src_labels, p=0.0, max_ioa=0.3, tries=10):
    """Без масок сегментации: каждый объект источника с вероятностью p вставляется прямоугольной вырезкой
    в случайное место, если закрывает уже имеющиеся объекты не больше чем на max_ioa."""
    if p <= 0 or not len(src_boxes):
        return img, boxes, labels
    img = img.copy()
    H, W = img.shape[:2]
    boxes, labels = list(boxes), list(labels)
    for box, label in zip(src_boxes, src_labels):
        if np.random.rand() >= p:
            continue
        x1, y1 = int(max(box[0], 0)), int(max(box[1], 0))
        x2, y2 = int(min(box[2], src.shape[1])), int(min(box[3], src.shape[0]))
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2 or w >= W or h >= H:
            continue
        for _ in range(tries):
            nx, ny = np.random.randint(0, W - w + 1), np.random.randint(0, H - h + 1)
            cand = np.array([nx, ny, nx + w, ny + h], dtype=np.float32)
            if all(_ioa(b, cand) < max_ioa for b in boxes):
                img[ny:ny + h, nx:nx + w] = src[y1:y2, x1:x2]
                boxes.append(cand)
                labels.append(label)
                break
    return img, np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def _ioa(box, cover):
    """Доля площади box, закрытая прямоугольником cover."""
    iw = max(0.0, min(box[2], cover[2]) - max(box[0], cover[0]))
    ih = max(0.0, min(box[3], cover[3]) - max(box[1], cover[1]))
    return iw * ih / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
