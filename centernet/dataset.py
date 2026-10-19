from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import encode_targets
from .augment import augment_hsv, copy_paste, flip_lr, flip_ud, mixup, mosaic, random_perspective

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = new_size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_size - nw, new_size - nh
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def labels_dir(img_dir: Path) -> Path:
    """Правило YOLO: последний компонент 'images' в пути заменяется на 'labels'."""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir


AUG_KEYS = ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear", "perspective",
            "hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr")


class YOLODataset(Dataset):
    """
    img_dir - каталог с картинками (train/images, images/train, ...).
    Разметка ищется в соседнем каталоге, где 'images' заменено на 'labels'.
    Формат строки: cls cx cy w h (нормированные [0,1]).
    augment - словарь с ключами AUG_KEYS (вероятности и амплитуды как в Ultralytics) или None для валидации.
    num_classes, stride - если заданы, в цели добавляются плотные карты CenterNet (hm, wh, off, mask),
    закодированные здесь, на CPU в воркерах DataLoader, а не в модели на GPU.
    """

    def __init__(self, img_dir: str | Path, img_size: int = 640, augment: dict | None = None,
                 num_classes: int | None = None, stride: int = 4):
        self.img_size = img_size
        self.num_classes, self.stride = num_classes, stride
        self.augment = {k: float(augment[k]) for k in AUG_KEYS} if augment else None
        img_dir = Path(img_dir)
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        lbl_dir = labels_dir(img_dir)
        self.samples: List[Tuple[Path, Path]] = [
            (p, lbl_dir / (p.stem + ".txt"))
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in _IMG_EXTS
        ]

    def __len__(self):
        return len(self.samples)

    def close_mosaic(self) -> None:
        """Последние эпохи без mosaic / mixup / copy_paste, как close_mosaic в Ultralytics."""
        if self.augment:
            self.augment.update(mosaic=0.0, mixup=0.0, copy_paste=0.0)

    def _load(self, idx):
        """Исходная RGB картинка, боксы xyxy в её пикселях, метки."""
        img_path, lbl_path = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        rows = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                p = line.strip().split()
                if len(p) >= 5:
                    rows.append([float(v) for v in p[:5]])
        rows = np.array(rows, dtype=np.float32).reshape(-1, 5)
        cx, cy, bw, bh = rows[:, 1] * w0, rows[:, 2] * h0, rows[:, 3] * w0, rows[:, 4] * h0
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        return img, boxes.astype(np.float32), rows[:, 0].astype(np.int64)

    def _resized(self, idx):
        """Длинная сторона = img_size, без полей (кусок для mosaic)."""
        img, boxes, labels = self._load(idx)
        r = self.img_size / max(img.shape[:2])
        if r != 1:
            img = cv2.resize(img, (round(img.shape[1] * r), round(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        return img, boxes * r, labels

    def _letterboxed(self, idx):
        img, boxes, labels = self._load(idx)
        img, ratio, (pad_x, pad_y) = letterbox(img, self.img_size)
        boxes = (boxes * ratio + np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)).clip(0, self.img_size)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return img, boxes[keep], labels[keep]

    def _base(self, idx):
        """mosaic (4 картинки) или одна картинка, затем геометрия: поворот, сдвиг, масштаб, скос, перспектива."""
        a, s = self.augment, self.img_size
        geometry = dict(degrees=a["degrees"], translate=a["translate"], scale=a["scale"], shear=a["shear"],
                        perspective=a["perspective"])
        if np.random.rand() < a["mosaic"]:
            parts = [self._resized(i) for i in [idx] + list(np.random.randint(0, len(self), 3))]
            img, boxes, labels = mosaic(*zip(*parts), s)
            return random_perspective(img, boxes, labels, **geometry, border=-s // 2)
        return random_perspective(*self._letterboxed(idx), **geometry)

    def __getitem__(self, idx):
        if self.augment is None:
            img, boxes, labels = self._letterboxed(idx)
        else:
            a = self.augment
            img, boxes, labels = self._base(idx)
            if np.random.rand() < a["mixup"]:
                img, boxes, labels = mixup(img, boxes, labels, *self._base(np.random.randint(0, len(self))))
            if a["copy_paste"] > 0:
                img, boxes, labels = copy_paste(img, boxes, labels, *self._letterboxed(np.random.randint(0, len(self))),
                                                p=a["copy_paste"])
            img = augment_hsv(img, a["hsv_h"], a["hsv_s"], a["hsv_v"])
            if np.random.rand() < a["flipud"]:
                img, boxes = flip_ud(img, boxes)
            if np.random.rand() < a["fliplr"]:
                img, boxes = flip_lr(img, boxes)

        img_t = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels), "image_id": idx}
        if self.num_classes is not None:
            out = self.img_size // self.stride
            dense = encode_targets(boxes, labels, self.num_classes, out, out, self.stride)
            target.update({k: torch.from_numpy(v) for k, v in dense.items()})
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)
