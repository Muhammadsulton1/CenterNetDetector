from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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


class YOLODataset(Dataset):
    """
    Ожидаемая структура::

        data_root/
            train/images/*.jpg   valid/images/*.jpg
            train/labels/*.txt   valid/labels/*.txt

        Также поддерживается layout:
            images/train/*.jpg   images/val/*.jpg
            labels/train/*.txt   labels/val/*.txt
                  # формат строки: cls cx cy w h  (нормированные [0,1])
    """

    def __init__(self, data_root: str, split: str = "train", img_size: int = 640):
        self.img_size = img_size
        root = Path(data_root)
        split_aliases = [split]
        if split == "val":
            split_aliases.append("valid")

        candidates = []
        for split_name in split_aliases:
            candidates.extend([
                (root / split_name / "images", root / split_name / "labels"),
                (root / "images" / split_name, root / "labels" / split_name),
            ])

        img_dir, lbl_dir = next(
            ((img, lbl) for img, lbl in candidates if img.exists()),
            candidates[0],
        )
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        self.samples: List[Tuple[Path, Path]] = [
            (p, lbl_dir / (p.stem + ".txt"))
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in _IMG_EXTS
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img, ratio, (pad_x, pad_y) = letterbox(img, self.img_size)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cls = int(p[0])
                cx_n, cy_n, bw_n, bh_n = map(float, p[1:5])
                cx = cx_n * w0 * ratio + pad_x
                cy = cy_n * h0 * ratio + pad_y
                bw = bw_n * w0 * ratio
                bh = bh_n * h0 * ratio
                x1 = float(np.clip(cx - bw / 2, 0, self.img_size))
                y1 = float(np.clip(cy - bh / 2, 0, self.img_size))
                x2 = float(np.clip(cx + bw / 2, 0, self.img_size))
                y2 = float(np.clip(cy + bh / 2, 0, self.img_size))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        return img_t, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": idx,
            "orig_size": (h0, w0),
        }


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)
