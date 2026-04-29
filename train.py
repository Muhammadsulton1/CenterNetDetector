from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms.functional as TF
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import YOLODataset, collate_fn
from model import DinoV2CenterNet

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset(config: dict, project_dir: Path) -> tuple[Path, int, list[str]]:
    data_cfg = config.get("data", {})
    dataset_yaml = project_dir / data_cfg.get("dataset_yaml", "detection_dataset_yolo/dataset.yaml")
    dataset_info = read_yaml(dataset_yaml)

    root_override = data_cfg.get("root_override")
    if root_override:
        data_root = project_dir / root_override
    else:
        raw_root = Path(dataset_info.get("path", dataset_yaml.parent))
        data_root = raw_root if raw_root.is_absolute() else project_dir / raw_root

    names_raw = dataset_info.get("names", [])
    if isinstance(names_raw, dict):
        names = [names_raw[k] for k in sorted(names_raw, key=lambda x: int(x))]
    else:
        names = list(names_raw)

    num_classes = int(dataset_info.get("nc", len(names) or 1))
    return data_root, num_classes, names


def normalize_batch(imgs: torch.Tensor) -> torch.Tensor:
    return TF.normalize(imgs, IMNET_MEAN, IMNET_STD)


def move_targets_to_device(targets: Iterable[dict], device: torch.device) -> list[dict]:
    return [
        {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in target.items()}
        for target in targets
    ]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def average_precision(tp: torch.Tensor, fp: torch.Tensor, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1)

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
    return torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]).item()


def detection_metrics(
    detections: list[torch.Tensor],
    targets: list[dict],
    num_classes: int,
    iou_threshold: float = 0.5) -> dict[str, float]:
    aps, total_tp, total_fp, total_gt = [], 0.0, 0.0, 0

    for cls in range(num_classes):
        cls_preds = []
        gt_by_image = {}
        matched_by_image = {}
        num_gt = 0

        for image_idx, target in enumerate(targets):
            gt_mask = target["labels"].cpu() == cls
            gt_boxes = target["boxes"].cpu()[gt_mask]
            gt_by_image[image_idx] = gt_boxes
            matched_by_image[image_idx] = torch.zeros(len(gt_boxes), dtype=torch.bool)
            num_gt += len(gt_boxes)

            pred = detections[image_idx]
            if pred.numel() == 0:
                continue
            pred = pred[pred[:, 5].long() == cls]
            for row in pred:
                cls_preds.append((float(row[4]), image_idx, row[:4].cpu()))

        cls_preds.sort(key=lambda item: item[0], reverse=True)
        tp = torch.zeros(len(cls_preds))
        fp = torch.zeros(len(cls_preds))

        for pred_idx, (_, image_idx, pred_box) in enumerate(cls_preds):
            gt_boxes = gt_by_image[image_idx]
            if gt_boxes.numel() == 0:
                fp[pred_idx] = 1
                continue

            ious = box_iou(pred_box[None, :], gt_boxes).squeeze(0)
            best_iou, best_gt = torch.max(ious, dim=0)
            if best_iou >= iou_threshold and not matched_by_image[image_idx][best_gt]:
                tp[pred_idx] = 1
                matched_by_image[image_idx][best_gt] = True
            else:
                fp[pred_idx] = 1

        ap = average_precision(tp, fp, num_gt)
        if not torch.isnan(torch.tensor(ap)):
            aps.append(ap)
        total_tp += tp.sum().item()
        total_fp += fp.sum().item()
        total_gt += num_gt

    precision = total_tp / max(total_tp + total_fp, 1.0)
    recall = total_tp / max(total_gt, 1)
    map50 = sum(aps) / max(len(aps), 1)
    return {"map50": map50, "precision": precision, "recall": recall}


def build_loaders(config: dict, data_root: Path) -> tuple[DataLoader, DataLoader]:
    train_cfg = config.get("train", {})
    img_size = int(config.get("preprocess", {}).get("imgsz", 640))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_ds = YOLODataset(str(data_root), split="train", img_size=img_size)
    val_ds = YOLODataset(str(data_root), split="valid", img_size=img_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, epoch_idx=0, ema_alpha=0.97) -> float:
    model.train()
    ema, total_loss = None, 0.0
    pbar = tqdm(loader, desc=f"train {epoch_idx}")

    for imgs, targets in pbar:
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(imgs, targets)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        ema = loss_value if ema is None else ema_alpha * ema + (1 - ema_alpha) * loss_value
        pbar.set_postfix(loss=f"{ema:.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, num_classes, epoch_idx=0, topk=100, conf_threshold=0.001) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_detections, all_targets = [], []
    pbar = tqdm(loader, desc=f"valid {epoch_idx}")

    for imgs, targets in pbar:
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets_device = move_targets_to_device(targets, device)

        loss = model(imgs, targets_device)
        total_loss += loss.item()

        detections = model.infer(imgs, topk=topk, conf_thres=conf_threshold)
        all_detections.extend(detections)
        all_targets.extend(targets)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = detection_metrics(all_detections, all_targets, num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "names": names,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config = read_yaml(project_dir / args.config)
    train_cfg = config.get("train", {})
    metrics_cfg = config.get("metrics", {})

    requested_device = str(train_cfg.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    torch.manual_seed(int(train_cfg.get("seed", 42)))

    data_root, num_classes, names = resolve_dataset(config, project_dir)
    train_loader, val_loader = build_loaders(config, data_root)

    model = DinoV2CenterNet(num_classes=num_classes, img_stride=int(config.get("model", {}).get("img_stride", 4))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("scheduler_step_size", 5)),
        gamma=float(train_cfg.get("scheduler_gamma", 0.5)),
    )

    output_dir = project_dir / config.get("outputs", {}).get("project", "runs") / config.get("outputs", {}).get("name", "exp")
    best_map50 = -1.0
    epochs = int(train_cfg.get("epochs", 5))

    print(f"Dataset: {data_root}")
    print(f"Classes ({num_classes}): {names}")
    print(f"Device: {device}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch_idx=epoch,
            ema_alpha=float(train_cfg.get("ema_alpha", 0.97)),
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            device,
            num_classes,
            epoch_idx=epoch,
            topk=int(metrics_cfg.get("topk_inference", 100)),
            conf_threshold=float(metrics_cfg.get("eval_conf_threshold", 0.001)),
        )
        scheduler.step()

        print(
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"map50={val_metrics['map50']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, val_metrics, names)
        if val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, val_metrics, names)


if __name__ == "__main__":
    main()