from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms.functional as TF
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

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


def build_loaders(config: dict, data_root: Path) -> tuple[DataLoader, DataLoader]:
    train_cfg = config.get("train", {})
    img_size = int(config.get("preprocess", {}).get("imgsz", 640))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))

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
        drop_last=True
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


def train_one_epoch(model, ema_model, loader, optimizer, device, epoch_idx=0, accumulation_steps=1, max_norm=10.0) -> float:
    model.train()
    model.backbone.eval()

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"train {epoch_idx}")

    optimizer.zero_grad(set_to_none=True)

    for i, (imgs, targets) in enumerate(pbar):
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        loss = model(imgs, targets)

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Обновляем EMA веса ТОЛЬКО после шага оптимизатора
            if ema_model is not None:
                ema_model.update_parameters(model)

        loss_value = loss.item()
        total_loss += loss_value
        pbar.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(eval_model, loader, device, num_classes, epoch_idx=0, topk=100, conf_threshold=0.001) -> dict[str, float]:
    eval_model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"valid {epoch_idx}")

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for imgs, targets in pbar:
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets_device = move_targets_to_device(targets, device)

        # Вычисление лосса (для статистики)
        loss = eval_model(imgs, targets_device)
        total_loss += loss.item()

        actual_model = eval_model.module if hasattr(eval_model, "module") else eval_model
        detections = actual_model.infer(imgs, topk=topk, conf_thres=conf_threshold)

        formatted_preds = []
        formatted_targets = []

        for b_idx in range(len(imgs)):
            det = detections[b_idx]
            if det.numel() > 0:
                formatted_preds.append(dict(
                    boxes=det[:, :4],
                    scores=det[:, 4],
                    labels=det[:, 5].long(),
                ))
            else:
                formatted_preds.append(dict(
                    boxes=torch.empty((0, 4)),
                    scores=torch.empty((0,)),
                    labels=torch.empty((0,), dtype=torch.long),
                ))

            tgt = targets[b_idx]
            formatted_targets.append(dict(
                boxes=tgt["boxes"].cpu(),
                labels=tgt["labels"].cpu(),
            ))

        metric.update(formatted_preds, formatted_targets)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    result = metric.compute()

    return {
        "loss": total_loss / max(len(loader), 1),
        "map50": result["map_50"].item(),
        "map50_95": result["map"].item(),
        "mar_100": result["mar_100"].item()
    }


def save_checkpoint(path: Path, model, ema_model, optimizer, scheduler, epoch: int, metrics: dict,
                    names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model else None,
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

    model = DinoV2CenterNet(num_classes=num_classes, img_stride=int(config.get("model",
                                                                               {}).get("img_stride", 4))).to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-6, 'weight_decay': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=max(1, int(train_cfg.get("epochs", 100))), eta_min=1e-6)

    output_dir = project_dir / config.get("outputs", {}).get("project", "runs") / config.get("outputs", {}).get("name",
                                                                                                                "exp")
    best_map50 = -1.0
    epochs = int(train_cfg.get("epochs", 100))

    batch_size = int(train_cfg.get("batch_size", 4))
    target_batch = 16
    accumulation_steps = max(1, target_batch // batch_size)

    print(f"Dataset: {data_root}")
    print(f"Classes ({num_classes}): {names}")
    print(f"Device: {device}")
    print(
        f"Batch size: {batch_size}, Accumulation steps: {accumulation_steps} (Effective batch: {batch_size * accumulation_steps})")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, ema_model, train_loader, optimizer,
                                     device, epoch_idx=epoch, accumulation_steps=accumulation_steps, max_norm=10.0)

        val_metrics = validate_one_epoch(ema_model, val_loader, device,
                                         num_classes, epoch_idx=epoch, topk=int(metrics_cfg.get("topk_inference", 100)),
                                         conf_threshold=float(metrics_cfg.get("eval_conf_threshold", 0.001)))
        scheduler.step()

        print(
            f"epoch={epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"map50={val_metrics['map50']:.4f} | "
            f"map50-95={val_metrics['map50_95']:.4f} | "
            f"mar100={val_metrics['mar_100']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, ema_model, optimizer, scheduler, epoch, val_metrics, names)
        if val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            save_checkpoint(output_dir / "best.pt", model, ema_model, optimizer, scheduler, epoch, val_metrics, names)


if __name__ == "__main__":
    main()
