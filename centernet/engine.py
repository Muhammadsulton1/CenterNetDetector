from __future__ import annotations

import contextlib
import copy
import csv
import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from .dataset import AUG_KEYS, YOLODataset, collate_fn, letterbox
from .model import CenterNetModel
from .utils import encode_targets

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DEFAULTS = read_yaml(Path(__file__).parent / "cfg" / "default.yaml")


def get_cfg(overrides: dict | None = None, base: dict | None = None) -> dict:
    """defaults < base (аргументы из чекпоинта) < overrides."""
    overrides = overrides or {}
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"неизвестные ключи: {sorted(unknown)}; допустимые: {sorted(DEFAULTS)}")
    return {**DEFAULTS, **(base or {}), **overrides}


def increment_path(path: Path, exist_ok: bool = False) -> Path:
    """runs/train/exp -> runs/train/exp2 -> exp3 ..."""
    if exist_ok or not path.exists():
        return path
    n = 2
    while Path(f"{path}{n}").exists():
        n += 1
    return Path(f"{path}{n}")


def resolve_dataset(data: str | Path) -> dict:
    """Читает dataset.yaml в формате Ultralytics: path, train, val, nc, names."""
    data_yaml = Path(data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    info = read_yaml(data_yaml)

    raw = Path(info.get("path", "."))
    root = raw
    if not raw.is_absolute():
        # относительно cwd, затем относительно папки yaml; иначе сама папка yaml
        root = next((p for p in (Path.cwd() / raw, data_yaml.parent / raw) if p.exists()), data_yaml.parent)

    def split_dir(key: str) -> Path:
        rel = info.get(key)
        if not rel and key == "val" and (root / "valid" / "images").exists():
            rel = "valid/images"
        if not isinstance(rel, str):
            raise ValueError(f"в {data_yaml} не задан путь {key!r}")
        return root / rel

    names_raw = info.get("names", [])
    if isinstance(names_raw, dict):
        names = [names_raw[k] for k in sorted(names_raw, key=int)]
    else:
        names = list(names_raw)
    return {
        "train": split_dir("train"),
        "val": split_dir("val"),
        "nc": int(info.get("nc", len(names) or 1)),
        "names": names,
    }


def normalize_batch(imgs: torch.Tensor) -> torch.Tensor:
    return TF.normalize(imgs, IMNET_MEAN, IMNET_STD)


def move_targets_to_device(targets: Iterable[dict], device: torch.device) -> list[dict]:
    return [
        {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in target.items()}
        for target in targets
    ]


def build_dataset(img_dir: Path, cfg: dict, train: bool = False, num_classes: int | None = None) -> YOLODataset:
    ds = YOLODataset(img_dir, img_size=int(cfg["imgsz"]), augment={k: cfg[k] for k in AUG_KEYS} if train else None,
                     num_classes=num_classes, stride=int(cfg["img_stride"]))
    ds.samples = ds.samples[: max(1, int(len(ds.samples) * float(cfg["fraction"])))]
    return ds


def build_loader(ds: YOLODataset, cfg: dict, train: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg["batch"]),
        shuffle=train,
        num_workers=int(cfg["workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=train,
    )


def find_batch_size(run_step, max_batch: int, start: int = 16, margin: float = 0.85) -> int:
    """Наибольший batch, для которого run_step(batch) не падает с OOM, умноженный на margin (запас памяти)."""
    def fits(b: int) -> bool:
        try:
            run_step(b)
            return True
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False

    lo, hi = 0, max_batch + 1  # lo помещается (0 - ничего), hi нет (max_batch + 1 - ещё не пробовали)
    b = max(1, min(start, max_batch))
    while hi - lo > 1:
        if fits(b):
            lo = b
        else:
            hi = b
        b = min(lo * 2, max_batch) if hi > max_batch else (lo + hi) // 2  # удвоение, затем бинарный поиск
    torch.cuda.empty_cache()  # кэш после проб фрагментирован: иначе настоящий шаг уйдёт за пределы VRAM
    return max(1, int(lo * margin))


@contextlib.contextmanager
def cuda_memory_cap(device: torch.device, fraction: float = 0.9):
    """Лимит аллокатора на время обучения. Кэш PyTorch бывает в 1.5 раза больше занятой памяти; без лимита он
    разрастается за пределы VRAM, и драйвер Windows уходит в общую память (итерации в разы медленнее)."""
    if device.type != "cuda":
        yield
        return
    idx = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_per_process_memory_fraction(fraction, idx)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, idx)


def trial_step(model: CenterNetModel, imgsz: int, device: torch.device, budget: int | None = None, amp: bool = False):
    """Шаг forward + backward на случайном батче, как в обучении.

    На CUDA аллокатор ограничен budget байтами (по умолчанию 90% VRAM), и OOM случается на самом лимите.
    Без этого драйвер Windows при нехватке памяти уходит в общую память и шаг тянется минутами.
    """
    cuda = device.type == "cuda"
    if cuda:
        idx = torch.cuda.current_device() if device.index is None else device.index
        total = torch.cuda.get_device_properties(idx).total_memory
        budget = int(0.9 * total) if budget is None else budget
    box = np.array([[imgsz * 0.25, imgsz * 0.25, imgsz * 0.75, imgsz * 0.75]], dtype=np.float32)
    dense = encode_targets(box, np.zeros(1, dtype=np.int64), model.head.num_classes, imgsz // model.stride,
                           imgsz // model.stride, model.stride)
    target = {k: torch.from_numpy(v).to(device) for k, v in dense.items()}

    def run_step(b: int) -> None:
        saved = copy.deepcopy(model.state_dict())  # BatchNorm не должен видеть случайные картинки
        if cuda:
            torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), idx)
        try:
            model.train()
            model.backbone.eval()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = model(torch.rand(b, 3, imgsz, imgsz, device=device), [target] * b)
            loss.backward()
        except RuntimeError as e:
            if "INT_MAX" not in str(e):
                raise
            raise torch.OutOfMemoryError(f"batch {b}: тензор больше лимита CUDA-ядра ({e})") from e
        finally:
            if cuda:
                torch.cuda.set_per_process_memory_fraction(1.0, idx)
            model.load_state_dict(saved)
            model.zero_grad(set_to_none=True)

    return run_step


def select_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(requested))


def lr_factor(step: int, total_steps: int, warmup_steps: int, lrf: float, cos: bool) -> float:
    """Множитель lr на шаге оптимизатора: линейный разогрев до 1, затем спад до lrf (косинус или линейно)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    p = min(max((step - warmup_steps) / max(total_steps - warmup_steps - 1, 1), 0.0), 1.0)
    return lrf + (1 - lrf) * (1 + math.cos(math.pi * p)) / 2 if cos else 1 - (1 - lrf) * p


def train_one_epoch(model, ema_model, loader, optimizer, device, epoch_idx=0, accumulation_steps=1, max_norm=10.0,
                    scheduler=None, amp=False, scaler=None) -> float:
    model.train()
    model.backbone.eval()

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"train {epoch_idx}")

    optimizer.zero_grad(set_to_none=True)
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=amp)

    for i, (imgs, targets) in enumerate(pbar):
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = model(imgs, targets)

        scaler.scale(loss / accumulation_steps).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)  # клиппинг по настоящим градиентам, а не масштабированным
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()  # lr меняется на каждом шаге оптимизатора

            # Обновляем EMA веса ТОЛЬКО после шага оптимизатора
            ema_model.update_parameters(model)

        loss_value = loss.item()
        total_loss += loss_value
        pbar.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / max(len(loader), 1)


PR_CONF, PR_IOU = 0.25, 0.5  # порог score и IoU для precision / recall (как в YOLO)


def count_matches(pred: dict, gt: dict, conf: float = PR_CONF, iou_thr: float = PR_IOU) -> tuple[int, int, int]:
    """(TP, FP, FN) для одной картинки: жадное сопоставление по убыванию score внутри класса."""
    keep = pred["scores"] >= conf
    order = pred["scores"][keep].argsort(descending=True)
    boxes, labels = pred["boxes"][keep][order], pred["labels"][keep][order]
    used = torch.zeros(len(gt["boxes"]), dtype=torch.bool)
    ious = box_iou(boxes, gt["boxes"]) if len(boxes) and len(gt["boxes"]) else None
    tp = 0
    for i in range(len(boxes)):
        if ious is None:
            break
        candidates = (ious[i] >= iou_thr) & (gt["labels"] == labels[i]) & ~used
        if candidates.any():
            used[torch.where(candidates, ious[i], torch.zeros(())).argmax()] = True
            tp += 1
    return tp, len(boxes) - tp, len(gt["boxes"]) - tp


@torch.no_grad()
def validate_one_epoch(eval_model, loader, device, desc="valid", topk=100, conf_threshold=0.001,
                       per_class=False, amp=False) -> dict:
    eval_model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=per_class)
    metric50 = (MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
                if per_class else None)
    tp = fp = fn = 0
    actual_model = eval_model.module if hasattr(eval_model, "module") else eval_model

    for imgs, targets in pbar:
        imgs = normalize_batch(imgs).to(device, non_blocking=True)
        targets_device = move_targets_to_device(targets, device)

        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = eval_model(imgs, targets_device)  # лосс для статистики
            detections = actual_model.infer(imgs, topk=topk, conf_thres=conf_threshold)
        total_loss += loss.item()

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
        if metric50 is not None:
            metric50.update(formatted_preds, formatted_targets)
        for pred, gt in zip(formatted_preds, formatted_targets):
            m_tp, m_fp, m_fn = count_matches(pred, gt)
            tp, fp, fn = tp + m_tp, fp + m_fp, fn + m_fn
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    result = metric.compute()
    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "map50": result["map_50"].item(),
        "map50_95": result["map"].item(),
        "mar_100": result["mar_100"].item(),
    }
    if per_class:
        classes, ap = (torch.atleast_1d(result[k]).tolist() for k in ("classes", "map_per_class"))
        ap50 = torch.atleast_1d(metric50.compute()["map_per_class"]).tolist()
        # AP = -1: у класса нет объектов в разметке, считать нечего
        metrics["per_class"] = {int(c): {"ap50": a50, "ap50_95": a} for c, a50, a in zip(classes, ap50, ap) if a >= 0}
    return metrics


def format_per_class(per_class: dict, names: list[str]) -> str:
    rows = [f"{'class':<16}{'AP50':>8}{'AP50-95':>9}"]
    rows += [f"{names[c] if c < len(names) else c:<16}{v['ap50']:>8.3f}{v['ap50_95']:>9.3f}"
             for c, v in sorted(per_class.items())]
    return "\n".join(rows)


def preprocess_image(image_path: Path, img_size: int, device: torch.device):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    resized, ratio, (pad_x, pad_y) = letterbox(rgb, img_size)
    img = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    img = normalize_batch(img).unsqueeze(0).to(device)

    meta = {"orig_w": orig_w, "orig_h": orig_h, "ratio": ratio, "pad_x": pad_x, "pad_y": pad_y}
    return img, bgr, meta


def to_original(det: torch.Tensor, meta: dict) -> torch.Tensor:
    """Боксы из letterbox-координат в координаты исходного изображения."""
    out = det.clone()
    out[:, [0, 2]] = ((out[:, [0, 2]] - meta["pad_x"]) / meta["ratio"]).clamp(0, meta["orig_w"] - 1)
    out[:, [1, 3]] = ((out[:, [1, 3]] - meta["pad_y"]) / meta["ratio"]).clamp(0, meta["orig_h"] - 1)
    return out


def draw_detections(image_bgr, detections: torch.Tensor, names: list[str]):
    color = (0, 255, 0)
    for x1, y1, x2, y2, score, cls_id in detections.tolist():
        x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
        label = names[cls_id] if cls_id < len(names) else str(cls_id)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, f"{label} {score:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return image_bgr


def collect_images(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def ema_with_warmup(decay: float, tau: float = 2000.0):
    """multi_avg_fn для AveragedModel: decay растёт от 0 до заданного (как в Ultralytics), иначе на коротких
    прогонах в EMA остаются почти необученные начальные веса (0.999 ** 530 = 59% после 5 эпох)."""
    def fn(ema_params, params, num_averaged):
        return get_ema_multi_avg_fn(decay * (1 - math.exp(-float(num_averaged) / tau)))(ema_params, params, num_averaged)

    return fn


def transfer_weights(src: dict, model: torch.nn.Module) -> tuple[int, int]:
    """Загружает из src все тензоры с тем же именем и формой (при другом числе классов не совпадёт
    только выход heatmap). Возвращает (перенесено, всего)."""
    own = model.state_dict()
    matched = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(matched, strict=False)
    return len(matched), len(own)


class EarlyStopper:
    """True, если fitness не росла `patience` эпох подряд (0 - выключено). Равенство не считается ростом."""

    def __init__(self, patience: int):
        self.patience, self.best, self.best_epoch = patience, float("-inf"), 0

    def __call__(self, epoch: int, fitness: float) -> bool:
        if fitness > self.best:
            self.best, self.best_epoch = fitness, epoch
        return self.patience > 0 and epoch - self.best_epoch >= self.patience


class CenterNet:
    """CenterNet + DINOv2 в стиле ultralytics.YOLO.

    CenterNet("facebook/dinov2-small") - новая модель (строится в train(), когда известно число классов);
    CenterNet("runs/train/exp/best.pt") - обученная модель, параметры восстанавливаются из чекпоинта.
    """

    def __init__(self, model: str = DEFAULTS["model"]):
        self.backbone_name = str(model)
        self.model: CenterNetModel | None = None
        self.args: dict = {}
        self.ckpt_path: str | None = None
        if self.backbone_name.endswith(".pt"):
            self._load(self.backbone_name)

    def _load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.ckpt_path = str(path)
        self.args = ckpt["args"]
        self.backbone_name = self.args["model"]
        self.model = CenterNetModel(self.backbone_name, self.args["nc"], self.args["img_stride"],
                                    self.args.get("feature_layers", 1), pretrained=False)
        self.model.load_state_dict(ckpt["model"])

    @property
    def names(self) -> list[str]:
        return self.args.get("names", [])

    def _base(self) -> dict:
        return {k: self.args[k] for k in ("data", "imgsz", "img_stride", "feature_layers") if self.args.get(k) is not None}

    def _require_model(self) -> CenterNetModel:
        if self.model is None:
            raise ValueError("модель не обучена: укажите model=path/to/best.pt или сначала вызовите train()")
        return self.model

    RESUME_OVERRIDES = ("device", "workers", "batch")  # что можно поменять при resume (как в Ultralytics)

    def train(self, **overrides) -> dict:
        ckpt, resume_path = None, None
        if overrides.get("resume"):
            resume_path = self.ckpt_path if overrides["resume"] is True else str(overrides["resume"])
            if not resume_path:
                raise ValueError("resume=True: загрузите CenterNet('runs/train/exp/last.pt') или укажите resume=path/to/last.pt")
            changed = sorted(set(overrides) - {"resume", *self.RESUME_OVERRIDES})
            if changed:
                raise ValueError(f"при resume аргументы берутся из чекпоинта, менять можно только "
                                 f"{', '.join(self.RESUME_OVERRIDES)}; получено: {changed}")
            ckpt = torch.load(resume_path, map_location="cpu")
            if "train_model" not in ckpt:
                raise ValueError(f"{resume_path}: в чекпоинте нет состояния обучения, resume возможен только с last.pt")
            base = {k: v for k, v in ckpt["args"].items() if k in DEFAULTS}
            cfg = get_cfg({k: v for k, v in overrides.items() if k != "resume"}, base=base)
            if ckpt.get("finished"):
                print(f"Обучение уже завершено ({ckpt['epoch']}/{cfg['epochs']} эпох): {resume_path}")
                return ckpt.get("best_metrics", ckpt["metrics"])
            self.backbone_name = cfg["model"]
        else:
            cfg = get_cfg(overrides, base=self._base())
            cfg["model"] = self.backbone_name
        if not cfg["data"]:
            raise ValueError("укажите data=path/to/dataset.yaml")
        ds = resolve_dataset(cfg["data"])
        nc, names = ds["nc"], ds["names"]

        device = select_device(cfg["device"])
        amp = bool(cfg["amp"]) and device.type == "cuda"  # на CPU float16-autocast не ускоряет
        torch.manual_seed(int(cfg["seed"]))

        arch = (self.backbone_name, nc, int(cfg["img_stride"]), int(cfg["feature_layers"]))
        if ckpt is not None:  # resume: продолжаем с не-EMA весов последней эпохи
            model = CenterNetModel(*arch, pretrained=False)
            model.load_state_dict(ckpt["train_model"])
        elif self.model is not None:  # дообучение с чекпоинта: переносим всё, что совпадает по форме
            model = CenterNetModel(*arch, pretrained=False)
            n, total = transfer_weights(self.model.state_dict(), model)
            print(f"Transferred {n}/{total} items from {self.ckpt_path}")
        else:
            model = CenterNetModel(*arch)
        self.model = model = model.to(device)
        cfg["img_stride"] = model.stride

        train_ds = build_dataset(ds["train"], cfg, train=True, num_classes=nc)
        freeze_epochs = int(cfg["freeze_epochs"])
        model.set_backbone_trainable(freeze_epochs >= 0)  # батч подбираем под худший случай: бэкбон тоже учится
        if int(cfg["batch"]) < 1:  # batch=-1: подбор под память
            if device.type == "cuda":
                cfg["batch"] = find_batch_size(trial_step(model, int(cfg["imgsz"]), device, amp=amp), len(train_ds))
            else:
                cfg["batch"] = int(cfg["nbs"])
        cfg["batch"] = min(int(cfg["batch"]), len(train_ds))  # иначе drop_last оставит 0 батчей
        print(f"Batch size: {cfg['batch']}")
        train_loader = build_loader(train_ds, cfg, train=True)
        val_loader = build_loader(build_dataset(ds["val"], cfg, num_classes=nc), cfg, train=False)

        # use_buffers=True: иначе running stats BatchNorm в EMA-копии остаются начальными
        ema_model = AveragedModel(model, multi_avg_fn=ema_with_warmup(float(cfg["ema_decay"])),
                                  use_buffers=True)
        wd = float(cfg["weight_decay"])
        optimizer = torch.optim.AdamW([
            {"params": list(model.backbone_parameters()), "lr": float(cfg["lr_backbone"]), "weight_decay": wd},
            {"params": model.new_parameters(), "lr": float(cfg["lr0"]), "weight_decay": wd},
        ])
        epochs = int(cfg["epochs"])
        accumulation = max(1, int(cfg["nbs"]) // int(cfg["batch"]))
        steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_steps, warmup_steps = epochs * steps_per_epoch, round(float(cfg["warmup_epochs"]) * steps_per_epoch)
        lrf, cos_lr = float(cfg["lrf"]), bool(cfg["cos_lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, lrf, cos_lr))
        scaler = torch.amp.GradScaler(device.type, enabled=amp)

        best, best_metrics, start_epoch = -1.0, {}, 1
        stopper = EarlyStopper(int(cfg["patience"]))
        if ckpt is not None:
            ema_model.module.load_state_dict(ckpt["model"])
            ema_model.n_averaged.fill_(int(ckpt["ema_updates"]))
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if amp and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            best, best_metrics, start_epoch = ckpt["best"], ckpt["best_metrics"], ckpt["epoch"] + 1
            stopper.best, stopper.best_epoch = ckpt["best"], ckpt["best_epoch"]
            save_dir = Path(resume_path).parent
            print(f"Resuming {save_dir} с эпохи {start_epoch}/{epochs}")
        else:
            save_dir = increment_path(Path(cfg["project"]) / "train" / cfg["name"], bool(cfg["exist_ok"]))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_args = {**cfg, "nc": nc, "names": names}
        (save_dir / "args.yaml").write_text(yaml.safe_dump(ckpt_args, allow_unicode=True), encoding="utf-8")

        print(f"Dataset: {cfg['data']} | classes ({nc}): {names}")
        print(f"AMP: {amp} | Device: {device} | batch {cfg['batch']} x accumulation {accumulation} | save_dir: {save_dir}")

        def save(path: Path, epoch: int, metrics: dict, full: bool, finished: bool = False) -> None:
            state = {
                "epoch": epoch,
                "model": ema_model.module.state_dict(),  # EMA-веса: по ним считаются метрики и идёт инференс
                "metrics": metrics,
                "args": ckpt_args,
            }
            if full:  # всё, что нужно для resume
                state.update(train_model=model.state_dict(), ema_updates=int(ema_model.n_averaged),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             scaler=scaler.state_dict(), best=best, best_epoch=stopper.best_epoch,
                             best_metrics=best_metrics, finished=finished)
            torch.save(state, path)

        columns = ["epoch", "train_loss", "val_loss", "precision", "recall", "map50", "map50_95", "mar_100",
                   "lr_head", "lr_backbone", "time"]
        mode = "a" if ckpt is not None else "w"
        mosaic_closed = False
        with cuda_memory_cap(device), (save_dir / "results.csv").open(mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if ckpt is None:
                writer.writerow(columns)
            for epoch in range(start_epoch, epochs + 1):
                started = time.time()
                close = int(cfg["close_mosaic"])
                if close > 0 and not mosaic_closed and epoch >= max(1, epochs - close + 1):
                    print(f"Closing mosaic: последние {cfg['close_mosaic']} эпох без mosaic / mixup / copy_paste")
                    train_ds.close_mosaic()  # воркеры DataLoader пересоздаются каждую эпоху и увидят изменение
                    mosaic_closed = True
                model.set_backbone_trainable(0 <= freeze_epochs < epoch)
                train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, device, epoch_idx=epoch,
                                             accumulation_steps=accumulation, scheduler=scheduler, amp=amp,
                                             scaler=scaler)
                m = validate_one_epoch(ema_model, val_loader, device, desc=f"valid {epoch}",
                                       topk=int(cfg["topk"]), conf_threshold=0.001, amp=amp)
                lr_backbone, lr_head = (g["lr"] for g in optimizer.param_groups)  # lr на конец эпохи

                m["train_loss"], m["val_loss"], m["epoch"], m["time"] = train_loss, m["loss"], epoch, time.time() - started
                m["lr_head"], m["lr_backbone"] = lr_head, lr_backbone
                writer.writerow([m[c] for c in columns])
                f.flush()

                is_best = m["map50_95"] > best
                if is_best:
                    best, best_metrics = m["map50_95"], m
                stop = stopper(epoch, m["map50_95"])
                print(f"epoch={epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={m['loss']:.4f} | "
                      f"P={m['precision']:.3f} | R={m['recall']:.3f} | map50={m['map50']:.4f} | "
                      f"map50-95={m['map50_95']:.4f} | mar100={m['mar_100']:.4f} | lr={lr_head:.1e}/{lr_backbone:.1e} | "
                      f"{m['time']:.0f}s" + (" *" if is_best else ""))

                save(save_dir / "last.pt", epoch, m, full=True, finished=stop or epoch == epochs)
                if is_best:
                    save(save_dir / "best.pt", epoch, m, full=False)
                if stop:
                    print(f"Early stopping: map50-95 не росла {stopper.patience} эпох, лучшая эпоха {stopper.best_epoch}")
                    break

        self._load(str(save_dir / "best.pt"))
        print(f"Best: epoch {best_metrics['epoch']} ({save_dir / 'best.pt'}), метрики по классам на валидации:")
        self.val(**{k: cfg[k] for k in ("data", "batch", "imgsz", "workers", "device", "fraction", "topk")})
        return best_metrics

    def val(self, **overrides) -> dict:
        model = self._require_model()
        overrides.setdefault("conf", 0.001)
        cfg = get_cfg(overrides, base=self._base())
        if not cfg["data"]:
            raise ValueError("укажите data=path/to/dataset.yaml")
        device = select_device(cfg["device"])
        if int(cfg["batch"]) < 1:
            cfg["batch"] = int(cfg["nbs"])
        cfg["img_stride"] = model.stride
        loader = build_loader(build_dataset(resolve_dataset(cfg["data"])["val"], cfg, num_classes=model.head.num_classes),
                              cfg, train=False)
        metrics = validate_one_epoch(model.to(device), loader, device, topk=int(cfg["topk"]),
                                     conf_threshold=float(cfg["conf"]), per_class=True)
        print(" | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "per_class"))
        print(format_per_class(metrics["per_class"], self.names))
        return {**metrics, "per_class": {self.names[c] if c < len(self.names) else c: v
                                         for c, v in metrics["per_class"].items()}}

    def predict(self, source: str | Path | None = None, **overrides) -> list[torch.Tensor]:
        """Возвращает по тензору [N, 6] (x1, y1, x2, y2, score, class) на картинку, в координатах оригинала."""
        model = self._require_model()
        if source is not None:
            overrides["source"] = source
        cfg = get_cfg(overrides, base=self._base())
        if not cfg["source"]:
            raise ValueError("укажите source=path/to/image_or_dir")
        device = select_device(cfg["device"])
        model.to(device).eval()

        paths = collect_images(Path(cfg["source"]))
        save_dir = None
        if cfg["save"]:
            save_dir = increment_path(Path(cfg["project"]) / "predict" / cfg["name"], bool(cfg["exist_ok"]))
            save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(paths)} images | device: {device} | classes: {self.names}")

        results = []
        for path in paths:
            img, bgr, meta = preprocess_image(path, int(cfg["imgsz"]), device)
            det = to_original(model.infer(img, topk=int(cfg["topk"]), conf_thres=float(cfg["conf"]))[0], meta)
            results.append(det)
            msg = f"{path.name}: {len(det)} detections"
            if save_dir is not None:
                cv2.imwrite(str(save_dir / path.name), draw_detections(bgr, det, self.names))
                msg += f" -> {save_dir / path.name}"
            print(msg)
        return results
