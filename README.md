# CenterNetDetector

Object detection training with CenterNet heads on **YOLO-format** datasets (Ultralytics-style `dataset.yaml` + `.txt` labels). Backbones via **timm** (CNN) or **transformers** (ViT/DINO-class). Training and inference are driven by YAML config (similar workflow to Ultralytics: one CLI + edits in `configs/`).

## Install

```
pip install -r requirements.txt
```

Run commands from the project directory so paths like `configs/default.yaml` resolve.

## Dataset layout

Your `dataset.yaml`:

- `path` — dataset root (images + labels tree). If it points to another PC, override with `data.root_override` in the training YAML.
- `train:` / `val:` — folders under root with images (e.g. `train/images`).
- Labels live under the parallel `labels/` tree (`train/labels/` with `.txt` per image, same relative path).

Each label line: `class_id cx cy w h` (normalized YOLO format).

The default training config assumes [detection_dataset_yolo/dataset.yaml](detection_dataset_yolo/dataset.yaml) and sets `root_override` to `detection_dataset_yolo` so images resolve locally.

## Train

```
python -m src.train --cfg configs/default.yaml --data detection_dataset_yolo/dataset.yaml
```

Optional flags: `--device cpu`, `--epochs 10`, `--seed 42`.

Artifacts under `runs/<name>_<timestamp>/`:

- `best.pt` — highest validation metric (`train.monitor_metric`, default mean mAP@[0.5:0.95]); `meta.val_metrics` logged each epoch when that score improves. Holds `ema_model` when weight EMA is enabled.
- `last.pt` — most recent epoch
- Checkpoints contain `model`, optional `ema_model`, `optimizer`, nested `config`, `meta`

- `metrics.json` — mAP@0.50 and mAP averaged over configurable IoUs (defaults to COCO [0.5:0.05:0.95])
- Per-class PNGs: Precision–Recall curve (`pr_curve_class_*.png`) and Precision/Recall versus confidence (`precision_recall_vs_conf_class_*.png`)
- `train_loss.png`, plus `config_used.yaml` snapshot

## Inference

```
python -m src.predict --weights runs/exp_YYYYMMDD_HHMMSS/best.pt --source path/to/img_or_folder --out predictions --ema
```

Use `--ema` to load smoothed weights from `ema_model` when the checkpoint includes them.

### Augmentations (train only)

See `augment` in [configs/default.yaml](configs/default.yaml):

- Random grayscale, color jitter
- Random resized crop (scaled relative to fixed `imgsz`, boxes updated via torchvision v2)
- Batch **CutMix** (paste cropped region from another image in the batch, merge intersecting GT boxes into the pasted patch)

Disable all training augmentations with `augment.enabled: false`. CutMix alone off: `augment.cutmix.enabled: false`.

### EMA weights

`train.weight_ema_decay` (default `0.9999`) maintains a shadow copy averaged after each optimizer step; set to `0.0` to disable. Validation each epoch uses the **EMA weights** when EMA is on. Inference can use `--ema` to load the saved shadow.

### Backbone selection

Edit [configs/default.yaml](configs/default.yaml):

- CNN: `backbone.type: cnn`, `backbone.name: resnet50` (or another timm model supporting `features_only`).
- ViT: `backbone.type: vit`, `backbone.name` = Hugging Face model id; tune `vit_patch_size` / `vit_num_prefix_tokens`.

## Config reference

| Section | Role |
|---------|------|
| `data` | `dataset_yaml`, optional `root_override` |
| `preprocess` | `imgsz`, ImageNet `mean` / `std` |
| `augment` | grayscale, jitter, resized crop; `cutmix` prob/beta |
| `backbone` | `type`, `name`, `pretrained`, ViT sizing |
| `model` | `img_stride`, `heatmap_bias_init`, focal and Gaussian overlap |
| `train` | batch size, epochs, LR, `ema_alpha` (loss display), `weight_ema_decay` (weights), `monitor_metric` (`map_50` or `map_50_95`), `freeze_backbone_epochs` |
| `metrics` | `map_iou_thresholds`, evaluation confidence, `topk_inference`, `pr_curve_metric_iou` |
