from __future__ import annotations

import sys

import yaml

from .engine import DEFAULTS, CenterNet, read_yaml

MODES = ("train", "val", "predict")
USAGE = """centernet MODE [key=value ...]

  centernet train   data=dataset.yaml model=facebook/dinov2-small epochs=50 batch=-1 imgsz=640
  centernet train   data=dataset.yaml model=timm/convnext_small.in12k_ft_in1k freeze_epochs=3  # любой поддерживаемый бэкбон
  centernet val     model=runs/train/exp/best.pt data=dataset.yaml
  centernet predict model=runs/train/exp/best.pt source=img_or_dir conf=0.25

  cfg=my.yaml накладывается на значения по умолчанию, аргументы командной строки - поверх него.
  Ключи по умолчанию: """ + ", ".join(DEFAULTS)


def parse_args(pairs: list[str]) -> dict:
    args = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"ожидался аргумент key=value, получено: {pair!r}")
        args[key] = yaml.safe_load(value)
    if "cfg" in args:
        args = {**read_yaml(args.pop("cfg")), **args}
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(USAGE)
        return 0
    mode, *pairs = argv
    if mode not in MODES:
        print(f"неизвестный режим {mode!r}, доступны: {', '.join(MODES)}\n\n{USAGE}")
        return 2
    try:
        args = parse_args(pairs)
        model = args.pop("model", DEFAULTS["model"])
        getattr(CenterNet(model), mode)(**args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Ошибка: {e}")
        return 2
    return 0
