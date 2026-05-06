from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from dataset import letterbox
from model import DinoV2CenterNet


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(image_path: Path, img_size: int, device: torch.device):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    resized, ratio, (pad_x, pad_y) = letterbox(rgb, img_size)
    img = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    img = TF.normalize(img, IMNET_MEAN, IMNET_STD)
    img = img.unsqueeze(0).to(device)

    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "ratio": ratio,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return img, bgr, meta


def scale_box_to_original(box, meta):
    x1, y1, x2, y2 = box

    x1 = (x1 - meta["pad_x"]) / meta["ratio"]
    x2 = (x2 - meta["pad_x"]) / meta["ratio"]
    y1 = (y1 - meta["pad_y"]) / meta["ratio"]
    y2 = (y2 - meta["pad_y"]) / meta["ratio"]

    x1 = int(np.clip(x1, 0, meta["orig_w"] - 1))
    x2 = int(np.clip(x2, 0, meta["orig_w"] - 1))
    y1 = int(np.clip(y1, 0, meta["orig_h"] - 1))
    y2 = int(np.clip(y2, 0, meta["orig_h"] - 1))

    return x1, y1, x2, y2


def draw_detections(image_bgr, detections, names, meta):
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det.tolist()
        x1, y1, x2, y2 = scale_box_to_original((x1, y1, x2, y2), meta)

        cls_id = int(cls_id)
        label = names[cls_id] if cls_id < len(names) else str(cls_id)
        text = f"{label} {score:.2f}"

        color = (0, 255, 0)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_bgr,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image_bgr


def collect_images(source: Path):
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/exp/best.pt")
    parser.add_argument("--source", default="detection_dataset_yolo/valid/images")
    parser.add_argument("--out", default="runs/exp/inference")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.weights, map_location=device)
    names = checkpoint.get("names", [])
    num_classes = len(names) if names else 4

    model = DinoV2CenterNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(source)
    print(f"Found {len(image_paths)} images")
    print(f"Device: {device}")
    print(f"Classes: {names}")

    for image_path in image_paths:
        img, original_bgr, meta = preprocess_image(image_path, args.imgsz, device)

        with torch.no_grad():
            detections = model.infer(img, topk=args.topk, conf_thres=args.conf)[0]

        result = draw_detections(original_bgr, detections, names, meta)
        save_path = out_dir / image_path.name
        cv2.imwrite(str(save_path), result)

        print(f"{image_path.name}: {len(detections)} detections -> {save_path}")


if __name__ == "__main__":
    main()