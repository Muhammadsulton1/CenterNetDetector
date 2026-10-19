import torch
import torch.nn.functional as F
import torch.nn as nn

from .backbone import build_backbone
from .loss import _focal_loss


def decode(hm, wh, off, stride, topk=100, conf_thres=0.3, img_hw=None):
    """Пики heatmap -> по тензору [K, 6] (x1, y1, x2, y2, score, class) на картинку, в пикселях входа."""
    B, _, Hs, Ws = hm.shape
    H, W = img_hw or (Hs * stride, Ws * stride)
    hm = hm.clamp(0, 1)
    off = off.clamp(0, 1)  # цель кодируется как cx_s - int(cx_s) in [0, 1)

    results = []
    for b in range(B):
        heat = hm[b]  # (C, Hs, Ws)
        num_classes = heat.shape[0]

        hmax = F.max_pool2d(heat.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        keep_mask = (heat == hmax) & (heat > conf_thres)

        num_peaks = keep_mask.sum().item()
        if num_peaks == 0:
            results.append(torch.zeros((0, 6)))
            continue

        heat_filtered = heat * keep_mask.float()
        k = min(topk, num_peaks)
        scores, flat_indices = torch.topk(heat_filtered.flatten(), k)
        class_idx, row, col = torch.unravel_index(flat_indices, shape=(num_classes, Hs, Ws))

        offset = off[b].permute(1, 2, 0)[row, col]  # (K, 2)
        wh_det = wh[b].permute(1, 2, 0)[row, col]  # (K, 2)

        cx = (col.float() + offset[:, 0]) * stride
        cy = (row.float() + offset[:, 1]) * stride
        w = wh_det[:, 0] * stride
        h = wh_det[:, 1] * stride

        x1 = (cx - w / 2).clamp(0, W)
        y1 = (cy - h / 2).clamp(0, H)
        x2 = (cx + w / 2).clamp(0, W)
        y2 = (cy + h / 2).clamp(0, H)
        results.append(torch.stack([x1, y1, x2, y2, scores, class_idx.float()], dim=1).cpu())
    return results


class CenterNetHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.num_classes = num_classes
        hidden_ch = 256
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.hm = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, num_classes, 1),
        )
        self.wh = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 2, 1),
        )
        self.off = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 2, 1),
        )
        nn.init.constant_(self.hm[-1].bias, -2.19)

    def forward(self, x):
        x = self.shared(x)
        hm = self.hm(x)  # логиты (B, C, H/4, W/4): sigmoid берут лосс и decode во float32
        wh = F.softplus(self.wh(x))  # (B, 2, H/4, W/4)
        off = self.off(x)  # (B, 2, H/4, W/4)
        return hm, wh, off


class CenterNetModel(nn.Module):
    """CenterNet поверх любого бэкбона из backbone.build_backbone (DINOv2/DINOv3/ViT, ConvNeXt, timm)."""

    def __init__(self,
                 backbone: str = "facebook/dinov2-small",
                 num_classes: int = 80,
                 img_stride: int = 4,
                 feature_layers: int = 4,
                 pretrained: bool = True):
        super().__init__()
        self.stride = img_stride
        self.backbone = build_backbone(backbone, feature_layers, pretrained)
        self.set_backbone_trainable(False)  # по умолчанию предобученная часть заморожена
        self.head = CenterNetHead(self.backbone.out_channels, num_classes)

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.backbone.model.parameters():
            p.requires_grad = flag

    def backbone_parameters(self):
        return self.backbone.model.parameters()

    def new_parameters(self):
        """Всё, что учится с нуля: голова и neck (FPN / склейка слоёв)."""
        pretrained = {id(p) for p in self.backbone_parameters()}
        return [p for p in self.parameters() if id(p) not in pretrained]

    def _features(self, pixel_values):
        """Карта признаков (B, C, H // stride, W // stride), ячейка r покрывает пиксели [stride * r, stride * (r + 1))."""
        _, _, H, W = pixel_values.shape
        s = self.stride
        fm, reduction = self.backbone(pixel_values)  # карта покрывает дополненную до кратного картинку
        Hp, Wp = fm.shape[-2] * reduction, fm.shape[-1] * reduction
        fm = F.interpolate(fm, size=(-(-Hp // s), -(-Wp // s)), mode="bilinear", align_corners=False)
        # NCHW: после permute токенов карта channels_last, а cuDNN BatchNorm в ней в ~14 раз медленнее
        return fm[:, :, : H // s, : W // s].contiguous()

    def forward(self, pixel_values, targets):
        """Лосс CenterNet. targets - по словарю на картинку с плотными целями от DataLoader
        (utils.encode_targets): hm (C, h, w), wh и off (2, h, w), mask (h, w), где h, w = размер / stride."""
        hm, wh, off = (t.float() for t in self.head(self._features(pixel_values)))  # лосс во float32 и под AMP
        hm_t, wh_t, off_t, mask = (torch.stack([t[k] for t in targets]).float() for k in ("hm", "wh", "off", "mask"))
        if hm_t.shape != hm.shape:
            raise ValueError(f"цели {tuple(hm_t.shape)} не совпадают с выходом модели {tuple(hm.shape)}: "
                             f"датасет должен кодировать с тем же stride и числом классов")
        m = mask.unsqueeze(1)
        # как в статье CenterNet: heatmap / число объектов, L1 (RegL1Loss) / (2 координаты * число объектов)
        n = torch.clamp(mask.sum(), min=1)
        reg = F.l1_loss(wh * m, wh_t * m, reduction="sum")
        offset = F.l1_loss(off * m, off_t * m, reduction="sum")
        return _focal_loss(hm, hm_t) / n + 0.1 * reg / (2 * n) + offset / (2 * n)

    @torch.no_grad()
    def infer(self, pixel_values: torch.Tensor, topk: int = 100, conf_thres: float = 0.3):
        B, _, H, W = pixel_values.shape

        fm = self._features(pixel_values)

        # 2. Голова CenterNet
        hm, wh, off = self.head(fm)
        return decode(hm.float().sigmoid(), wh.float(), off.float(), self.stride, topk, conf_thres, (H, W))
