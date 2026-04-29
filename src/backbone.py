import torch.nn.functional as F
import torch.nn as nn
import torch

from src.loss import _focal_loss
from src.utils import _gaussian_radius, _draw_gaussian


class CenterNetHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(256, num_classes, 1)
        self.wh = nn.Conv2d(256, 2, 1)
        self.off = nn.Conv2d(256, 2, 1)
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, x):
        x = self.shared(x)
        hm = self.hm(x).sigmoid()  # (B, C, H/4, W/4)
        wh = self.wh(x)  # (B, 2, H/4, W/4)
        off = self.off(x)  # (B, 2, H/4, W/4)
        return hm, wh, off


class BackboneAdapter(nn.Module):
    def __init__(self, backbone, backbone_type: str = "vit", patch_size: int = 14, num_prefix_tokens: int = 1):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type  # "vit" | "cnn"
        self.patch_size = patch_size
        self.num_prefix_tokens = num_prefix_tokens  # 1 для base, 5 для dinov2-reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if self.backbone_type == "vit":
            out = self.backbone(x, output_hidden_states=False).last_hidden_state
            patch_tokens = out[:, self.num_prefix_tokens:, :]  # drop CLS/reg tokens
            ph, pw = H // self.patch_size, W // self.patch_size
            return patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)
        else:  # CNN через timm: feature_only=True
            return self.backbone(x)[-1]  # последний stage


class CenterNet(nn.Module):
    def __init__(self, adapter: BackboneAdapter, num_classes: int, img_stride: int = 4):
        super().__init__()
        self.stride = img_stride
        self.backbone = adapter
        hdim = self._get_hdim(adapter)
        self.head = CenterNetHead(hdim, num_classes)

    @staticmethod
    def _get_hdim(adapter: BackboneAdapter) -> int:
        if adapter.backbone_type == "vit":
            return adapter.backbone.config.hidden_size
        else:
            return adapter.backbone.feature_info[-1]["num_chs"]

    def _get_feature_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Единая точка получения fm для forward и infer."""
        B, _, H, W = pixel_values.shape
        fm = self.backbone(pixel_values)  # → (B, C, ph, pw) через адаптер
        return F.interpolate(
            fm, size=(H // self.stride, W // self.stride),
            mode="bilinear", align_corners=False
        )

    def _encode_targets(self, boxes, labels, fm_h, fm_w, device):
        num_classes = self.head.hm.out_channels
        heatmap = torch.zeros((num_classes, fm_h, fm_w))
        wh_t = torch.zeros((2, fm_h, fm_w))
        off_t = torch.zeros((2, fm_h, fm_w))
        reg_mask = torch.zeros((fm_h, fm_w))

        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx_s, cy_s = cx / self.stride, cy / self.stride
            if not (0 <= cx_s < fm_w and 0 <= cy_s < fm_h):
                continue
            bw, bh = (x2 - x1) / self.stride, (y2 - y1) / self.stride
            radius = _gaussian_radius((bh, bw))
            _draw_gaussian(heatmap[cls], (cx_s, cy_s), radius)

            ix, iy = int(cx_s), int(cy_s)
            reg_mask[iy, ix] = 1
            wh_t[:, iy, ix] = torch.tensor([bw, bh])
            off_t[:, iy, ix] = torch.tensor([cx_s - ix, cy_s - iy])

        return heatmap.to(device), wh_t.to(device), off_t.to(device), reg_mask.to(device)

    def forward(self, pixel_values, targets):
        B, _, H, W = pixel_values.shape
        fm = self._get_feature_map(pixel_values)
        hm, wh, off = self.head(fm)

        total_hm, total_reg, total_off, count = 0, 0, 0, 0
        for k in range(B):
            tgt = targets[k]
            heat_t, wh_t, off_t, mask = self._encode_targets(
                tgt["boxes"], tgt["labels"],
                hm.shape[2], hm.shape[3], pixel_values.device)
            total_hm += _focal_loss(hm[k], heat_t)
            m = mask.unsqueeze(0)
            total_reg += F.l1_loss(wh[k] * m, wh_t * m, reduction='sum')
            total_off += F.l1_loss(off[k] * m, off_t * m, reduction='sum')
            count += mask.sum()
        if count > 0:
            total_reg /= count
            total_off /= count
        return total_hm + total_reg + total_off

    @torch.no_grad()
    def infer(self, pixel_values: torch.Tensor, topk: int = 100, conf_thres: float = 0.3):
        B, _, H, W = pixel_values.shape
        fm = self._get_feature_map(pixel_values)
        hm, wh, off = self.head(fm)
        hm = hm.clamp_(0, 1)
        off = off.clamp(-0.5, 0.5)

        results = []
        for b in range(B):
            heat = hm[b]
            num_classes, Hs, Ws = heat.shape

            hmax = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
            keep_mask = (heat == hmax) & (heat > conf_thres)
            heat_filtered = heat * keep_mask.float()

            k = min(topk, heat_filtered.numel())
            scores, flat_indices = torch.topk(heat_filtered.flatten(), k)
            valid = scores > 0
            if valid.sum() == 0:
                results.append(torch.zeros((0, 6)))
                continue
            scores, flat_indices = scores[valid], flat_indices[valid]

            class_idx, row, col = torch.unravel_index(
                flat_indices, shape=(num_classes, Hs, Ws))

            offset = off[b, :, row, col]
            wh_det = wh[b, :, row, col]

            cx = (col.float() + offset[0]) * self.stride
            cy = (row.float() + offset[1]) * self.stride
            w = wh_det[0] * self.stride
            h = wh_det[1] * self.stride

            x1 = (cx - w / 2).clamp(0, W)
            y1 = (cy - h / 2).clamp(0, H)
            x2 = (cx + w / 2).clamp(0, W)
            y2 = (cy + h / 2).clamp(0, H)

            results.append(torch.stack([x1, y1, x2, y2, scores, class_idx.float()], dim=1).cpu())
        return results
