import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn.functional as F
import torch.nn as nn

from loss import _focal_loss
from utils import _gaussian_radius, _draw_gaussian


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
        hm = self.hm(x).sigmoid()  # (B, C, H/4, W/4)
        wh = F.softplus(self.wh(x))  # (B, 2, H/4, W/4)
        off = self.off(x)  # (B, 2, H/4, W/4)
        return hm, wh, off


class DinoV2CenterNet(nn.Module):
    def __init__(self,
                 dino_name: str = "facebook/dinov2-small",
                 num_classes: int = 80,
                 img_stride: int = 4):
        super().__init__()
        self.stride = img_stride
        self.backbone = Dinov2Model.from_pretrained(dino_name)
        for p in self.backbone.parameters():  # freeze
            p.requires_grad = False
        hdim = self.backbone.config.hidden_size
        self.head = CenterNetHead(hdim, num_classes)

    def _encode_targets(self, boxes, labels, fm_h, fm_w, device):
        """boxes: FloatTensor[N,4] in (x1,y1,x2,y2) pixel coords."""
        num_classes = self.head.num_classes
        heatmap = torch.zeros((num_classes, fm_h, fm_w), device=device)
        wh_t = torch.zeros((2, fm_h, fm_w), device=device)
        off_t = torch.zeros((2, fm_h, fm_w), device=device)
        reg_mask = torch.zeros((fm_h, fm_w), device=device)

        for box, cls in zip(boxes, labels):
            cls_idx = int(cls.item())
            if not 0 <= cls_idx < num_classes:
                continue

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx_s, cy_s = cx / self.stride, cy / self.stride

            if not (0 <= cx_s < fm_w and 0 <= cy_s < fm_h):
                continue

            bw, bh = (x2 - x1) / self.stride, (y2 - y1) / self.stride

            # Защита от малых или отрицательных радиусов
            radius = _gaussian_radius((float(bh.item()), float(bw.item())))
            radius = max(0, int(radius))

            ix, iy = int(cx_s), int(cy_s)

            # Передаем целочисленные координаты (ix, iy) для отрисовки пика
            _draw_gaussian(heatmap[cls_idx], (ix, iy), radius)

            reg_mask[iy, ix] = 1
            # Создаем тензоры напрямую на нужном устройстве
            wh_t[:, iy, ix] = torch.tensor([bw, bh], dtype=torch.float32, device=device)
            off_t[:, iy, ix] = torch.tensor([cx_s - ix, cy_s - iy], dtype=torch.float32, device=device)

        return heatmap, wh_t, off_t, reg_mask

    def forward(self, pixel_values, targets):
        B, _, H, W = pixel_values.shape
        tok = self.backbone(
            pixel_values,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state
        patch_tokens = tok[:, 1:, :]

        ph, pw = H // 14, W // 14
        fm = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)
        fm = F.interpolate(fm, size=(H // self.stride, W // self.stride),
                           mode='bilinear', align_corners=False)

        hm, wh, off = self.head(fm)

        total_hm, total_reg, total_off, count = 0, 0, 0, 0
        for k in range(B):
            tgt = targets[k]
            heat_t, wh_t, off_t, mask = self._encode_targets(
                tgt["boxes"], tgt["labels"],
                hm.shape[2], hm.shape[3],
                pixel_values.device
            )

            total_hm += _focal_loss(hm[k], heat_t)
            m = mask.unsqueeze(0)  # (1, H, W)
            total_reg += F.l1_loss(wh[k] * m, wh_t * m, reduction='sum')
            total_off += F.l1_loss(off[k] * m, off_t * m, reduction='sum')
            count += mask.sum()

        # Нормализация loss'ов по количеству объектов (count), как в оригинальной статье
        if count > 0:
            total_reg /= count
            total_off /= count
            # hm loss часто тоже делят на count (число объектов),
            # чтобы избежать доминации фона при пустых изображениях
            hm_loss = total_hm / max(1, count)
        else:
            hm_loss = total_hm / B

        wh_loss = total_reg
        off_loss = total_off
        return hm_loss + 0.1 * wh_loss + off_loss

    @torch.no_grad()
    def infer(self, pixel_values: torch.Tensor, topk: int = 100, conf_thres: float = 0.3):
        B, _, H, W = pixel_values.shape

        # 1. Backbone => feature-map
        tok = self.backbone(
            pixel_values,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state
        patch_tokens = tok[:, 1:, :]

        ph, pw = H // 14, W // 14
        fm = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)
        fm = F.interpolate(fm, size=(H // self.stride, W // self.stride),
                           mode="bilinear", align_corners=False)

        # 2. Голова CenterNet
        hm, wh, off = self.head(fm)
        hm = hm.clamp_(0, 1)
        off = off.clamp(-0.5, 0.5)

        # 3. Пост-обработка по каждому изображению
        results = []
        for b in range(B):
            heat = hm[b]  # (C, Hs, Ws)
            num_classes, Hs, Ws = heat.shape

            # Используем unsqueeze/squeeze для безопасности F.max_pool2d
            hmax = F.max_pool2d(heat.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
            keep_mask = (heat == hmax) & (heat > conf_thres)

            # Считаем количество валидных пиков, чтобы не сортировать нули
            num_peaks = keep_mask.sum().item()
            if num_peaks == 0:
                results.append(torch.zeros((0, 6)))
                continue

            heat_filtered = heat * keep_mask.float()
            k = min(topk, num_peaks)
            scores, flat_indices = torch.topk(heat_filtered.flatten(), k)

            # Декодируем 1D индексы в (class_idx, row, col)
            class_idx, row, col = torch.unravel_index(flat_indices, shape=(num_classes, Hs, Ws))

            # 4. Извлекаем wh/off корректно
            # wh[b] имеет размер (2, Hs, Ws). permute(1,2,0) делает его (Hs, Ws, 2).
            # Доступ по [row, col] дает нам тензор формы (K, 2).
            offset = off[b].permute(1, 2, 0)[row, col]  # (K, 2)
            wh_det = wh[b].permute(1, 2, 0)[row, col]  # (K, 2)

            # 5. Проекция обратно в пространство исходного изображения
            cx = (col.float() + offset[:, 0]) * self.stride
            cy = (row.float() + offset[:, 1]) * self.stride
            w = wh_det[:, 0] * self.stride
            h = wh_det[:, 1] * self.stride

            x1 = (cx - w / 2).clamp(0, W)
            y1 = (cy - h / 2).clamp(0, H)
            x2 = (cx + w / 2).clamp(0, W)
            y2 = (cy + h / 2).clamp(0, H)

            # Собираем тензор детекций: (x1, y1, x2, y2, score, class)
            detections = torch.stack([x1, y1, x2, y2, scores, class_idx.float()], dim=1)
            results.append(detections.cpu())

        return results
