import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn.functional as F
import torch.nn as nn

from loss import _focal_loss
from utils import _gaussian_radius, _draw_gaussian


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
        num_classes = self.head.hm.out_channels
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
            radius = _gaussian_radius((float(bh.item()), float(bw.item())))
            _draw_gaussian(heatmap[cls_idx], (float(cx_s.item()), float(cy_s.item())), radius)

            ix, iy = int(cx_s), int(cy_s)
            reg_mask[iy, ix] = 1
            wh_t[:, iy, ix] = torch.stack([bw, bh])
            off_t[:, iy, ix] = torch.stack([cx_s - ix, cy_s - iy])
        return heatmap, wh_t, off_t, reg_mask

    def forward(self, pixel_values, targets):
        """
        pixel_values : FloatTensor[B,3,H,W] in [0,1]
        targets      : list of dict{boxes, labels} per image
        Returns total scalar loss.
        """
        B, _, H, W = pixel_values.shape
        tok = self.backbone(
            pixel_values,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state  # (B, 1+P, C)
        patch_tokens = tok[:, 1:, :]  # drop CLS
        ph, pw = H // 14, W // 14
        fm = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)  # (B,C,ph,pw)
        fm = F.interpolate(fm, size=(H // self.stride, W // self.stride),
                           mode='bilinear', align_corners=False)
        hm, wh, off = self.head(fm)
        total_hm, total_reg, total_off, count = 0, 0, 0, 0
        for k in range(B):
            tgt = targets[k]
            heat_t, wh_t, off_t, mask = self._encode_targets(
                tgt["boxes"], tgt["labels"],
                hm.shape[2], hm.shape[3],
                pixel_values.device)
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
        """
        Инференс-путь модели CenterNet-головы на признаках DinoV2.

        Parameters
        ----------
        pixel_values : FloatTensor[B, 3, H, W]
            RGB-изображения в пространстве ImageNet, **нормализованные в [0, 1]**.
            Считаем, что H и W кратны `self.stride` (обычно 4).
        topk : int
            Жёсткий лимит на число детекций **на изображение** после фильтрации
            локальных максимумов.
        conf_thres : float
            Порог для откладывания слабых пиков heat-map’ы.
            Это **score-threshold**, а не IoU-threshold; изменяя его мы торговали
            precision ↔ recall ещё до NMS.

        Returns
        -------
        list[Tensor[num_det, 6]]
            Для каждого изображения — `Tensor` с колонками

            `(x1, y1, x2, y2, score, class)`
            где координаты — в **исходной системе пикселей**, а класс —
            целое `0 ≤ c < num_classes`.

        Как работает инфренес сети
        PER-IMAGE POST-PROCESSING  (loop b = 0 … B-1)                            │
        │  1.  heat   ← hm[b]                                                      │
        │  2.  pooled ← MaxPool₍3×3₎(heat)                                         │
        │  3.  keep_mask = (heat = pooled) ∧ (heat > τ), τ = conf_thres            │
        │  4.  coords = nonzero(keep_mask) → (cls, y, x)                           │
        │  5.  scores = heat[keep_mask]                                            │
        │      sort (scores ↓), keep top-k                                         │
        │  6.  Gather regression:                                                  │
        │          w, h  = wh[b][:, y, x]                                          │
        │          δx,δy = off[b][:, y, x]                                         │
        │          x*  = x + δx     ;     y* = y + δy                              │
        │  7.  Back-project to pixel space (stride s):                             │
        │          x₁ = (x* – w/2)·s  y₁ = (y* – h/2)·s                            │
        │          x₂ = (x* + w/2)·s  y₂ = (y* + h/2)·s                            │
        │  8.  det = [x₁, y₁, x₂, y₂, score, cls]                                  │
        │      move to CPU, append
                """
        B, _, H, W = pixel_values.shape

        # ───────────────── 2. Backbone ⇒ feature-map ───────────────────
        # DinoV2 — Vision Transformer, который возвращает (B, 1 + P, C),
        # где первый токен — CLS. Для детекции он бесполезен: spatial info = 0.
        tok = self.backbone(
            pixel_values,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state
        patch_tokens = tok[:, 1:, :]  # drop CLS → (B, P, C)

        # Отбрасываем CLS, а оставшиеся P=ph*pw токены ре-шейпим
        # CenterNet предпочитает stride = 4, т.е. разрешение выше, чем у ViT.
        # Поднимаем fm биллинеаром — простой, дешёвый способ,
        # который в практике почти не бьёт по mAP.
        ph, pw = H // 14, W // 14
        fm = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)  # (B,C,ph,pw)
        fm = F.interpolate(fm, size=(H // self.stride, W // self.stride),
                           mode="bilinear", align_corners=False)

        # ───────────────── 3. Голова CenterNet ─────────────────────────
        # hm   ∈ [0,1] — heat-map для K классов
        # wh   — шаблоны (w,h) на каждой решётке
        # off  — суб-пиксельный сдвиг центра (dx,dy) ∈ (-0.5,0.5]
        hm, wh, off = self.head(fm)
        hm = hm.clamp_(0, 1)
        off = off.clamp(-0.5, 0.5)

        # ───────────────── 4. Пост-обработка по каждому изображению ────
        results = []
        for b in range(B):
            heat = hm[b]  # (C, Hs, Ws)
            num_classes, Hs, Ws = heat.shape

            # Фильтрация локальных максимумов
            # т.к нету nms у CenterNet фильтрация происходит через MaxPool2d ищется пик в heatmap
            # max_pool2d c k=3 делает non-max suppression в 8-связности:
            hmax = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
            keep_mask = (heat == hmax) & (heat > conf_thres)
            heat_filtered = heat * keep_mask.float()

            k = min(topk, heat_filtered.numel())
            scores, flat_indices = torch.topk(heat_filtered.flatten(), k)

            # Если ни одного пика не выжило — добавляем пустой тензор и гуляем дальше.
            valid = scores > 0
            if valid.sum() == 0:
                results.append(torch.zeros((0, 6)))
                continue
            scores = scores[valid]
            flat_indices = flat_indices[valid]

            # Достаём регрессии wh/off в тех же точках, где нашли пики
            class_idx, row, col = torch.unravel_index(flat_indices,
                                                      shape=(num_classes, Hs, Ws))

            # Предсказанные ширина и высота (в коордах feature-карты!)
            offset = off[b, :, row, col]
            wh_det = wh[b, :, row, col]

            # Переводим всё обратно в пространство исходного изображения.
            # Формула:  x_img = (x_feat ± w/2) * stride
            cx = (col.float() + offset[0]) * self.stride
            cy = (row.float() + offset[1]) * self.stride
            w = wh_det[0] * self.stride
            h = wh_det[1] * self.stride

            # Собираем финальный тензор детекций:
            # (x1,y1,x2,y2,score,class)

            x1 = (cx - w / 2).clamp(0, W)
            y1 = (cy - h / 2).clamp(0, H)
            x2 = (cx + w / 2).clamp(0, W)
            y2 = (cy + h / 2).clamp(0, H)

            detections = torch.stack([x1, y1, x2, y2, scores, class_idx.float()], dim=1)
            # CPU — чтобы не держать всё в GPU-RAM
            results.append(detections.cpu())

        return results
