from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoBackbone, AutoConfig, AutoModel

try:  # transformers 5.x
    from transformers.backbone_utils import BackboneMixin
except ImportError:  # 4.x
    from transformers.utils.backbone_utils import BackboneMixin


def load_hf(name: str, pretrained: bool = True) -> nn.Module:
    """ViT-подобные модели (DINOv2, DINOv2 with registers, DINOv3-ViT, ViT) через AutoModel,
    остальные (ConvNeXt, DINOv3-ConvNeXt, ResNet, ...) как многомасштабный AutoBackbone со стадиями 1-4."""
    cfg = AutoConfig.from_pretrained(name)
    if hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "patch_size"):
        return AutoModel.from_pretrained(name) if pretrained else AutoModel.from_config(cfg)
    if pretrained:
        return AutoBackbone.from_pretrained(name, out_indices=[1, 2, 3, 4])
    return AutoBackbone.from_config(AutoConfig.from_pretrained(name, out_indices=[1, 2, 3, 4]))


def pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    """Дополняет справа и снизу до кратного m, иначе края картинки не попадут в карту признаков."""
    H, W = x.shape[-2:]
    return F.pad(x, (0, -W % m, 0, -H % m))


class ViTFeatures(nn.Module):
    """Токены патчей ViT с n слоёв, склеенные по каналам: карта (B, n * hidden, H / patch, W / patch).

    patch и число регистр-токенов берутся из config, поэтому подходят DINOv2 (14), DINOv2 with registers,
    DINOv3 (16, CLS + 4 регистра) и обычный ViT.
    """

    def __init__(self, model: nn.Module, n_layers: int = 1):
        super().__init__()
        cfg = model.config
        self.model = model  # предобученная часть: замораживается/размораживается целиком
        self.reduction = self.pad_to = cfg.patch_size
        self.prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        L = cfg.num_hidden_layers
        self.layers = [round(L * (j + 1) / n_layers) for j in range(n_layers)]  # равномерно, последний слой включён
        self.norm = getattr(model, "layernorm", None) or getattr(model, "norm", None)
        self.out_channels = cfg.hidden_size * n_layers
        # обычный ViT требует флаг для нестандартного размера, Dinov2/DINOv3 интерполируют сами и флага не знают
        self.kwargs = {"interpolate_pos_encoding": True} if "interpolate_pos_encoding" in inspect.signature(model.forward).parameters else {}

    def forward(self, x: torch.Tensor):
        x = pad_to_multiple(x, self.pad_to)
        B, _, H, W = x.shape
        if len(self.layers) == 1:
            states = [self.model(x, **self.kwargs).last_hidden_state]
        else:
            hidden = self.model(x, output_hidden_states=True, **self.kwargs).hidden_states
            states = [self.norm(hidden[i]) if self.norm is not None else hidden[i] for i in self.layers]
        maps = [s[:, self.prefix:, :].reshape(B, H // self.pad_to, W // self.pad_to, -1).permute(0, 3, 1, 2) for s in states]
        return torch.cat(maps, dim=1), self.reduction


class HierFeatures(nn.Module):
    """Иерархические бэкбоны (ConvNeXt, DINOv3-ConvNeXt, ResNet, timm): карты шагов 4..32 + FPN (top-down сумма)."""

    def __init__(self, model: nn.Module, get_maps, fpn_channels: int = 128):
        super().__init__()
        self.model, self.get_maps = model, get_maps
        was_training = model.training
        model.eval()
        with torch.no_grad():
            maps = get_maps(torch.zeros(1, 3, 128, 128))
        model.train(was_training)
        ratios = [128 // m.shape[-1] for m in maps]
        self.keep = [i for i, r in enumerate(ratios) if 4 <= r <= 32]
        self.reduction = ratios[self.keep[0]]
        self.pad_to = ratios[self.keep[-1]]
        self.lateral = nn.ModuleList(nn.Conv2d(maps[i].shape[1], fpn_channels, 1) for i in self.keep)
        self.smooth = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.out_channels = fpn_channels

    def forward(self, x: torch.Tensor):
        maps = [m for i, m in enumerate(self.get_maps(pad_to_multiple(x, self.pad_to))) if i in self.keep]
        p = self.lateral[-1](maps[-1])
        for m, lateral in zip(reversed(maps[:-1]), reversed(list(self.lateral)[:-1])):
            p = F.interpolate(p, size=m.shape[-2:], mode="nearest") + lateral(m)
        return self.smooth(p), self.reduction


def build_backbone(name: str, feature_layers: int = 1, pretrained: bool = True) -> nn.Module:
    """name: имя HuggingFace (facebook/dinov2-small, facebook/dinov3-vits16-pretrain-lvd1689m, ...)
    или timm/<имя модели> для моделей из timm."""
    if name.startswith("timm/"):
        import timm
        model = timm.create_model(name[len("timm/"):], pretrained=pretrained, features_only=True)
        return HierFeatures(model, lambda x: list(model(x)))
    model = load_hf(name, pretrained)
    if isinstance(model, BackboneMixin):
        return HierFeatures(model, lambda x: list(model(x).feature_maps))
    return ViTFeatures(model, feature_layers)
