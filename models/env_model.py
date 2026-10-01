"""环境线索模型：全景视图 → 县级概率分布（附市级/省级/坐标辅助头）。

主干对每条视图独立提特征，再用掩码注意力在视图维池化，天然支持可变的
视图数（1–8）。掩码用有限负值而非 -inf：fp16 下 -inf 参与 softmax 易出 NaN。
"""
import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    """视图维加性注意力池化，忽略无效槽位。让模型自己决定哪些朝向更有信息量。"""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(64, dim // 4)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.mask_value = -1e4

    def forward(self, feats, vmask):
        """feats (B, V, D)，vmask (B, V) bool → (B, D) 与 (B, V) 权重。"""
        s = self.score(feats).squeeze(-1)                 # (B, V)
        s = s.masked_fill(~vmask, self.mask_value)
        a = torch.softmax(s.float(), dim=1).to(feats.dtype)
        a = a * vmask.to(a.dtype)
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (feats * a.unsqueeze(-1)).sum(dim=1), a


class EnvModel(nn.Module):
    """多任务环境线索模型。

    县级为主头；市级/省级头在县级信号很弱时提供可读指标，并把大尺度地理
    知识通过共享主干灌给县级表示；坐标头提供平滑梯度。
    """

    def __init__(self, n_counties, n_cities, n_provinces,
                 backbone="convnext_tiny", pretrained=True,
                 drop=0.2, freeze_backbone=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0)
        dim = self.backbone.num_features
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.view_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.pool = MaskedAttentionPool(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

        self.head_county = nn.Linear(dim, n_counties)
        self.head_city = nn.Linear(dim, n_cities)
        self.head_prov = nn.Linear(dim, n_provinces)
        self.head_coord = nn.Linear(dim, 2)

        # 归一化放在模型里而非 Dataset：训练与推理必须共用同一套常数。
        # 主干是 ImageNet 预训练的，不归一化等于把权重用在错分布的输入上。
        self.register_buffer("pixel_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_backbone_trainable(self, trainable):
        self.freeze_backbone = not trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, views, vmask, return_view_logits=False):
        """views (B, V, 3, H, W) uint8 或 float，vmask (B, V) bool。

        输入 uint8（0–255）。必须先转 float：autocast 只转权重，不转整数
        张量，直接喂 uint8 会报 Input type (unsigned char)。
        """
        b, v = views.shape[0], views.shape[1]
        flat = views.reshape(b * v, *views.shape[2:])
        flat = flat.float().div_(255.0)
        flat = (flat - self.pixel_mean) / self.pixel_std
        if self.training and return_view_logits:
            # Third-round training avoids backbone work on padded slots. Export/eval
            # retain the fixed-shape path, with exactly the same valid-view features.
            valid = vmask.reshape(-1)
            real = self.backbone(flat[valid])
            feats = real.new_zeros((b * v, real.shape[-1]))
            feats[valid] = real
        else:
            feats = self.backbone(flat)
        feats = feats.reshape(b, v, -1)
        feats = self.view_proj(feats)

        pooled, attn = self.pool(feats, vmask)
        pooled = self.drop(self.norm(pooled))

        result = {
            "county": self.head_county(pooled),
            "city": self.head_city(pooled),
            "prov": self.head_prov(pooled),
            "coord": self.head_coord(pooled),
            "attn": attn,
        }
        if return_view_logits:
            # Shared county head forces every real view to carry usable location evidence.
            result["view_county"] = self.head_county(self.drop(self.norm(feats)))
        return result


def build_model(n_counties, n_cities, n_provinces, cfg):
    """按配置构建模型。"""
    return EnvModel(
        n_counties=n_counties,
        n_cities=n_cities,
        n_provinces=n_provinces,
        backbone=cfg.get("backbone", "convnext_tiny"),
        pretrained=cfg.get("pretrained", True),
        drop=cfg.get("drop", 0.2),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )
