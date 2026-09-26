"""环境线索模型：全景视图 → 县级概率分布（附市级/省级/坐标辅助头）。

主干对每条视图独立提特征，再用**掩码注意力**在视图维池化。视图数可变
（1–8），池化天然支持，因此同一套权重既能吃单张截图也能吃八视图全景。

掩码用有限的负值而不是 -inf：T4 上跑 fp16，-inf 参与 softmax 容易出 NaN。
"""
import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    """视图维的加性注意力池化，忽略无效槽位。

    比平均池化多出的能力是：让模型自己决定哪些朝向更有信息量——正对
    街景车前进方向的视图和背对的可能差别很大。
    """

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
        return (feats * a.unsqueeze(-1)).sum(dim=1), a


class EnvModel(nn.Module):
    """多任务环境线索模型。

    主头是县级分类。市级与省级头是层级先验——早期县级信号很弱时，这两个
    头仍能提供可读的指标，也把"大尺度地理"的知识通过共享主干灌给县级的
    表示。坐标回归头提供平滑梯度，其输出还能直接参与选点。
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

        # 归一化放在模型里而非 Dataset：训练与推理必须用同一套常数，
        # 放在模型内部就不存在两边写歪的可能。主干是 ImageNet 预训练的，
        # 不归一化等于把预训练权重用在分布完全不同的输入上。
        self.register_buffer("pixel_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_backbone_trainable(self, trainable):
        self.freeze_backbone = not trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, views, vmask):
        """views (B, V, 3, H, W) uint8 或 float，vmask (B, V) bool。

        输入是 uint8（0–255）。必须先转 float 再归一化：autocast 只把权重
        转成 fp16，**不会转整数张量**，直接把 uint8 喂进卷积会报
        "Input type (unsigned char) and bias type (c10::Half) should be the same"。
        """
        b, v = views.shape[0], views.shape[1]
        flat = views.reshape(b * v, *views.shape[2:])
        flat = flat.float().div_(255.0)
        flat = (flat - self.pixel_mean) / self.pixel_std
        feats = self.backbone(flat)
        feats = feats.reshape(b, v, -1)
        feats = self.view_proj(feats)

        pooled, attn = self.pool(feats, vmask)
        pooled = self.drop(self.norm(pooled))

        return {
            "county": self.head_county(pooled),
            "city": self.head_city(pooled),
            "prov": self.head_prov(pooled),
            "coord": self.head_coord(pooled),
            "attn": attn,
        }


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
