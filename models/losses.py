"""多任务损失，县级用地理软标签（理由见 data/labels.py）。

损失统一在 float32 下算：fp16 下 log_softmax 与软标签相乘容易下溢成 0，
梯度随之消失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, soft_targets, w_county=1.0, w_city=0.3, w_prov=0.2,
                 w_coord=0.1, label_smoothing=0.05, ignore_index=-1,
                 geo_mix=1.0, w_view=0.0):
        super().__init__()
        self.register_buffer("soft", torch.as_tensor(soft_targets,
                                                     dtype=torch.float32))
        if not 0 <= geo_mix <= 1:
            raise ValueError("geo_mix must be in [0,1]")
        self.geo_mix = geo_mix
        self.w_view = w_view
        self.w_county = w_county
        self.w_city = w_city
        self.w_prov = w_prov
        self.w_coord = w_coord
        self.ls = label_smoothing
        self.ignore_index = ignore_index

    def _ce(self, logits, target):
        valid = target != self.ignore_index
        if not valid.any():
            return logits.float().sum() * 0.0
        return F.cross_entropy(logits[valid].float(), target[valid],
                               label_smoothing=self.ls)

    def forward(self, out, batch, collect_parts=True):
        county_target = batch["county"]
        valid = county_target != self.ignore_index

        # 县级：软标签交叉熵
        if valid.any():
            logp = F.log_softmax(out["county"].float(), dim=1)
            target_soft = self.soft[county_target.clamp(min=0)] * self.geo_mix
            hard = F.one_hot(county_target.clamp(min=0), self.soft.shape[0]).float()
            target_soft = target_soft + (1.0 - self.geo_mix) * hard
            # 留一点均匀质量，避免过度自信
            if self.ls > 0:
                c = target_soft.shape[1]
                target_soft = (1 - self.ls) * target_soft + self.ls / c
            loss_county = -(target_soft * logp).sum(dim=1)[valid].mean()
        else:
            loss_county = out["county"].sum() * 0.0

        loss_view = out["county"].float().sum() * 0.0
        if self.w_view and "view_county" in out and valid.any():
            view_logp = F.log_softmax(out["view_county"].float(), dim=-1)
            per_view = -(target_soft[:, None, :] * view_logp).sum(dim=-1)
            mask = batch["vmask"].float()
            # Average per sample before averaging the batch: 4 views do not get 4x weight.
            loss_view = ((per_view * mask).sum(1) / mask.sum(1).clamp_min(1))[valid].mean()

        loss_city = self._ce(out["city"], batch["city"])
        loss_prov = self._ce(out["prov"], batch["prov"])

        # 坐标：归一化空间上的 SmoothL1
        loss_coord = F.smooth_l1_loss(out["coord"].float(), batch["coord"],
                                      reduction="mean")

        total = (self.w_county * loss_county
                 + self.w_city * loss_city
                 + self.w_prov * loss_prov
                 + self.w_coord * loss_coord
                 + self.w_view * loss_view)
        if not collect_parts:
            return total, {}
        return total, {
            "county": float(loss_county.detach()),
            "view": float(loss_view.detach()),
            "city": float(loss_city.detach()),
            "prov": float(loss_prov.detach()),
            "coord": float(loss_coord.detach()),
        }
