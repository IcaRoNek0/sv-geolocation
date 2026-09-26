"""多任务损失，县级用地理软标签。

县级不用 one-hot 交叉熵：相邻县的地貌与建筑高度相似，one-hot 会把"隔壁县"
和"隔着半个中国的县"同等惩罚。软标签让邻近的县分享一部分目标质量，这是
本方案"无文字线索 + 每县约 70 个点位"条件下的核心手段。

损失统一在 float32 下计算：T4 只能跑 fp16，而 log_softmax 与软标签相乘在
fp16 下容易下溢成 0，梯度随之消失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, soft_targets, w_county=1.0, w_city=0.3, w_prov=0.2,
                 w_coord=0.1, label_smoothing=0.05, ignore_index=-1):
        super().__init__()
        self.register_buffer("soft", torch.as_tensor(soft_targets,
                                                     dtype=torch.float32))
        self.w_county = w_county
        self.w_city = w_city
        self.w_prov = w_prov
        self.w_coord = w_coord
        self.ls = label_smoothing
        self.ignore_index = ignore_index

    def _ce(self, logits, target):
        return F.cross_entropy(logits.float(), target, ignore_index=self.ignore_index,
                               label_smoothing=self.ls)

    def forward(self, out, batch):
        county_target = batch["county"]
        valid = county_target != self.ignore_index

        # 县级：软标签交叉熵，目标分布由类别下标查出
        if valid.any():
            logp = F.log_softmax(out["county"].float(), dim=1)
            target_soft = self.soft[county_target.clamp(min=0)]
            # 留一点均匀质量，避免对任何单一类别过度自信
            if self.ls > 0:
                c = target_soft.shape[1]
                target_soft = (1 - self.ls) * target_soft + self.ls / c
            loss_county = -(target_soft * logp).sum(dim=1)[valid].mean()
        else:
            loss_county = out["county"].sum() * 0.0

        loss_city = self._ce(out["city"], batch["city"])
        loss_prov = self._ce(out["prov"], batch["prov"])

        # 坐标：在归一化空间上算 SmoothL1，越界不惩罚得比错省更狠
        loss_coord = F.smooth_l1_loss(out["coord"].float(), batch["coord"],
                                      reduction="mean")

        total = (self.w_county * loss_county
                 + self.w_city * loss_city
                 + self.w_prov * loss_prov
                 + self.w_coord * loss_coord)
        return total, {
            "county": float(loss_county.detach()),
            "city": float(loss_city.detach()),
            "prov": float(loss_prov.detach()),
            "coord": float(loss_coord.detach()),
        }
