#!/usr/bin/env python
"""把训练好的 checkpoint 导出成 ONNX，供没有 torch 的机器本地推理。

存在的理由：训练跑在 Kaggle/Colab（有 CUDA），但本地机器未必装得上 torch。
实测 Termux/ARM 上 PyPI 没有 aarch64 轮子，apt 的 python3-torch 虽能装上，
`import torch` 却是 SIGSEGV（退出码 139）。onnxruntime 则正常。

    python tools/export_onnx.py --run <checkpoint 目录> [--out model.onnx]

导出后本地只需 onnxruntime + numpy + PIL，不需要 torch。
"""
import argparse
from pathlib import Path

import torch

# torch.onnx.export 内部要 import onnx 来序列化，而 onnx 不在 torch 的依赖里
try:
    import onnx  # noqa: F401
except ImportError:
    raise SystemExit('需要 onnx 包：pip install -q onnx onnxscript')

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.env_model import EnvModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--views", type=int, default=None,
                    help="导出时固定的视图数；默认从 classes.json 读")
    args = ap.parse_args()

    import json
    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    vcfg = meta.get("views", {})
    n_max = args.views or vcfg.get("n_max", 4)
    size = vcfg.get("size", 224)

    ckpt = args.run / "best.pt"
    if not ckpt.exists():
        ckpt = args.run / "last.pt"
    out = args.out or (args.run / "model.onnx")

    model = EnvModel(len(meta["adcodes"]), len(meta["cities"]),
                     len(meta["provinces"]), backbone=meta["backbone"],
                     pretrained=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # 输入保持模型原始契约：uint8 视图 + bool 掩码。归一化在模型内部，
    # 所以导出后本地不必重复实现它——两边不一致会静默给出错的预测。
    views = torch.zeros(1, n_max, 3, size, size, dtype=torch.uint8)
    vmask = torch.ones(1, n_max, dtype=torch.bool)

    torch.onnx.export(
        model, (views, vmask), str(out),
        input_names=["views", "vmask"],
        output_names=["county", "city", "prov", "coord", "attn"],
        dynamic_axes={"views": {0: "batch"}, "vmask": {0: "batch"}},
        opset_version=17,
        # 显式关掉 dynamo 导出器。torch 2.6+ 的默认值在版本间变过，而两者
        # 依赖不同（dynamo 要 onnxscript）；固定走稳定路径，可预测。
        dynamo=False,
    )
    print(f"已导出 {out}  {out.stat().st_size/1e6:.0f} MB")
    print(f"类别 {len(meta['adcodes'])} 县  视图 {n_max}×{size}²")
    print()
    print("本地需要下载：")
    print(f"  {out.name}  （模型）")
    print(f"  classes.json  （类别表，缺了它 logit 无法映回 adcode）")
    print()
    print("本地推理：python inference_onnx.py --run <目录> --image <图>")


if __name__ == "__main__":
    main()
