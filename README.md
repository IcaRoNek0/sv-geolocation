# 纯视觉街景县级定位

输入单张街景截图或全景，输出县级概率分布；坐标选点是附加的近似距离优化。
主要使用 **PyTorch** 训练与推理，ONNX 仅作为 Termux 的运行产物。

## 当前状态

- 数据：124,751 张、26 个分片，107,099 张训练，输出空间 2,604 县。
- 第二轮：149 个有真值的外部全景 Top-1 28.9%、Top-5 53.0%；单图显著弱于全景。
- 第三轮：硬/软县级标签混合、每视图辅助监督、截图增强、双卡 DDP 训练和验证。
- 街景车：独立车号/年份任务、俯视图预处理、轨迹先验和保守融合接口已实现。
  排除个人上传类型后 434 个训练车号；尚未训练验证识别效果。
- OCR 未实现。第三轮代码不代表已有新的高精度权重。

完整诊断、补数据建议及验收口径见 [第二轮复盘](reports/round2-review.md)。
旧 [PLAN.md](PLAN.md) 保留设计历史，以复盘和第三轮配置为准。

## Kaggle 双 T4

使用 [kaggle.ipynb](kaggle.ipynb)，具体步骤见 [KAGGLE.md](KAGGLE.md)。
`colab.ipynb` 是相同内容的 Colab 兼容入口。

```sh
# 两张卡共同训练环境模型
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config configs/round3.yaml --data data/shards --out runs/round3_env_v1

# 独立训练街景车模型，同样使用两张卡
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config configs/vehicle.yaml --data data/shards --out runs/vehicle_v1
```

首次训练不加 `--resume`。续跑使用相同配置；改变实验设置时用新目录，必要时
`--init /path/to/best.pt` 仅初始化权重。现有 ONNX 文件不能直接继续训练。

## 推理

```sh
python inference.py --run runs/round3_env_v1 --image shot.jpg --mode screenshot --json env.json
python inference.py --run runs/round3_env_v1 --image pano.jpg --mode panorama
python inference_vehicle.py --run runs/vehicle_v1 --image pano.jpg --mode panorama
```

默认把输入当截图。宽高比不能可靠区分全景与截图；请明确指定模式。
车分支必须有可见车顶，默认不融合。轨迹来自 `data/pool/vehicle_coverage.json`，
推理不读取文件名/panoID/EXIF 中的位置或车辆编号。
县名和边界是可选显示资源；独立仓库缺少相邻 GIS 项目时仍输出县码。

## 验证

```sh
python -m unittest discover -s tests -q
python tools/preflight.py --data data/shards
python tools/make_notebook.py
python tools/check_notebook.py
```

没有 PyTorch 的环境会明确跳过 Torch 测试。Kaggle 笔记本会运行这些测试，
并在正式训练前执行双卡短冒烟。当前本地未完成第三轮 GPU 实跑。

导出到 Termux 时，另装 `requirements-export.txt` 并运行
`python tools/export_onnx.py --run runs/round3_env_v1`；默认检查数值一致性。

图像、原始数据库、checkpoint 和 ONNX 大文件不入 git。仓库保留代码、配置、
轻量派生标签/点位表与审阅报告，训练分片仍使用既有 Kaggle Dataset。
