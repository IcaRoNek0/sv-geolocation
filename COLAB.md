# Colab 兼容入口

当前 `colab.ipynb` 与 `kaggle.ipynb` 由同一生成脚本维护，完整流程见
[KAGGLE.md](KAGGLE.md)。优先使用 Kaggle 双 T4；Colab 会自动识别单卡并启动一个进程。

Colab 需要先把完整分片、samples.jsonl、split.json 放入
`/content/drive/MyDrive/sv/shards`。笔记本挂载 Drive 后拷贝到本地训练，checkpoint
写回 Drive。请确认 Drive 空间足够容纳约 13.1 GB 分片与 checkpoint。

双卡训练的 last.pt 不可在改变卡数后直接恢复旧调度器；这种情况应新建实验，
使用 `INIT_WEIGHTS` 初始化 `.pt` 权重。完整环境模型是 `configs/round3.yaml`，
街景车为 `configs/vehicle.yaml`。旧的“冒烟验证集应接近 100%”说法不正确：
过拟合能力看训练样本准确率，独立验证集仍测泛化。
