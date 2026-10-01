# Kaggle 双 T4 操作流程

以 PyTorch 训练为主，打开仓库根目录的 `kaggle.ipynb`。`colab.ipynb` 是同内容的
兼容入口；不要使用旧笔记本里的后台 nohup / 默认 --resume 流程。

## 首次准备

1. 下载 GitHub 最新 `kaggle.ipynb`，在 Kaggle 创建 Notebook 并导入文件。
2. Settings → Accelerator → **GPU T4 x2**；Internet → **On**。
3. Add Input 挂载已有街景数据集。它需要包含 `samples.jsonl`、`split.json` 和
   26 个 tar，或这些 tar 的解包目录。当前有效样本数为 **124,751**。
4. 私有 GitHub 仓库：Add-ons → Secrets 新建 `GITHUB_TOKEN`（仓库只读即可），
   授权笔记本使用。公开仓库不需要。不要把 token 写到单元格、输出或 git URL。
5. 第一格保留：`TASK='environment'`、`RUN_NAME='round3_env_v1'`、
   `RESUME_FROM=''`、`INIT_WEIGHTS=''`、`RUN_SMOKE=True`、`EXPORT_ONNX=False`。
   如果挂载了多个街景数据集，填写 `DATA_OVERRIDE` 指向正确版本。

## 顺序运行

- **第 1 步：代码。** 输出 Git commit，拉取失败会立即停止，避免默默运行旧代码。
- **第 2 步：依赖。** 保留 Kaggle 的 PyTorch/CUDA，安装 timm 等；打印两张 GPU，
  运行测试（包含本机无法运行的 Torch 回归测试）。
- **第 3 步：数据。** 全量核对图像 key、samples、split、车辆日期组和少量解码。
  发现缺图或不匹配时先修输入，不继续训练。
- **第 4 步：冒烟。** 双卡、64 张、2 轮，验证计算和通信链路；不要求已经记住样本。
  若需检验过拟合能力，可独立跑更多轮，不能用冒烟验证集衡量泛化。
- **第 5 步：正式训练。** 前台执行，输出持续可见。环境配置为 15 轮、每卡 batch 8、
  梯度累积 2；总有效 batch 32。耗时以第一轮实际测量为准。
- **第 6 步：保存。** 检查 `best.pt`、`last.pt`、`classes.json`、`config.json`、
  `log.jsonl`。重点看全国单图 Top-1、宏平均及远距离误差，不能只看训练准确率。
- **第 7 步：可选导出。** 需要 Termux 推理时设置 `EXPORT_ONNX=True`，会先做
  PyTorch/ONNX 数值一致性检查。主产物仍是 `.pt`。

Kaggle 用 **Save Version → Save & Run All** 保留执行后的输出。训练在前台，
不会后台刚启动就结束版本。`/kaggle/working` 不是永久存储；尚未保存的交互会话
若被回收，不能保证 checkpoint 还在。重要 checkpoint 可及时手动下载。

## 断点续跑

将上次已保存版本输出通过 Add Input 挂载，然后把 `RESUME_FROM` 指到包含
`last.pt` 和 `classes.json` 的具体运行目录。使用新的 `RUN_NAME` 避免覆盖当前
非空目录。笔记本会把所需文件复制到可写目录后加 `--resume`。

数据版本、配置、GPU 数、batch 和总 epochs 必须相同。续跑会恢复优化器、调度器、
AMP scaler 和 epoch 内批次位置，但不保证 dropout 随机流逐位一致。旧第二轮要
改损失、增强或轮数时用新实验和 `INIT_WEIGHTS`，不要恢复旧优化器。

如有第二轮 `best.pt`，可设置 `INIT_WEIGHTS` 为其路径；同时挂载同目录的
`classes.json`。只有 ONNX 时无法直接续训，使用 ImageNet 预训练初始化第三轮即可。

## 街景车模型

完成环境基线后创建另一个 Kaggle 版本：

```python
TASK = 'vehicle'
RUN_NAME = 'vehicle_v1'
EPOCHS = 10
BATCH_SIZE = 32
RESUME_FROM = ''
INIT_WEIGHTS = ''
```

其余流程相同，两卡都参与该任务，不是“一张卡训环境、一张卡训车辆”。
车辆输入为从现有全景生成的俯视图，无需为基线重新采集。县级字段在该任务的
内部输出中复用为车号，city 头复用为年份；`task: vehicle` 区分模型类型，
环境推理程序会拒绝把车号解释成县码。

先评不同日期/不同县的车号识别、未知车和车不可见输入，再启用融合。默认融合
强度为 0，普通道路截图没有车顶时应只使用环境模型。

## 调整性能

先查看双卡是否都有显存占用和利用率，再看训练日志：

- `samples_per_sec` 是双卡合计；`gpu_peak_gb` 是 rank 0 的峰值已分配显存。
- 等数据时间长：先试每 rank 1 或 2 个 worker，避免超过 Kaggle CPU 核数太多；
  启动前已设 OMP/BLAS 单线程，模型开启了预取、pin_memory 和投影缓存。
- 算力占满而显存有余：在新实验中将环境每卡 batch 从 8 试到 12/16。不要在同一
  checkpoint 中途改 batch 后继续旧调度器。有效 batch 改变也可能需要重新调学习率。
- OOM：先减每卡 batch。增加 accum 可以保持有效 batch，不能让当前微批占更少显存。
- 车辆采用单视图，每卡 batch 32 起步；如果 OOM，同样减小。

不承诺双卡必为单卡的两倍，预处理和存储仍可能成为瓶颈。

## 本地 PyTorch 推理

```sh
python inference.py --run runs/round3_env_v1 --image shot.jpg --mode screenshot --json env.json
python inference.py --run runs/round3_env_v1 --image pano.jpg --mode panorama --json env.json
python inference_vehicle.py --run runs/vehicle_v1 --image pano.jpg --mode panorama --json car.json
```

需要实验融合时，为车辆命令追加：

```sh
--prior data/pool/vehicle_coverage.json --env-json env.json --strength 0.25
```

上述环境概率与车辆输入必须来自同一地点。Termux 用 `inference_onnx.py` 替代环境
入口，并给车辆入口添加 `--backend onnx`。始终配套下载相同运行的类别表。

## 当前验证范围

本地完成数据检查、非 Torch 回归测试、第二轮导出模型重评和笔记本语法检查。
当前机器无法运行 PyTorch/CUDA，尚未在真实 Kaggle 双 T4 会话完成第三轮训练；
因此依赖格的 Torch 测试与双卡冒烟格是正式训练前的必做检查。

## 第三轮慢速排查与本次修复

`round3` 截图中的红字是 Python 3.12 fork 和 PyTorch 2.10 pin_memory 的
DeprecationWarning；截图底部为 `Ran 99 tests / OK (skipped=2)`、returncode=0，
不是训练异常退出。fork 警告说明多线程父进程派生 worker 有死锁风险；如果实际
卡在取数，可先设 `WORKERS=0` 验证。pin_memory 警告来自框架内部，暂不禁用
异步传输，也不屏蔽整个警告类别。若发生真正退出，需要训练日志中的 traceback。

发现原版第三轮同时使用有效视图打包和 `cudnn.benchmark=True`：主干的输入
batch 随每批有效视图数改变，可能反复触发昂贵的算法搜索。本次默认关闭自动
调优，且打包开启时禁止自动调优；`pack_views` 独立于单视图辅助损失。
保留原有打包计算以节省双 T4 显存与计算量。训练不再为未使用的损失明细逐项
执行 GPU 到 CPU 标量转换。20 秒的具体来源尚需真实 T4 测量，不能据此保证提速倍数。

操作顺序：

1. 先保留当前 `last.pt`、`classes.json` 和日志，再停止旧训练进程。重跑拉取代码
   单元格只影响后续新进程，不能修改已经运行的训练进程。
2. 导入最新 `kaggle.ipynb`；保持数据、双卡、batch、accum、epochs、损失和视图设置。
   新会话按上面的断点续跑流程设置 `RESUME_FROM`；同会话的原 OUT 自动续跑。
   本次执行策略和计时变更兼容原第三轮 checkpoint，不必重新训练或补采数据。
3. 排查时设 `RUN_BENCHMARK=True`，可先设 `RUN_TRAINING=False`。第 3.5 格分别
   输出 packed/dense 的每卡稳态均值、中位数、P90 和峰值显存；日志保存到
   `/kaggle/working/benchmark_packed.log` 与 `benchmark_dense.log`。
4. 双卡冒烟通过后设 `RUN_TRAINING=True`。观察几十批后的两张卡日志：
   `recent20` 为最近 20 批平均，`data` 为等待 DataLoader，`work` 含传输、前后向、
   优化器和 DDP 等待。它们是主机端墙钟测量，不是独立 CUDA kernel 计时；已有
   训练指标的标量读取会等待 GPU。`epoch_avg` 还包含启动、恢复跳批与存盘。
5. 若 `data` 很长，比较每 rank 的 WORKERS=0/1/2，并检查数据是否已完成挂载；
   若合成基准也很慢，检查 GPU 利用率、两卡耗时和 DDP。若基准快但 `work` 慢，
   检查另一张卡的取数延迟：快卡可能在 DDP 等慢卡，不能只看 rank 0。

基准每个模式启动独立双卡进程，采用合成输入和未预训练的相同主干，包含混合
精度、辅助损失、梯度累积和优化器；不含真实读取/增强，不衡量准确率。默认
关闭，避免每次正常续训重复消耗配额。若 dense 确实更快且显存允许，可在配置
中设置 `pack_views: false` 做后续对比，保持单视图损失权重不变。
