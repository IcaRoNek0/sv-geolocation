# Colab 操作手册

核心限制：**Colab 访问不到本机**。路径只能是

    本地采集 → 打包分片 → 上传 Drive → Colab 读取训练

每个会话重复同样三步：挂载 Drive、把分片拷到本地磁盘、拉起训练。

---

## 一、本机（一次性）

```sh
cd ai/collect

# 1. 全部元数据与图像（可反复重跑，状态库自动续传）
python fetch_pano.py meta
python fetch_pano.py images --concurrency 160

# 2. 失败项重试：把 fail 退回 meta_ok 再跑一次 images
python -c "
import sqlite3; c=sqlite3.connect('../data/fetch_state.sqlite3')
c.execute(\"UPDATE panos SET state='meta_ok' WHERE state='fail'\"); c.commit()"
python fetch_pano.py images --concurrency 160

# 3. 察看实际体积与层级分布
python fetch_pano.py report

# 4. 打包成 WebDataset 分片
python pack_shards.py --shard-mb 500

# 5. 生成训练/评估划分（只做一次，之后固定）
python make_splits.py
```

产物在 `ai/data/shards/`：

| 文件 | 内容 | 去向 |
|---|---|---|
| `sv-XXXX.tar` | 图像分片，约 500 MB/片 | **上传 Drive** |
| `samples.jsonl` | 全量元数据 | 上传 Drive（很小） |
| `split.json` | 划分，固定不变 | 上传 Drive（很小） |
| `manifest.json` | 分片清单 | 上传 Drive |

上传用 rclone 最省事。**别传散图**——几千个小文件在 Drive 的 FUSE 上会拖垮
Colab 的读取速度。

---

## 二、Colab（每个会话）

新建 notebook → 菜单「代码执行程序 → 更改运行时类型 → 硬件加速器选 T4 GPU」。

```python
# 1. 挂载 Drive + 确认拿到 GPU
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi                      # 没看到 Tesla T4 就回去换运行时

# 2. 把分片拷到本地磁盘 —— 关键，Drive 直读慢 3-5 倍
!mkdir -p /content/data
!cp -r /content/drive/MyDrive/sv/shards /content/data/
!ls -la /content/data/shards | head
```

```python
# 3. 取代码（仓库是私有的，需要 token；公开后可直接 wget）
!git clone https://github.com/IcaRoNek0/sv-geolocation.git /content/ai
%cd /content/ai
!pip install -q timm pyyaml

# 4. 挂在后台跑，避免浏览器断开时中断
!nohup python train.py --config configs/default.yaml \
    --data /content/data/shards \
    --out /content/drive/MyDrive/sv/runs/base \
    > /content/train.log 2>&1 &
!sleep 60 && tail -20 /content/train.log
```

**先做 M0 冒烟**，确认能过拟合再跑正式训练：

```python
!python train.py --config configs/default.yaml \
    --data /content/data/shards --out /content/drive/MyDrive/sv/runs/smoke \
    --overfit 64 --resume
```

64 条样本应当很快把训练损失压到接近零、同县留出 top-1 接近 100%。
做不到就说明管线有问题，**此时不要往下跑**，先查数据与标签。

---

## 三、训练中

```python
!tail -5 /content/train.log
```

关键看 `val_same` 那一行：

```
e3 同县留出 top1 0.021 top5 0.083 宏平均 0.015 多数类基线 0.012 中位误差 412.3 km
   候选邻近率(top5 落在真值 150km 内) 0.271
```

**多数类基线必须并列看**。top-1 高于基线不等于模型在学地理——它可能只是
背下了"样本最多的那个县"。

**候选邻近率比 top-1 更早给出信号**：模型可能还点不中正确的县，但只要它的
top-5 候选连成一片且围绕真值，就说明它在学地理。这个数长期停在基线附近，
说明模型没学到东西，加数据量也没用，该查标签与划分。

---

## 四、断线之后

Colab 空闲 90 分钟断、硬上限 12 小时（实际 4–6 小时常被回收）。断线不是
故障，是常态。重开 notebook 后**重跑第二节的三段**，训练会自动续上：

```python
!nohup python train.py --config configs/default.yaml \
    --data /content/data/shards \
    --out /content/drive/MyDrive/sv/runs/base \
    --resume > /content/train.log 2>&1 &
```

`--resume` 从 Drive 上的 `last.pt` 接着跑。每 25 分钟自动存一次盘，所以最多
丢 25 分钟的进度。

**别关浏览器标签页**，关了通常会断。

---

## 五、推理

```python
!python inference.py --run /content/drive/MyDrive/sv/runs/base \
    --image /content/test.jpg --topk 5
```

输入全景或截图都行，自动判别：

```
输入 test.jpg  2048×1024  全景 → 8 个视图

县级候选（前 5）：
  510105  概率  18.3%   点位数 92
  510104  概率  11.7%   点位数 84
  ...

选点（20 个候选县的真实点位加权中位数）：
  经度 104.07120   纬度 30.65941
```

---

## 六、会踩的坑

| 现象 | 原因 |
|---|---|
| 训练比预计慢很多 | T4 是 2018 年的卡，且只支持 fp16；算力瓶颈而非显存 |
| 出现 NaN | 多半是在 fp16 下做 softmax/log_softmax；本仓库的损失已强制 float32 |
| `CUDA out of memory` | 调小 `batch_size` 或 `views.size`，别调 `accum`（accum 不省显存） |
| 续跑后指标跳变 | 分片顺序漂移了；确认 `--seed` 没变 |
| Drive 写满 | 免费档 15 GB，checkpoint 会累积；删掉旧的 `best.pt`/`last.pt` |
| 分配不到 GPU | 免费档每周 15–40 GPU 小时且动态调整，高峰期只能等 |
