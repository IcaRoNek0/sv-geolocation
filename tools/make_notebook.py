#!/usr/bin/env python
"""生成 colab.ipynb。

用脚本生成而不是手写 notebook JSON：手写容易在转义和结构上出错，
而 notebook 格式错了在 Colab 里只会给一句含糊的导入失败。

改完本文件后运行：
    python tools/make_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "colab.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text, title=None):
    """加一个代码单元格。

    单元格魔法（%%bash 等）必须是**第一行**，而 Colab 的 #@title 也要抢首行。
    两者不能共存：%%bash 前面插了 #@title 就不再被识别为魔法，整格会按
    Python 解析并报语法错误。因此带 %% 的格子不加 title——每个 bash 格上方
    本来就有 markdown 小标题。
    """
    src = text.strip("\n")
    if title and not src.startswith("%%"):
        src = f"#@title {title}\n" + src
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(keepends=True)})


# ─────────────────────────────────────────────────────────────────────
md("""
# 街景定位 · 训练与推理

输入一张无元数据的街景图，输出中国县级概率分布与期望得分最高的坐标。

**这个 notebook 是启动器，训练逻辑全在仓库里的 `train.py`。** 改超参请改
`configs/default.yaml` 而不是在这里改——notebook 里改的东西没法版本管理，
下次从 GitHub 打开就没了。

---

## 怎么用

1. 菜单「代码执行程序 → 更改运行时类型 → 硬件加速器选 **T4 GPU**」
2. 从上往下依次跑 ①②③
3. 第 ③ 格会告诉你走**路径 A**（Colab 直接采集，省 4 GB 上传）还是**路径 B**（本地上传）
4. 之后按提示继续

**断线是常态不是故障**：Colab 会话 4–6 小时会被回收。重开之后跑
**① → ② → ⑥ → ⑩**，训练会自动从断点续上。
""")

# ── ①
md("## ① 挂载 Drive 并确认 GPU")
code("""
from google.colab import drive
drive.mount('/content/drive')

import subprocess, torch
out = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True).stdout.strip()
print('GPU:', out or '(没拿到，检查运行时是否选了 T4)')
print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f'compute capability {cc[0]}.{cc[1]} →',
          'bf16 可用' if cc[0] >= 8 else 'Turing 及更早，只能 fp16（正常，训练脚本会自动选）')
""", title="① 挂载 Drive + 确认 GPU")

# ── ②
md("""## ② 取代码

仓库是私有的，需要 GitHub token（只读权限即可）。token 用 `getpass` 读入、
经 `subprocess` 传给 git，**不会出现在 notebook 输出里**。

把仓库改成公开的话，token 那步直接回车跳过即可。
""")
code("""
import os, subprocess
from getpass import getpass

REPO = 'IcaRoNek0/sv-geolocation'
BRANCH = 'main'

EXPECT = '1c2338a'      # 本 notebook 对应的代码版本，见 README

if os.path.isdir('/content/ai/.git'):
    print('代码已存在，拉取最新')
    r = subprocess.run(['git', '-C', '/content/ai', 'pull'],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip() or '(无输出)')
    if r.returncode != 0:
        # 拉取失败必须报错：静默失败会让你拿旧代码跑，还以为是新代码
        raise SystemExit('拉取失败。常见原因：/content/ai 里有本地改动。\\n'
                         '删掉 /content/ai 重跑本格即可（分片在 /content/data，不受影响）。')
else:
    tok = getpass('GitHub token（仓库已公开则直接回车）: ').strip()
    url = (f'https://{tok}@github.com/{REPO}.git' if tok
           else f'https://github.com/{REPO}.git')
    # 用 subprocess 而非 ! 前缀：! 会把带 token 的完整命令打进输出
    r = subprocess.run(['git', 'clone', '-q', '-b', BRANCH, url, '/content/ai'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('克隆失败（token 无效？仓库名不对？）\\n' + r.stderr[-500:])

os.chdir('/content/ai')
head = subprocess.run(['git', 'log', '--oneline', '-1'],
                      capture_output=True, text=True).stdout.strip()
print('当前代码：', head)
if EXPECT not in head:
    print(f'⚠ 期望版本含 {EXPECT}。若刚拉取过仍不符，说明服务器上的 main 还没更新。')
""", title="② 取代码")

code("""
!pip install -q aiohttp tqdm timm pyyaml
""", title="②b 依赖")

# ── ③
md("""## ③ 连通性测试：决定走 A 还是 B

百度对境外/数据中心 IP 的行为未验证。这一格花两分钟测一次，通过就走路径 A
（Colab 自己抓图，省掉 4.25 GB 上传），不通过就走路径 B。

先跑下面那一格。
""")
code("""
import time, urllib.request

H = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.baidu.com/'}
PID = '09006900001410191436379165M'          # 已知有效的样例点位
URL = f'https://mapsv0.bdimg.com/?qt=pdata&sid={PID}&pos=0_0&z=3'

REQUESTS = 116272          # 全量采集的总瓦片请求数

try:
    t = time.time()
    body = urllib.request.urlopen(
        urllib.request.Request(URL, headers=H), timeout=20).read()
    dt = time.time() - t

    if len(body) > 5000:
        est = REQUESTS / 160 * dt / 60
        print(f'✓ 通  {len(body)} 字节 / {dt:.2f} 秒')
        print(f'  单请求 {dt:.2f}s，并发 160 下粗估采集 {est:.0f} 分钟')
        print()
        print('→ 走【路径 A】：继续跑 ④ ⑤')
    else:
        print(f'✗ 返回内容异常（{len(body)} 字节）')
        print('→ 走【路径 B】：跳到下面的 B 段说明')
except Exception as e:
    print(f'✗ 不通：{type(e).__name__}: {e}')
    print('→ 走【路径 B】：跳到下面的 B 段说明')
""", title="③ 测试能否直连百度")

# ── ④
md("""---
# 路径 A：Colab 直接采集

`sample_pool.jsonl`（含全部 21,746 个 panoID 与坐标）已在仓库里，所以 Colab
能自己去百度抓。

**采集、打包、备份必须在同一个会话里做完。** `/content` 随会话清空，
中途断线则全部重来（约 40 分钟）。这是省掉 4 GB 上传的代价。
""")
code("""
%%bash
set -e
cd /content/ai
python collect/fetch_pano.py meta --concurrency 32
python collect/fetch_pano.py images --concurrency 160
python collect/fetch_pano.py report
""", title="④【路径 A】采集（约 25–40 分钟）")

md("""期望看到 `done 21,746 / fail 0`。

若是首次运行且中断过，重跑本格会从断点续上（状态存在 sqlite 里）；
但**会话被回收后 `/content` 清空**，那种情况下只能从头再来。
""")

code("""
%%bash
set -e
cd /content/ai
python collect/pack_shards.py
mkdir -p /content/drive/MyDrive/sv/shards
cp data/shards/*.tar /content/drive/MyDrive/sv/shards/
cp data/shards/samples.jsonl /content/drive/MyDrive/sv/shards/
du -sh /content/drive/MyDrive/sv/shards
""", title="⑤【路径 A】打包并备份到 Drive")

# ── B
md("""---
# 路径 B：本地上传

如果第 ③ 格不通，**在手机/本机的 Termux 里**执行：

```sh
apt-get install -y rclone
rclone config
```

配置向导：`n` 新建 → 名字填 `gdrive` → 类型选 `drive` → 后面一路回车 →
最后在浏览器里打开它给的链接授权。

```sh
cd /data/data/com.termux/files/home/sv/ai
rclone copy data/shards gdrive:sv/shards --progress
```

4.25 GB，按 20 Mbps 上行约 30 分钟。

传完之后**回到本 notebook 继续跑 ⑥**，两条路径从这里汇合。
""")

# ── ⑥
md("""---
# ⑥ shards 从 Drive 拷到本地磁盘

**两条路径都要跑这一格。** Drive 是 FUSE 挂载，直读比本地磁盘慢 3–5 倍，
训练时逐 batch 从 Drive 读会拖垮吞吐。
""")
code("""
%%bash
mkdir -p /content/data
cp /content/drive/MyDrive/sv/shards/*.tar /content/data/ 2>/dev/null || true
cp /content/drive/MyDrive/sv/shards/samples.jsonl /content/data/ 2>/dev/null || true
cp /content/ai/data/shards/split.json /content/data/
""", title="⑥a 拷分片")

code("""
from pathlib import Path
tars = sorted(Path('/content/data').glob('*.tar'))
if not tars:
    raise SystemExit('没有分片——路径 B 的上传还没完成？')
print(f'{len(tars)} 个分片, {sum(t.stat().st_size for t in tars)/1e9:.2f} GB')
for f in ('samples.jsonl', 'split.json'):
    p = Path('/content/data') / f
    print(f'  {f}: {"✓" if p.exists() else "✗ 缺失"}')
""", title="⑥ 分片拷到本地磁盘")

# ── ⑦
md("""---
# ⑦ 冒烟测试（必做）

看进度条上的 **acc** 那一列。它跑的是训练集上的即时准确率，几十个 batch
之内就该冲到接近 1.0——**64 条样本背不下来就说明管线坏了，不是模型不行。**

这一步存在的意义是把"管线坏了"和"模型学不会"这两件完全不同的事分开。
跳过它，你会在几小时后面对一个指标很差的模型，无从判断该查数据还是调参。

`--eval-every 5` 是因为验证集上的前向和训练一样贵，而冒烟阶段不需要
每轮都验——训练准确率已经回答了"能不能过拟合"。
""")
code("""
%%bash
cd /content/ai
python train.py --config configs/default.yaml \\
    --data /content/data --points data/pool/county_points.npz \\
    --out /content/drive/MyDrive/sv/runs/smoke --overfit 64 --eval-every 5
""", title="⑦ 冒烟：64 条样本能否过拟合")

# ── ⑧
md("""---
# ⑧ 正式训练

`nohup ... &` 放后台，页面断开也不影响。每 25 分钟自动存 checkpoint 到 Drive。
""")
code("""
%%bash
cd /content/ai
nohup python train.py --config configs/default.yaml \\
    --data /content/data --points data/pool/county_points.npz \\
    --out /content/drive/MyDrive/sv/runs/base --resume > /content/train.log 2>&1 &
sleep 90
tail -25 /content/train.log
""", title="⑧ 正式训练（后台）")

# ── ⑨
md("""---
# ⑨ 看进度

每轮会打印这样一行：

```
e07 损失 2.1341 训练top1 0.093  112s (14s 等数据 / 98s 计算)  140 样本/s  剩余约 62 分钟
e07 同县留出 top1 0.021 top5 0.083 宏平均 0.015 多数类基线 0.012 中位误差 412.3 km
      候选邻近率(top5 落在真值 150km 内) 0.271
```

**先看"等数据 / 计算"那个比值。** 等数据占大头说明数据管线是瓶颈，
加 GPU 或调模型都没用；计算占大头才是正常的。这是判断该优化哪一端的
唯一证据。

然后看指标：

```

- **top-1 高于多数类基线 ≠ 模型在学地理**，它可能只是背下了样本最多的那个县。
  两个数必须并列看。
- **候选邻近率比 top-1 更早给出信号**：模型可能还点不中正确的县，但只要
  top-5 候选连成一片且围绕真值，就说明它在学地理。这个数长期贴着基线，
  说明模型没学到东西——此时加数据量没有用，该查标签与划分。

正式训练想让日志安静一点可以加 `--eval-every 2`。
""")
code("""
!tail -8 /content/train.log
""", title="⑨ 看训练进度")

code("""
import json
from pathlib import Path
p = Path('/content/drive/MyDrive/sv/runs/base/log.jsonl')
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    print(f'已完成 {len(rows)} 个 epoch')
    print(f'{"ep":>3} {"loss":>8} {"top1":>7} {"top5":>7} {"宏平均":>8} {"基线":>7} {"邻近率":>7} {"中位km":>8}')
    for r in rows[-12:]:
        v = r.get('val_same', {})
        print(f'{r["epoch"]:>3} {r.get("train_loss", 0):>8.4f} '
              f'{v.get("top1", 0):>7.3f} {v.get("top5", 0):>7.3f} '
              f'{v.get("macro_recall", 0):>8.3f} {v.get("majority_baseline", 0):>7.3f} '
              f'{v.get("top5_nearby_150km", 0):>7.3f} '
              f'{v.get("top1_median_km", float("nan")):>8.1f}')
else:
    print('还没有日志')
""", title="⑨b 指标曲线（读 Drive 上的日志）")

# ── ⑩
md("""---
# ⑩ 断线后重开：从这里继续

Colab 会话被回收是常态。重开 notebook 后依次跑 **① ② ⑥**，然后跑下面这格。
`--resume` 会从 Drive 上的 `last.pt` 接着跑，最多丢 25 分钟进度。
""")
code("""
%%bash
cd /content/ai
nohup python train.py --config configs/default.yaml \\
    --data /content/data --points data/pool/county_points.npz \\
    --out /content/drive/MyDrive/sv/runs/base --resume > /content/train.log 2>&1 &
sleep 60
tail -12 /content/train.log
""", title="⑩ 续跑")

# ── ⑪
md("""---
# ⑪ 推理

给一张街景图，输出县级分布与坐标。全景和截图都能吃，自动判别：

- 宽高比 ≥ 1.6 视为全景 → 切 8 个视图
- 否则视为截图 → 缩放为单视图

单视图能工作是**被训练过的能力**（训练时以 15% 概率只给单视图），
不是推理时才遇到的情形。
""")
code("""
IMAGE = '/content/drive/MyDrive/sv/test.jpg'   #@param {type:"string"}
""", title="⑪a 指定测试图")

code("""
%%bash
cd /content/ai
python inference.py --run /content/drive/MyDrive/sv/runs/base \\
    --image "{IMAGE}" --topk 5
""", title="⑪b 单图推理")

md("""没有测试图的话，随便抓一张出来：

```python
!python -c "
from data.shards import ShardIndex, decode_jpeg
from PIL import Image
import glob
idx = ShardIndex(sorted(glob.glob('/content/data/*.tar')))
k = idx.keys[1234]
Image.fromarray(decode_jpeg(idx.read(k))).save('/content/test.jpg')
print('已保存 /content/test.jpg，来自', k)
"
```

不过要注意：这张图是训练集里的，模型多半见过。**要测真实能力，
得用推理时没见过的地方**——`split.json` 里 `test_county` 的样本就是干这个的。
""")

# ── 排错
md("""---
# 排错

| 现象 | 原因 / 处理 |
|---|---|
| `nvidia-smi` 没输出 | 运行时没选 T4。菜单「代码执行程序 → 更改运行时类型」 |
| 克隆失败 | token 无效或过期；或仓库名写错。token 需要 `repo` 权限 |
| 第 ③ 格不通 | 走路径 B。百度对数据中心 IP 的行为不稳定 |
| 训练出现 NaN | 多半是 fp16 下的 softmax/log_softmax；本仓库的损失已强制 float32，若仍出现请连同日志反馈 |
| `CUDA out of memory` | 调小 `configs/default.yaml` 里的 `batch_size` 或 `views.size`。**别调 `accum`**——梯度累积不省显存 |
| 续跑后指标跳变 | 分片顺序漂移了；确认 `--seed` 没变、`split.json` 是仓库里那份 |
| Drive 写满 | 免费档 15 GB；删掉旧的 `best.pt` / `last.pt` |
| 分配不到 GPU | 免费档每周 15–40 GPU 小时且动态调整，高峰期只能等 |
| 训练比预计慢很多 | T4 是 2018 年的卡，瓶颈是算力不是显存 |
""")


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print(f"已写出 {OUT}：{len(CELLS)} 个单元格（{n_md} 个 markdown）")


if __name__ == "__main__":
    main()
