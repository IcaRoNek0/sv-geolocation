#!/usr/bin/env python
"""生成 colab.ipynb（Colab 与 Kaggle 通用）。

用脚本生成而不是手写 notebook JSON：手写容易在转义和结构上出错，而格式
错了在 Colab 里只会给一句含糊的导入失败。

改完本文件后运行（第二条是门禁，必跑）：
    python tools/make_notebook.py && python tools/check_notebook.py

写生成代码时避开两个已踩过多次的坑：

一、不要写反斜杠加 n。转义差一层就变成真实换行，把字符串劈成两半，在
    notebook 里报的是一句与真实原因毫不相干的语法错误。要换行就多写一条
    print，不要用转义。

二、单元格里要放 Python 的 docstring 时，不能直接用三引号——外层是
    code 加三引号的字符串，会被提前截断。用井号注释代替。
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

    单元格魔法（%%bash 等）必须是第一行，而 #@title 也要抢首行，两者不能
    共存：%%bash 前面插了 #@title 就不再被识别为魔法，整格按 Python 解析
    并报语法错误。因此带 %% 的格子不加 title。
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

**Colab 与 Kaggle 都能跑**，第 ① 格自动判别平台并设好路径变量，后面各格
都用它们。

**这个 notebook 是启动器，训练逻辑全在仓库的 `train.py` 里。** 改超参请改
`configs/default.yaml`——notebook 里改的东西没法版本管理。

---

## 怎么用

1. 打开 GPU：Colab「代码执行程序 → 更改运行时类型 → T4 GPU」；
   Kaggle 右侧 Settings → Accelerator → GPU
2. **Kaggle 还要打开网络**：Settings → Internet → On（默认关闭）
3. 从上往下跑 ①②③。第 ③ 格会告诉你是自己采集还是用已上传的分片
4. 之后按提示继续

**断线是常态**：Colab 4–6 小时被回收，Kaggle 9–12 小时。重开之后跑
**① → ② → ⑥ → ⑩**，训练自动从断点续上。

## 平台差异

| | Colab | Kaggle |
|---|---|---|
| 数据 | Drive（FUSE，慢 3–5×），要拷到本地磁盘 | `/kaggle/input`，**本地盘只读，快**，不用拷 |
| 网络 | 默认开 | **默认关** |
| 会话 | 4–6 小时 | 9–12 小时，适合正式训练 |
| 输出 | `/content` 随会话清空；checkpoint 要写 Drive | `/kaggle/working`，**需 Save Version 才保留** |
| GPU | T4 ×1 | T4 ×2 或 P100（本训练只用单卡） |
""")

# ── ①
md("## ① 环境检测")
code("""
import os, subprocess

ON_KAGGLE = os.path.isdir('/kaggle')
print('平台:', 'Kaggle' if ON_KAGGLE else ('Colab' if os.path.isdir('/content') else '未知'))

def find_dataset(root, marker='samples.jsonl'):
    # 在 /kaggle/input 下找出分片数据集的实际挂载点。
    # Kaggle 的挂载路径不止一种（见过 /kaggle/input/<slug>/，也见过
    # /kaggle/input/datasets/<owner>/<slug>/），所以按标志文件找，不写死。
    from pathlib import Path
    root = Path(root)
    if not root.is_dir():
        return None
    for p in sorted(root.rglob(marker)):
        return p.parent
    return None


if ON_KAGGLE:
    WORK = '/kaggle/working'
    RUNS = WORK + '/runs'                   # 需 Save Version 才保留
    DATA = find_dataset('/kaggle/input')
    if DATA is None:
        print('在 /kaggle/input 下没找到 samples.jsonl —— 数据集没挂上。')
        print('  → 右上 Add Input → Your Datasets → sv-shards')
        print('  → 挂上后重跑本格')
        raise SystemExit(1)
    DATA = str(DATA)
else:
    WORK = '/content'
    DATA = '/content/data'
    RUNS = '/content/drive/MyDrive/sv/runs'
    from google.colab import drive
    drive.mount('/content/drive')

AI_DIR = WORK + '/ai'

# 导出到环境变量：bash 格用 $VAR 引用。这样不依赖 IPython 的 {var} 展开
# 语义（本机无 IPython 无法验证），$VAR 走 shell 自己的展开，一定生效。
os.environ.update(WORK=WORK, DATA=DATA, RUNS=RUNS, AI_DIR=AI_DIR)

print(f'WORK   = {WORK}')
print(f'DATA   = {DATA}')  # 自动发现，不写死路径
print(f'RUNS   = {RUNS}')
print(f'AI_DIR = {AI_DIR}')

out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                      '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print('\\nGPU:', out or '(没拿到，检查运行时是否选了 GPU)')
try:
    import torch
    print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()}')
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        print(f'  GPU{i} {torch.cuda.get_device_name(i)} '
              f'cc{cc[0]}.{cc[1]} → {"bf16" if cc[0] >= 8 else "fp16"}')
except ImportError:
    print('未装 torch，由 ②b 安装')

# 网络：Kaggle 默认关闭；路径 A 的采集与 git clone 都需要它
try:
    import urllib.request
    urllib.request.urlopen('https://github.com', timeout=8)
    print('\\n网络：可用')
except Exception as e:
    print(f'\\n网络：不可用（{type(e).__name__}）')
    if ON_KAGGLE:
        print('  → Settings → Internet → On，然后重跑本格')
""", title="① 环境检测")

# ── ②
md("""## ② 取代码

仓库是私有的，需要 GitHub token（只读权限即可）。token 经 `getpass` 读入、
由 `subprocess` 传给 git，不会出现在输出里。仓库若已公开，直接回车跳过。
""")
code("""
import os, subprocess
from getpass import getpass

REPO = 'IcaRoNek0/sv-geolocation'
BRANCH = 'main'

if os.path.isdir(AI_DIR + '/.git'):
    print('代码已存在，拉取最新')
    r = subprocess.run(['git', '-C', AI_DIR, 'pull'], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip() or '(无输出)')
    if r.returncode != 0:
        # 拉取失败必须报错：静默失败会让你拿旧代码跑，还以为是新代码
        raise SystemExit('拉取失败（代码目录里有本地改动？）。'
                         '删掉 ' + AI_DIR + ' 重跑本格即可，分片数据不受影响。')
else:
    tok = getpass('GitHub token（仓库已公开则直接回车）: ').strip()
    url = (f'https://{tok}@github.com/{REPO}.git' if tok
           else f'https://github.com/{REPO}.git')
    # 用 subprocess 而非 ! 前缀：! 会把带 token 的完整命令打进输出
    r = subprocess.run(['git', 'clone', '-q', '-b', BRANCH, url, AI_DIR],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('克隆失败（token 无效？仓库名不对？网络没开？）\\n'
                         + r.stderr[-500:])

os.chdir(AI_DIR)


def _rev(ref):
    return subprocess.run(['git', 'rev-parse', '--short', ref],
                          capture_output=True, text=True).stdout.strip()


head, remote = _rev('HEAD'), _rev('origin/main')
print(f'当前代码 {head}   远程 {remote}')
if head != remote:
    print('⚠ 本地与远程不一致——拉取没成功，你跑的可能是旧代码')
""", title="② 取代码")

code("""
!pip install -q aiohttp timm pyyaml
""", title="②b 依赖")

# ── ③
md("""## ③ 连通性测试：决定走 A 还是 B

百度对境外/数据中心 IP 的行为未验证。花两分钟测一次：通过就走路径 A
（在云端自己采集，省掉 4.25 GB 上传），不通过就用已上传的分片走路径 B。
""")
code("""
import time, urllib.request

H = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.baidu.com/'}
PID = '09006900001410191436379165M'
URL = f'https://mapsv0.bdimg.com/?qt=pdata&sid={PID}&pos=0_0&z=3'
REQUESTS = 116272          # 全量采集的瓦片请求总数

try:
    t = time.time()
    body = urllib.request.urlopen(
        urllib.request.Request(URL, headers=H), timeout=20).read()
    dt = time.time() - t
    if len(body) > 5000:
        print(f'✓ 通  {len(body)} 字节 / {dt:.2f} 秒')
        print(f'  单请求 {dt:.2f}s，并发 160 下粗估采集 {REQUESTS/160*dt/60:.0f} 分钟\\n')
        print('→ 走【路径 A】：继续跑 ④ ⑤')
    else:
        print(f'✗ 返回内容异常（{len(body)} 字节）')
        print('→ 走【路径 B】：见下面的 B 段')
except Exception as e:
    print(f'✗ 不通：{type(e).__name__}: {e}')
    print('→ 走【路径 B】：见下面的 B 段')
""", title="③ 测试能否直连百度")

# ── ④⑤
md("""---
# 路径 A：云端直接采集

`sample_pool.jsonl`（含全部 21,746 个 panoID 与坐标）已在仓库里，所以云端
能自己去百度抓。

**采集、打包、备份必须在同一个会话里做完**：工作目录随会话清空，中途断线
则全部重来（约 40 分钟）。这是省掉 4 GB 上传的代价。
""")
code("""
%%bash
set -e
cd $AI_DIR
python collect/fetch_pano.py meta --concurrency 32
python collect/fetch_pano.py images --concurrency 160
python collect/fetch_pano.py report
""", title="④【路径 A】采集（约 25–40 分钟）")

md("""期望看到 `done 21,746 / fail 0`。会话内中断可重跑本格续上（状态存 sqlite），
但**会话被回收后工作目录清空**，只能从头再来。
""")

code("""
%%bash
set -e
cd $AI_DIR
python collect/pack_shards.py
mkdir -p $WORK/shards
cp data/shards/*.tar data/shards/samples.jsonl $WORK/shards/
du -sh $WORK/shards
echo
echo "分片已在 $WORK/shards —— Kaggle 上把它 Save Version 后可作为 Dataset 复用"
""", title="⑤【路径 A】打包并留在工作目录")

# ── B
md("""---
# 路径 B：用已上传的分片

第 ③ 格不通时走这条路。分片在本机 `ai/data/shards/`（9 个 tar +
`samples.jsonl`，共 4.25 GB），要让它出现在云端。

**Colab**：rclone 传到 Drive（Termux 里执行）

```sh
apt-get install -y rclone && rclone config     # n → gdrive → drive → 一路回车 → 浏览器授权
cd /data/data/com.termux/files/home/sv/ai
rclone copy data/shards gdrive:sv/shards --progress     # 约 30 分钟
```

**Kaggle**：做成 Dataset。一次上传之后每次会话都是本地盘挂载、不下载，
比每次重新拉一遍划算得多。

```sh
pip install --no-deps kaggle
pip install python-dateutil requests requests-toolbelt python-slugify \
            text-unidecode six bleach python-dotenv kagglesdk protobuf

# token 存这里（不是 kaggle.json 那两个字段）
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
printf '%s' '<你的 KGAT_ token>' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
python -m kaggle config view          # 确认 username 与 auth_method: ACCESS_TOKEN

cd /data/data/com.termux/files/home/sv/ai/data/shards
printf '%s\n' '{"title":"sv-shards","id":"<用户名>/sv-shards","licenses":[{"name":"other"}]}' \
  > dataset-metadata.json
python -m kaggle datasets create -p . --dir-mode skip
```

`--no-deps` 是必需的：kaggle 的传递依赖里有需要 Rust 编译的包，Termux 下
maturin 会因为平台判定（android vs linux-gnu）不一致而拒绝构建。

上传完成后在 notebook 右侧 **Add Input → Your Datasets → sv-shards**。

第 ① 格会**自动发现**挂载点（Kaggle 的路径不止一种，见过
`/kaggle/input/<slug>/` 也见过 `/kaggle/input/datasets/<owner>/<slug>/`），
按标志文件 `samples.jsonl` 找，不写死路径。找不到会直接报错并提示去
Add Input。

挂上之后继续跑 ⑥。
""")

# ── ⑥
md("""---
# ⑥ 让分片就位

**两条路径都要跑这一格。**

Colab 的 Drive 是 FUSE 挂载，直读比本地磁盘慢 3–5 倍，必须拷到本地。
Kaggle 的 `/kaggle/input` 本来就是本地盘，直接用，**不拷**。

**Kaggle 会把上传的 tar 自动解开**，数据集里是 `sv-0000/<panoID>.jpg` 这样的
散文件而非 tar。读取器两种布局都认（`open_index` 自动判别），不需要做任何
处理。散文件在这里没有性能问题：实测每条读取 0.33ms 对 tar 的 0.08ms，
而每条样本的处理总耗时约 50ms。
""")
code("""
%%bash
set -e
if [ -d /kaggle/input ]; then
  echo "Kaggle：/kaggle/input 是本地盘（只读），直接用，不拷贝"
else
  echo "Colab：从 Drive 拷到本地磁盘（Drive 直读慢 3-5 倍）"
  mkdir -p /content/data
  cp /content/drive/MyDrive/sv/shards/*.tar /content/data/ 2>/dev/null || true
  cp /content/drive/MyDrive/sv/shards/samples.jsonl /content/data/ 2>/dev/null || true
fi

# split.json 定义评估口径，必须与分片同目录。注意 /kaggle/input 是**只读**的，
# 所以 Kaggle 上它必须已经在 Dataset 里，不能像 Colab 那样临时拷进去。
if [ -f "$DATA/split.json" ]; then
  echo "split.json 已就位"
elif [ -w "$DATA" ]; then
  cp "$AI_DIR/data/shards/split.json" "$DATA/"
  echo "split.json 已从仓库拷入"
else
  # 报错前先诊断。只说"缺文件"没用，得让人知道该查哪里。
  echo "✗ $DATA 不可用" >&2
  echo "  DATA 存在: $([ -e "$DATA" ] && echo 是 || echo 否)" >&2
  if [ -d /kaggle/input ]; then
    echo "  /kaggle/input 内容: $(ls /kaggle/input 2>/dev/null | tr '\n' ' ')" >&2
    echo "  → 若上面没有 sv-shards：右上 Add Input → Your Datasets → sv-shards" >&2
    echo "  → 若有但名字不同：Dataset 的 slug 必须正好是 sv-shards" >&2
  else
    echo "  → 非 Kaggle 环境，检查 Drive 里是否已上传分片" >&2
  fi
  exit 1
fi
ls -la $DATA | head
""", title="⑥a 分片就位")

code("""
import sys
from pathlib import Path

sys.path.insert(0, AI_DIR)
from data.shards import open_index

EXPECTED = 21746          # 采集阶段的样本总数

for f in ('samples.jsonl', 'split.json'):
    p = Path(DATA) / f
    print(f'{f}: {"✓" if p.exists() else "✗ 缺失"}')

idx = open_index(DATA)
n = len(idx)
idx.close()
print()
print(f'索引到 {n:,} 条样本（期望 {EXPECTED:,}）')
if n != EXPECTED:
    # 宁可在这里停下，也不要训练到一半才发现数据不全
    raise SystemExit(f'✗ 差 {EXPECTED - n:,} 条 —— 上传没完成，或 Dataset 未处理完'
                     f'（Kaggle 建数据集是异步的，稍等几分钟再跑本格）')
print('✓ 数据完整')
""", title="⑥b 完整性核对")

# ── ⑦
md("""---
# ⑦ 冒烟测试（必做）

看日志里的 **训练top1**。它应该在几百步内冲到接近 1.0——64 条样本背不下来
就说明管线坏了，不是模型不行。

**不要看损失降没降到 0，它降不到。** 县级用地理软标签，目标是分布在若干
相邻县上的概率，其熵就是损失下限。实测这批样本的目标熵是 **4.34**（均匀
分布是 7.20）。判断学没学会看 top1，不是看损失。

**验证集那一格全 0 也是正常的**：验证集里的县和训练集完全不同，而冒烟
模式下模型只背下了训练的那 64 个县，按构造就预测不出来。

这一步把"管线坏了"和"模型学不会"这两件完全不同的事分开。跳过它，你会在
几小时后面对一个指标很差的模型，无从判断该查数据还是调参。
""")
code("""
%%bash
cd $AI_DIR
python train.py --config configs/default.yaml \\
    --data $DATA --points data/pool/county_points.npz \\
    --out $RUNS/smoke --overfit 64 --eval-every 5 \\
    > $WORK/smoke.log 2>&1
tail -40 $WORK/smoke.log
""", title="⑦ 冒烟：64 条样本能否过拟合")

# ── ⑧
md("""---
# ⑧ 正式训练

`nohup ... &` 放后台，页面断开不影响。每 25 分钟自动存 checkpoint。

时间账（据冒烟实测外推）：约 **18 分钟/轮**，40 轮约 12 小时。但正式训练每轮
1970 步，**跑 2 轮就等于冒烟的全部步数**，预计 10–15 轮收敛。

跑起来后盯 `val_same` 那一行，连续几轮不涨就可以停，`best.pt` 会自动保留
最好的一版。想少等就改 `configs/default.yaml` 的 `epochs`。
""")
code("""
%%bash
cd $AI_DIR
nohup python train.py --config configs/default.yaml \\
    --data $DATA --points data/pool/county_points.npz \\
    --out $RUNS/base --resume > $WORK/train.log 2>&1 &
sleep 90
tail -25 $WORK/train.log
""", title="⑧ 正式训练（后台）")

# ── ⑨
md("""---
# ⑨ 看进度

纯文本日志，每 15 秒一行：

```
[   37.4s] e00 20/1970 步 20/39400 损失 6.8421 训练top1 0.000 lr 3.0e-04 0.76s/batch
...
e07 损失 2.1341 训练top1 0.093  112s (14s 等数据 / 98s 计算)  140 样本/s  剩余约 62 分钟
e07 同县留出 top1 0.021 top5 0.083 宏平均 0.015 多数类基线 0.012 中位误差 412.3 km
      候选邻近率(top5 落在真值 150km 内) 0.271
```

**先看"等数据 / 计算"的比值**——它决定该优化哪一端。等数据占大头说明瓶颈
在数据管线，加 GPU 或调模型都没用。

再看指标：

- **top-1 高于多数类基线 ≠ 模型在学地理**，可能只是背下了样本最多的县。
  两个数必须并列看。
- **候选邻近率比 top-1 更早给出信号**：模型可能还点不中正确的县，但只要
  top-5 候选连成一片且围绕真值，就说明在学地理。长期贴着基线说明没学到
  东西——此时加数据量没用，该查标签与划分。
""")
code("""
# 可反复重跑
!tail -12 $WORK/train.log
""", title="⑨ 看训练进度")

code("""
import json
from pathlib import Path

p = Path(RUNS) / 'base' / 'log.jsonl'
if not p.exists():
    print('还没有日志')
else:
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    print(f'已完成 {len(rows)} 个 epoch')
    hdr = f'{"ep":>3} {"损失":>8} {"训练top1":>8} {"top1":>7} {"top5":>7} {"宏平均":>8} {"基线":>7} {"邻近率":>7} {"中位km":>8}'
    print(hdr)
    for r in rows[::max(1, len(rows)//25)] + rows[-1:]:
        v = r.get('val_same', {})
        print(f'{r["epoch"]:>3} {r.get("train_loss",0):>8.3f} '
              f'{r.get("train_acc",0):>8.3f} {v.get("top1",0):>7.3f} '
              f'{v.get("top5",0):>7.3f} {v.get("macro_recall",0):>8.3f} '
              f'{v.get("majority_baseline",0):>7.3f} '
              f'{v.get("top5_nearby_150km",0):>7.3f} '
              f'{v.get("top1_median_km",float("nan")):>8.1f}')
""", title="⑨b 指标表（读日志）")

# ── ⑩
md("""---
# ⑩ 断线后重开

会话被回收是常态。重开 notebook 后依次跑 **① ② ⑥**，然后跑本格。
`--resume` 从上次的 `last.pt` 接着跑，最多丢 25 分钟进度。

**Kaggle 注意**：`/kaggle/working` 只有 Save Version 才保留。长时间训练
请中途 Save Version 一次，否则会话结束后 checkpoint 会丢。
""")
code("""
%%bash
cd $AI_DIR
nohup python train.py --config configs/default.yaml \\
    --data $DATA --points data/pool/county_points.npz \\
    --out $RUNS/base --resume > $WORK/train.log 2>&1 &
sleep 60
tail -12 $WORK/train.log
""", title="⑩ 续跑")

# ── ⑪
md("""---
# ⑪ 推理

自动判别输入形态：宽高比 ≥1.6 视为全景（切多视图），否则当作单视图截图。
单视图能工作是**被训练过的能力**（训练时以 15% 概率只给单视图）。
""")
code("""
IMAGE = '/content/test.jpg'   #@param {type:"string"}
""", title="⑪a 指定测试图")

code("""
%%bash
cd $AI_DIR
python inference.py --run $RUNS/base --image "{IMAGE}" --topk 5
""", title="⑪b 单图推理")

md("""没有测试图的话从分片里取一张：

```python
import glob, sys
sys.path.insert(0, AI_DIR)
from data.shards import ShardIndex, decode_jpeg
from PIL import Image
idx = ShardIndex(sorted(glob.glob(DATA + '/*.tar')))
k = idx.keys[1234]
Image.fromarray(decode_jpeg(idx.read(k))).save(WORK + '/test.jpg')
print('已保存', k)
```

但注意这是**训练集里的图，模型多半见过**。要测真实能力得用没见过的
地方——`test_county` 划分就是干这个的。
""")

md("""---
# ⑪c 用指定 panoID 做测试

从分片里取出指定的全景跑推理，并把**真值和所属划分一起打出来**。

注意：如果这条落在 `train` 划分里，模型训练时见过它，预测对了只说明推理
链路是通的，不说明泛化能力。格子会自己判断并提示——要测真实水平请把 PID
换成 `val_same` 或 `test_county` 里的样本。

（若训练正在跑，checkpoint 每 25 分钟被重写一次，极小概率读到写了一半的
文件。真遇到就重跑本格。）
""")

code("""
PID = '09019300011610251337528397P'   #@param {type:"string"}
""", title="⑪c 指定 panoID")

code("""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 没跑过 ① 格（或 kernel 重启过）就按平台推默认值，免得为一个变量卡住
if 'AI_DIR' not in globals():
    print('没找到 ① 格设的变量，按平台推默认值')
    _work = '/kaggle/working' if os.path.isdir('/kaggle') else '/content'
    AI_DIR, RUNS = _work + '/ai', _work + '/runs'
    DATA = None
    for _p in Path('/kaggle/input').rglob('samples.jsonl'):
        DATA = str(_p.parent)
        break
    if DATA is None:
        DATA = '/content/data'
    print(f'AI_DIR={AI_DIR}')
    print(f'DATA={DATA}')
    print(f'RUNS={RUNS}')

sys.path.insert(0, AI_DIR)
from data.prepare import ViewConfig
from data.shards import open_index
from inference import load_run, pick_amp_dtype, to_views
from models.env_model import EnvModel
from models.fusion import fuse, predict_location, top_counties
from utils.geo_utils import CountyPoints, haversine

# 真值与所属划分
samples = {json.loads(l)['panoid']: json.loads(l) for l in
           open(Path(AI_DIR) / 'data/shards/samples.jsonl', encoding='utf-8')}
split = json.load(open(Path(AI_DIR) / 'data/shards/split.json',
                       encoding='utf-8'))['assignments']
gt = samples.get(PID)
which = split.get(PID, '(不在数据集中)')
print('panoID:', PID)
print('真值  :', gt)
print('划分  :', which)
if which == 'train':
    print()
    print('注意：这条在训练集里，模型见过它。预测对了只说明推理链路是通的，')
    print('      不说明泛化能力。测真实水平请换 val_same / test_county 的样本。')

# 取图
idx = open_index(DATA)
pano = idx.read(PID)
idx.close()
img = np.asarray(Image.open(io.BytesIO(pano)).convert('RGB'))
is_pano = img.shape[1] / img.shape[0] >= 1.6
print()
print(f'已取出 {img.shape[1]}x{img.shape[0]} '
      f'{"全景" if is_pano else "截图"}  {len(pano)/1024:.0f} KB')

# 推理
meta, ckpt = load_run(Path(RUNS) / 'base')
vcfg = ViewConfig(**meta.get('views', {}))
views, vmask, n = to_views(img, is_pano, 8, vcfg.fov, vcfg.size, vcfg.n_max)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnvModel(len(meta['adcodes']), len(meta['cities']),
                 len(meta['provinces']), backbone=meta['backbone'],
                 pretrained=False).to(device)
state = torch.load(ckpt, map_location=device, weights_only=False)
model.load_state_dict(state['model'])
model.eval()
print(f'checkpoint {ckpt.name}  epoch {state.get("epoch")}')

amp = pick_amp_dtype(device)
ctx = (torch.autocast(device_type='cuda', dtype=amp) if amp is not None
       else contextlib.nullcontext())
with torch.no_grad(), ctx:
    out = model(torch.from_numpy(views).permute(0, 3, 1, 2)[None].to(device),
                torch.from_numpy(vmask)[None].to(device))

probs = fuse(out['county'][0].float().cpu().numpy(), adcodes=meta['adcodes'])
cp = CountyPoints(Path(AI_DIR) / 'data/pool/county_points.npz')
top = top_counties(probs, k=5)
lon, lat, used = predict_location(probs, cp, top_k=20)

print()
print(f'{n} 个视图，县级候选：')
for a, prob in top:
    tag = '   <- 真值' if gt and a == gt['adcode'] else ''
    print(f'  {a}  {prob*100:5.1f}%{tag}')
print()
print(f'选点   经度 {lon:.5f}  纬度 {lat:.5f}')
if gt and gt.get('lng') is not None:
    d = haversine(lon, lat, gt['lng'], gt['lat']) / 1000
    hit = '命中' if gt['adcode'] in [a for a, _ in top] else '未命中'
    print(f'真值   经度 {gt["lng"]:.5f}  纬度 {gt["lat"]:.5f}')
    print(f'误差   {d:.1f} km    top5 {hit}')
""", title="⑪c 取图并推理")

md("""---
# ⑪d 导出 ONNX（给本地推理）

如果本地机器装不上 torch（实测 Termux/ARM 上 apt 的 python3-torch 一 import
就 SIGSEGV），可以把模型导成 ONNX，本地只用 onnxruntime——它在这类环境里
正常。

导出后本地要下载的文件：**`model.onnx`（约 110 MB）+ `classes.json`（几 KB）**。
`classes.json` 不能漏：没有它，1337 个 logit 无法映回 adcode，预测会被
静默解释成别的县。选点用的 `county_points.npz` 已在代码仓库里，不用下。
""")

code("""
%%bash
cd $AI_DIR
python tools/export_onnx.py --run $RUNS/base
ls -la $RUNS/base/model.onnx
""", title="⑪d 导出 ONNX")

md("""导出后从 notebook 的 **Output** 标签下载 `runs/base/model.onnx` 和
`runs/base/classes.json`，放到本地同一个目录，然后：

```sh
python inference_onnx.py --run <那个目录> --image pano.jpg --topk 5
python inference_onnx.py --run <目录> --image pano.jpg \
    --truth 511424,103.510898,30.023553     # 传真值就直接算误差
```

本地需要：`onnxruntime`、`numpy`、`PIL`。不需要 torch。
""")

# ── ⑫
md("""---
# ⑫ 保存与恢复

## `/kaggle/working` 本身不持久

会话结束或重新打开 notebook，它就没了。要留下必须"存进版本"。

| | Quick Save | Save & Run All (Commit) |
|---|---|---|
| 做什么 | 只存 notebook 状态，**不重跑** | **从头重跑所有单元格** |
| 输出文件 | 默认不存；Advanced Settings 里勾 **"Save output for this version"** 才存 | 存 |

**不要点 Save & Run All** —— 本 notebook 的 ⑧ 格用 nohup 后台训练，
重跑会从头再练一遍，而后台进程在 notebook 跑完时会被杀掉。

**训练中途保存**：`Save Version → Quick Save`，并在弹窗的 Advanced Settings
里勾上 **"Save output for this version"**。这是唯一能在不重跑的前提下把
checkpoint 存进版本的办法。之后去 notebook 页面的 **Output** 标签下载。

**要跨会话续跑**，Quick Save 不够——有报告说保存的输出只留几小时。稳妥
做法是把它传成 Dataset 版本（下面两格）。

## 体积

`best.pt` 只存权重（约 112 MB），`last.pt` 含优化器状态（约 340 MB，
续跑必需）。同步默认只传前者加日志，够用；要跨会话续跑再带上 last.pt。
""")

code("""
import json, shutil, subprocess
from pathlib import Path

run_dir = Path(RUNS) / 'base'
if not (run_dir / 'best.pt').exists():
    raise SystemExit(f'{run_dir} 下还没有 best.pt —— 训练跑过了吗？')

out = subprocess.run(['python', '-m', 'kaggle', 'config', 'view'],
                     capture_output=True, text=True).stdout
user = next((l.split(':', 1)[1].strip() for l in out.splitlines()
             if 'username' in l), '')
if not user:
    raise SystemExit('拿不到 Kaggle 用户名，检查 ~/.kaggle/access_token')
SLUG = f'{user}/sv-runs'

stage = Path('/kaggle/working/_sync')
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir(parents=True)
for f in ('best.pt', 'log.jsonl', 'classes.json'):
    src = run_dir / f
    if src.exists():
        shutil.copy(src, stage / f)
# 要跨会话续跑就取消下面这行的注释（多传 340 MB）
# shutil.copy(run_dir / 'last.pt', stage / 'last.pt')

(stage / 'dataset-metadata.json').write_text(json.dumps(
    {'title': 'sv-runs', 'id': SLUG, 'licenses': [{'name': 'other'}]}),
    encoding='utf-8')
print('待同步:', sorted(x.name for x in stage.iterdir()))

listing = subprocess.run(['python', '-m', 'kaggle', 'datasets', 'list', '--mine'],
                         capture_output=True, text=True).stdout
existing = SLUG in listing
cmd = ['version', '-m', 'sync'] if existing else ['create']
print(f'{"更新" if existing else "首次创建"} {SLUG}')
subprocess.run(['python', '-m', 'kaggle', 'datasets'] + cmd
               + ['-p', str(stage), '--dir-mode', 'skip'])
""", title="⑫a 把训练结果同步成 Dataset（可反复跑）")

md("""同步完成后，去 https://www.kaggle.com/datasets/`<用户名>`/sv-runs 把可见性
设为 **Private**（默认就是），下个会话 Add Input 挂上即可。

## 恢复并续跑

挂上 sv-runs 之后跑下一格：把 checkpoint 从只读的 `/kaggle/input` 拷回
可写的 `$RUNS`，再跑 ⑩ 格续训。
""")

code("""
import shutil
from pathlib import Path

src = find_dataset('/kaggle/input', marker='best.pt')
if src is None:
    print('没找到 sv-runs 数据集 —— 先 Add Input 挂上，或本会话不需要恢复')
else:
    run_dir = Path(RUNS) / 'base'
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ('best.pt', 'last.pt', 'log.jsonl', 'classes.json'):
        p = Path(src) / f
        if p.exists():
            shutil.copy(p, run_dir / f)
            print(f'  恢复 {f}  {p.stat().st_size/1e6:.0f} MB')
    print()
    print(f'已恢复到 {run_dir}')
    print('跑 ⑩ 格继续训练（--resume 会从 last.pt 接上）')
""", title="⑫b 从 Dataset 恢复并续跑")

# ── 排错
md("""---
# 排错

| 现象 | 原因 / 处理 |
|---|---|
| 第 ① 格报"网络不可用" | Kaggle 默认关网。Settings → Internet → On，重跑 ① |
| `nvidia-smi` 没输出 | 运行时没选 GPU。Colab：代码执行程序 → 更改运行时类型；Kaggle：Settings → Accelerator |
| 克隆失败 | token 无效或过期（需要 `repo` 权限）；或网络没开 |
| 第 ③ 格不通 | 走路径 B |
| **启动先静默十几秒** | 正常（import torch/timm、扫 9 个分片索引、fork worker）。每个阶段都有打点，长时间停在某一行就是卡在那里 |
| 训练出现 NaN | 多半是 fp16 下的 softmax；本仓库的损失已强制 float32 |
| `CUDA out of memory` | 调小 `batch_size` 或 `views.size`。**别调 `accum`**——梯度累积不省显存 |
| 续跑后指标跳变 | 分片顺序漂移了；确认 `--seed` 与 `split.json` 没变 |
| Kaggle 找不到分片 | Dataset 的 slug 要叫 `sv-shards`，且要 Add Input 到本 notebook |
| Kaggle 数据集里是散文件不是 tar | 正常，Kaggle 上传后会自动解开。读取器两种布局都认 |
| Kaggle 训练完 checkpoint 没了 | `/kaggle/working` 需 Save Version 才保留 |
| 分配不到 GPU | Colab 免费档每周 15–40 GPU 小时且动态调整；Kaggle 每周 30 小时 |
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
