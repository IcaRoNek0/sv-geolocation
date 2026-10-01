#!/usr/bin/env python
"""Generate the Kaggle-first PyTorch notebook and its Colab-compatible alias."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def md(text):
    CELLS.append({'cell_type':'markdown', 'metadata':{}, 'source':text.strip().splitlines(True)})


def code(text):
    CELLS.append({'cell_type':'code', 'metadata':{}, 'execution_count':None,
                  'outputs':[], 'source':text.strip().splitlines(True)})


md('''# 街景定位第三轮：PyTorch / Kaggle 双 T4

训练产物以 `best.pt` / `last.pt` 为主，ONNX 仅用于 Termux 推理。
先在 Settings 选择 **GPU T4 x2**、Internet On，再 Add Input 挂载原有街景数据集。
当前数据为 124,751 张、26 个 tar（Kaggle 自动解包也支持）。不要重新划分。

按顺序运行所有格。正式训练在前台执行，支持 Save Version → Save & Run All，
不会因为后台命令提前结束而丢训练。先完成环境模型，再用另一个版本运行车辆模型。
云端实际可用时长取决于账户配额，代码不预设“几小时必能训完”。
''')
code('''from pathlib import Path
import os

TASK = 'environment'           # environment 或 vehicle
RUN_NAME = 'round3_env_v1'       # vehicle 时改成 vehicle_v1
DATA_OVERRIDE = ''             # 多个数据集时填写包含 samples.jsonl 的目录
RESUME_FROM = ''               # 上次输出中含 last.pt/classes.json 的目录（Add Input 挂载）
INIT_WEIGHTS = ''              # 可选 round2/best.pt，仅初始化环境模型权重；不能填 ONNX
RUN_SMOKE = True
RUN_BENCHMARK = False         # 排查慢速时开启：环境模型双卡纯计算 A/B，无图片读取
RUN_TRAINING = True
EPOCHS = 15 if TASK == 'environment' else 10
BATCH_SIZE = 8 if TASK == 'environment' else 32   # 每张 GPU 的样本数
WORKERS = 2                    # 每个 rank 的 CPU worker 数
EXPORT_ONNX = False            # 仅训练完成后需要 Termux 产物时开启
assert TASK in ('environment', 'vehicle')
assert not (RESUME_FROM and INIT_WEIGHTS)
ON_KAGGLE = Path('/kaggle').exists()
if ON_KAGGLE:
    WORK = Path('/kaggle/working')
    INPUT = Path('/kaggle/input')
    RUNS = WORK / 'runs'
else:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = Path('/content')
    INPUT = Path('/content/drive/MyDrive/sv/shards')
    RUNS = Path('/content/drive/MyDrive/sv/runs')
AI_DIR = WORK / 'ai'
OUT = RUNS / RUN_NAME
CONFIG = 'configs/round3.yaml' if TASK == 'environment' else 'configs/vehicle.yaml'
print('任务:', TASK, '输出:', OUT)
''')
md('''## 1. 获取代码
公开仓库直接克隆；私有仓库在 Kaggle Add-ons → Secrets 添加 `GITHUB_TOKEN` 并授权本笔记本。
Token 只通过子进程环境传递，不写入 remote URL、单元格输出或 git 配置文件。
更新采用 `pull --ff-only`；本地代码有修改时需自行处理，不会重置代码。
''')
code('''import base64
import subprocess
import sys

REPO = 'https://github.com/IcaRoNek0/sv-geolocation.git'
git_env = os.environ.copy()
git_env['GIT_TERMINAL_PROMPT'] = '0'
token = ''
if ON_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('GITHUB_TOKEN')
    except Exception:
        pass
else:
    from getpass import getpass
    token = getpass('私有仓库 token（公开仓库直接回车）: ').strip()
if token:
    encoded = base64.b64encode(('x-access-token:' + token).encode()).decode()
    git_env.update(GIT_CONFIG_COUNT='1',
                   GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + encoded)
if (AI_DIR / '.git').exists():
    cmd = ['git', '-C', str(AI_DIR), 'pull', '--ff-only']
else:
    cmd = ['git', 'clone', '--branch', 'main', REPO, str(AI_DIR)]
result = subprocess.run(cmd, env=git_env, capture_output=True, text=True)
if result.returncode:
    raise RuntimeError('Git 更新失败：检查网络、Secrets 权限或已有代码改动；未输出认证信息。')
del token, git_env
if 'encoded' in globals():
    del encoded
os.chdir(AI_DIR)
sys.path.insert(0, str(AI_DIR))
subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)
''')
md('''## 2. 依赖与 GPU 检查
使用 Kaggle 自带的 PyTorch/CUDA，不主动替换。缺少两张卡时明确显示实际卡数；
单卡仍能运行，但有效 batch 会改变，不能接着双卡的优化器状态续跑。
''')
code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'timm>=1.0,<2', 'pyyaml>=6', 'scipy', 'Pillow>=10'], check=True)
import torch
import timm
assert torch.cuda.is_available(), '请启用 GPU T4 x2'
GPUS = torch.cuda.device_count()
print('PyTorch', torch.__version__, 'timm', timm.__version__, 'GPU 数', GPUS)
for i in range(GPUS):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 2**20, 'MiB')
if ON_KAGGLE and GPUS != 2:
    print('当前不是双卡；如需双 T4，请停止并修改 Accelerator。')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], check=True)
''')
md('''## 3. 数据检查
自动发现含 `samples.jsonl` 和 `split.json` 的目录；多个候选时要求明确指定。
Kaggle 直接读只读输入。Colab 将分片拷到本地，避免训练期间直接读 Drive。
检查图像 key、元数据、划分和车辆／日期组的一致性，不仅检查总数。
''')
code('''import shutil

if DATA_OVERRIDE:
    DATA = Path(DATA_OVERRIDE)
else:
    candidates = sorted({p.parent for p in INPUT.rglob('samples.jsonl')
                         if (p.parent / 'split.json').exists()})
    if len(candidates) != 1:
        raise RuntimeError('数据目录不唯一，请设置 DATA_OVERRIDE。候选: ' + str(candidates))
    DATA = candidates[0]
if not ON_KAGGLE:
    local_data = WORK / 'data'
    local_data.mkdir(exist_ok=True)
    for p in DATA.iterdir():
        if p.is_file() and (p.suffix == '.tar' or p.name in ('samples.jsonl', 'split.json', 'manifest.json')):
            dest = local_data / p.name
            if not dest.exists() or dest.stat().st_size != p.stat().st_size:
                shutil.copy2(p, dest)
    DATA = local_data
print('数据:', DATA)
subprocess.run([sys.executable, 'tools/preflight.py', '--data', str(DATA)], check=True)
''')
md('''## 3.5 可选双卡性能诊断
训练慢时设 `RUN_BENCHMARK=True`，按顺序比较 packed / dense；每项预热 10 步、测量 40 步。
两项均关闭 cuDNN 自动调优、不下载预训练权重、不写 checkpoint，不衡量准确率。
合成输入只测计算与 DDP，真实训练还包含图片读取/增强，结果不能当完整训练速度。
默认仍使用 packed；先看正式训练的 `recent20 / data / work`，再决定是否调整。
''')
code('''if RUN_BENCHMARK and TASK == 'environment':
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    for mode in ('packed', 'dense'):
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=' + str(GPUS), 'tools/bench_train.py',
               '--config', CONFIG, '--batch-size', str(BATCH_SIZE), '--mode', mode]
        log_path = WORK / ('benchmark_' + mode + '.log')
        with log_path.open('w') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=env)
            try:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
        if status:
            raise RuntimeError('基准失败，详见 ' + str(log_path))
''')
md('''## 4. 双卡短冒烟
2 轮 × 64 张用于验证前向、反向、DDP 汇总、checkpoint 和验证流程，不要求已过拟合。
长过拟合实验可单独使用 `--overfit 64 --epochs 200`，不要将其验证准确率当模型效果。
默认正式训练为独立目录，不会从冒烟模型继续。
''')
code('''def launch(extra, log_path):
    import time
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node=' + str(GPUS), 'train.py', '--config', CONFIG,
           '--data', str(DATA), '--workers', str(WORKERS), '--batch-size', str(BATCH_SIZE)] + extra
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    print('执行:', ' '.join(cmd))
    with log_path.open('a') as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
        try:
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    if status:
        raise RuntimeError('训练退出码 ' + str(status) + '，详见 ' + str(log_path))

if RUN_SMOKE:
    smoke = RUNS / ('smoke_' + TASK)
    extra = ['--out', str(smoke), '--overfit', '64', '--epochs', '2', '--eval-every', '1']
    if (smoke / 'last.pt').exists():
        extra.append('--resume')
    launch(extra, WORK / ('smoke_' + TASK + '.log'))
    assert (smoke / 'last.pt').exists()
''')
md('''## 5. 正式训练 / 续跑
首次训练留空 RESUME_FROM。续跑时 Add Input 挂载上次 notebook 输出，把
RESUME_FROM 指向含 `last.pt/classes.json` 的目录；代码会复制到可写工作区。
同一会话 OUT 已有 last.pt 时自动续跑。续跑必须保持数据版本、配置、卡数、batch 和总轮数一致。
如需换损失或轮数，用新 RUN_NAME 和 INIT_WEIGHTS 从 `.pt` 初始化，重建优化器。

每 25 分钟及每轮结束存 last.pt，只有 rank 0 写盘；断点记录批次，但不承诺 dropout
随机流逐位复现。查看日志中的 samples_per_sec、gpu_peak_gb 和等数据时间来调 batch/worker，
双卡不是把显存合成一块。环境模型初始有效 batch 为 8×2×2=32。
''')
code('''if RESUME_FROM:
    source = Path(RESUME_FROM)
    assert (source / 'last.pt').exists() and (source / 'classes.json').exists()
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('目标目录非空，避免覆盖已有训练；请选择新 RUN_NAME')
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('last.pt', 'best.pt', 'classes.json', 'config.json', 'log.jsonl'):
        if (source / filename).exists():
            shutil.copy2(source / filename, OUT / filename)
if RUN_TRAINING:
    extra = ['--out', str(OUT), '--epochs', str(EPOCHS)]
    if (OUT / 'last.pt').exists():
        extra.append('--resume')
    elif INIT_WEIGHTS:
        extra.extend(['--init', str(INIT_WEIGHTS)])
    launch(extra, WORK / (RUN_NAME + '.log'))
''')
md('''## 6. 检查与保存产物
环境模型按 `val_national_single/top1` 选 best.pt；同时查看全景、同县、宏平均和长尾误差。
车辆模型按 `val_national/macro_recall` 选模型，并查看已知车号覆盖率与年份准确率。
两种任务都保留完整类别表。没有车辆模型权重时不要开启车辆融合。

Kaggle `/kaggle/working` 并非永久磁盘。用 **Save Version → Save & Run All** 保存运行后的
输出；会话意外结束前没有已保存版本的文件不能保证恢复。重要 checkpoint 可手动下载，
或在已有版本完成后将其输出 Add Input 到下一次运行。只复制输入输出文件，不能“原地”写入挂载数据集。
''')
code('''import json

log = OUT / 'log.jsonl'
if log.exists():
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    for row in rows[-5:]:
        print('epoch', row['epoch'], 'train_top1', row.get('train_acc'),
              'samples/s', row.get('samples_per_sec'), 'peakGB', row.get('gpu_peak_gb'))
        for key in ('val_national_single', 'val_national', 'val_same', 'test_county'):
            if key in row:
                print(key, row[key])
for name in ('best.pt', 'last.pt', 'classes.json', 'config.json'):
    p = OUT / name
    print(name, p.stat().st_size if p.exists() else '未生成')
if RUN_TRAINING:
    assert (OUT / 'best.pt').exists() and (OUT / 'last.pt').exists()
''')
md('''## 7. 可选：导出 Termux ONNX
主要训练产物仍为 `.pt`。需要手机推理时才打开第一格的 EXPORT_ONNX。
导出会验证单视图和满视图下 PyTorch/ONNX 的数值一致性；下载 model.onnx 与同目录 classes.json。
街景车分支使用 `inference_vehicle.py --backend onnx`；PyTorch 是其默认后端。
''')
code('''if EXPORT_ONNX:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-export.txt'], check=True)
    subprocess.run([sys.executable, 'tools/export_onnx.py', '--run', str(OUT)], check=True)
else:
    print('跳过 ONNX；best.pt / last.pt 是本轮训练产物。')
''')


def main():
    for i, cell in enumerate(CELLS):
        cell['id'] = f'sv-round3-{i:02d}'
    nb = {'nbformat':4, 'nbformat_minor':5,
          'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
                      'language_info':{'name':'python'},
                      'accelerator':'GPU'}, 'cells':CELLS}
    for name in ('kaggle.ipynb', 'colab.ipynb'):
        (ROOT/name).write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
        print('Generated', name)


if __name__ == '__main__':
    main()
