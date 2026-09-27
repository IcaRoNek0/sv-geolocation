"""WebDataset 分片的随机读取索引。纯标准库，可脱离 torch 测试。

扫一遍 tar 头建立 panoid → (分片, 偏移, 长度)，之后按偏移读取，不解包
（两万多个小文件解到磁盘既慢又占空间）。
"""
import io
import json
import os
import tarfile
from pathlib import Path


class ShardIndex:
    """扫描分片建立索引，按 key 随机读取成员。"""

    def __init__(self, shard_paths, want_json=False, progress=None):
        self.shards = [Path(p) for p in sorted(shard_paths)]
        if not self.shards:
            raise ValueError("没有找到任何分片")
        self.keys = []
        self._index = {}
        self._meta = {}
        self._fds = {}
        self._scan(want_json, progress)

    def _scan(self, want_json, progress=None):
        """扫 tar 头建偏移索引。

        want_json 默认关：训练用不到侧车 JSON（元数据来自 samples.jsonl），
        实测只省 0.1s，是清理而非优化。
        """
        for si, path in enumerate(self.shards):
            if progress:
                progress(si, len(self.shards), path.name)
            with tarfile.open(path, "r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = member.name
                    if name.endswith(".jpg") or name.endswith(".jpeg"):
                        key = name.rsplit(".", 1)[0]
                        self._index[key] = (si, member.offset_data, member.size)
                        self.keys.append(key)
                    elif want_json and name.endswith(".json"):
                        key = name.rsplit(".", 1)[0]
                        fh = tf.extractfile(member)
                        if fh is not None:
                            try:
                                self._meta[key] = json.loads(fh.read())
                            except json.JSONDecodeError:
                                pass
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self._index

    def meta(self, key):
        return self._meta.get(key, {})

    def _fd(self, si):
        fd = self._fds.get(si)
        if fd is None:
            fd = self._fds[si] = os.open(self.shards[si], os.O_RDONLY)
        return fd

    def read(self, key):
        """读出该样本的 JPEG 字节。

        必须用 os.pread 而非 seek+read：worker 是 fork 出来的，文件偏移量
        跨进程共享，seek+read 会让两个 worker 互相读到对方的字节——数据
        静默损坏而训练照常跑。

        长度校验同样是承重的：tarfile 对截断归档不报错，索引会指向文件
        末尾之外。
        """
        si, offset, size = self._index[key]
        blob = os.pread(self._fd(si), size, offset)
        if len(blob) != size:
            raise IOError(f"{key} 读取不完整：期望 {size} 字节，实得 {len(blob)}")
        return blob

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class FileIndex:
    """解包后的散文件布局：<子目录>/<key>.jpg（+ <key>.json）。

    Kaggle 上传 tar 后会**自动解开**，数据集里就是这种布局。与其重传，不如
    让读取器两种都认。散文件在这里没有性能问题——/kaggle/input 是本地盘，
    不像 Colab 的 Drive 是 FUSE。
    """

    def __init__(self, dirs, want_json=False, progress=None):
        self.keys = []
        self._paths = {}
        self._meta = {}
        for i, d in enumerate(sorted(Path(x) for x in dirs)):
            if progress:
                progress(i, len(dirs), d.name)
            for f in sorted(d.iterdir()):
                if f.suffix in (".jpg", ".jpeg"):
                    self._paths[f.stem] = f
                    self.keys.append(f.stem)
                elif want_json and f.suffix == ".json":
                    try:
                        self._meta[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self._paths

    def meta(self, key):
        return self._meta.get(key, {})

    def read(self, key):
        return self._paths[key].read_bytes()

    def close(self):
        pass


def open_index(data_dir, want_json=False, progress=None):
    """按布局自动选读取器：优先 tar 分片，其次解包后的目录。"""
    data_dir = Path(data_dir)
    tars = sorted(data_dir.glob("*.tar"))
    if tars:
        return ShardIndex(tars, want_json, progress)
    subs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if subs:
        return FileIndex(subs, want_json, progress)
    raise ValueError(f"{data_dir} 下既没有 .tar 分片，也没有解包后的子目录")


def load_samples(path):
    """读取 samples.jsonl，返回以 panoid 为键的字典。"""
    out = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["panoid"]] = r
    return out


def load_split(path):
    """读取 split.json，返回 (划分字典, 完整 payload)。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["assignments"], payload


def decode_jpeg(blob, min_width=None):
    """JPEG 字节 → (H, W, 3) uint8。

    min_width 给定时用 PIL draft 在 DCT 域按 1/2、1/4 直接解码，既省解码
    时间，也让后续采样要转 float32 的数据量成倍减少。
    """
    from PIL import Image
    import numpy as np
    with Image.open(io.BytesIO(blob)) as im:
        if min_width:
            # draft 的缩放是 1/1、1/2、1/4、1/8，且必须在 load 之前调用
            for scale in (2, 4, 8):
                if im.width / scale >= min_width:
                    im.draft("RGB", (im.width // scale, im.height // scale))
                    break
        return np.asarray(im.convert("RGB"))
