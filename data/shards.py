"""WebDataset 分片的随机读取索引。

只依赖标准库，可以在没有 torch 的环境里单独测试——索引偏移和 tar 头解析
正是最容易出错、又最需要测试的地方。

不做解包：两万多个小文件解到磁盘上既慢又占空间，而在 Colab 上从 Drive
拷贝分片本身是顺序读，反而快。所以启动时扫一遍 tar 头建立
`panoid → (分片, 偏移, 长度)` 索引，之后按偏移 seek 读取。

索引本身很小（每条三个整数），建立耗时在秒级。
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
        """扫一遍 tar 头建立偏移索引。

        want_json 默认关闭：侧车 JSON 训练用不到（元数据来自
        samples.jsonl），读了也是白读。实测在 21 746 条 / 9 个分片上
        只省 0.1 秒（5.2s → 5.1s，约 2%）——侧车文件很小且是顺序读，
        开销被 tar 头遍历盖过去了。所以这是个**顺手的清理，不是优化**；
        真需要 meta() 时打开即可。
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

        用 os.pread 而非 seek+read：DataLoader 的 worker 是 fork 出来的，
        **文件偏移量属于 open file description，是跨进程共享的**。用
        seek+read 时两个 worker 会互相把对方的文件位置挪走，读出的字节
        是别人的——数据静默损坏，训练照常跑，指标却毫无意义。
        pread 带偏移量读取、不动文件位置，从根上避开这个问题。

        长度校验同样是承重的：tarfile 对**截断的归档不报错**（实测截到
        40% 仍照常列出全部成员名），索引会因此指向文件末尾之外。
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


def decode_jpeg(blob):
    """JPEG 字节 → (H, W, 3) uint8。PIL 缺失时抛 ImportError。"""
    from PIL import Image
    import numpy as np
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(im.convert("RGB"))
