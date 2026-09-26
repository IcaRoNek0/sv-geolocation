"""torch Dataset：从分片读取样本并按划分过滤。

薄封装——真正的准备工作在 data/prepare.py（纯 numpy，可单独测试）。这里
只负责按 key 取图、解码、交给 prepare、转成张量。

分片索引在所有 worker 间共享（它只在父进程建一次，扫 tar 头要几秒）。
共享是安全的，因为 ShardIndex.read 用 os.pread 在偏移量处读、不移动文件
位置——若用 seek+read，fork 出的 worker 会共享文件偏移量，互相读出对方的
字节，数据静默损坏而训练照常跑。
"""
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from data.prepare import ViewConfig, make_sample
from data.shards import ShardIndex, decode_jpeg


class PanoramaDataset(Dataset):
    """按划分过滤的分片数据集。"""

    def __init__(self, shard_paths, split_assignments, samples, class_index,
                 coord_index, city_index, prov_index, split="train",
                 view_cfg=None, augment=True, seed=0):
        self.shard_paths = [str(p) for p in sorted(shard_paths)]
        self.split = split
        self.class_index = class_index
        self.coord_index = coord_index      # panoid -> (x, y) 归一化坐标
        self.city_index = city_index
        self.prov_index = prov_index
        self.view_cfg = view_cfg or ViewConfig()
        self.augment = augment
        self.seed = seed

        self.keys = [k for k, v in split_assignments.items()
                     if v == split and k in samples]
        self.keys.sort()
        if not self.keys:
            raise ValueError(f"划分 {split} 下没有任何样本")

        self._idx = None
        self._meta = samples

    def _index(self):
        """延迟构建：DataLoader worker 里各自建各自的句柄。"""
        if self._idx is None:
            self._idx = ShardIndex(self.shard_paths)
        return self._idx

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        wid = get_worker_info()
        # 每个 worker 用不同的随机流，否则同一 epoch 内不同 worker 的增强会同步。
        #
        # 不用内置 hash()：Python 的字符串哈希每个进程都加盐，同一个 --seed
        # 两次运行会得到不同的增强，指标之间的差异就分不清是模型还是噪声。
        # crc32 是确定的。
        rng = np.random.default_rng(
            (self.seed, i, wid.id if wid else 0,
             zlib.crc32(self.split.encode()) & 0xFFFF)
        )
        key = self.keys[i]
        blob = self._index().read(key)
        pano = decode_jpeg(blob)
        views, vmask = make_sample(pano, rng, self.view_cfg, augment_on=self.augment)

        return {
            "views": torch.from_numpy(views).permute(0, 3, 1, 2).contiguous(),
            "vmask": torch.from_numpy(vmask),
            "county": torch.tensor(self.class_index.get(key, -1), dtype=torch.long),
            "city": torch.tensor(self.city_index.get(key, -1), dtype=torch.long),
            "prov": torch.tensor(self.prov_index.get(key, -1), dtype=torch.long),
            "coord": torch.tensor(self.coord_index.get(key, (0.0, 0.0)),
                                  dtype=torch.float32),
        }


def make_loader(dataset, batch_size, shuffle, num_workers=2, seed=0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )
