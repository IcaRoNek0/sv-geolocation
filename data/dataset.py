"""torch Dataset：从分片读取样本并按划分过滤。

薄封装——真正的准备工作在 data/prepare.py（纯 numpy，可单独测试）。这里
只负责按 key 取图、解码、交给 prepare、转成张量。

分片索引在所有 worker 间共享（它只在父进程建一次，扫 tar 头要几秒）。
共享是安全的，因为 ShardIndex.read 用 os.pread 在偏移量处读、不移动文件
位置——若用 seek+read，fork 出的 worker 会共享文件偏移量，互相读出对方的
字节，数据静默损坏而训练照常跑。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from data.prepare import ViewConfig, make_sample, sample_rng
from data.shards import decode_jpeg, open_index


class PanoramaDataset(Dataset):
    """按划分过滤的分片数据集。"""

    def __init__(self, data_dir, split_assignments, samples, class_index,
                 coord_index, city_index, prov_index, split="train",
                 view_cfg=None, augment=True, seed=0, eval_mode="panorama", task="environment"):
        self.data_dir = data_dir
        self.split = split
        self.class_index = class_index
        self.coord_index = coord_index      # panoid -> (x, y) 归一化坐标
        self.city_index = city_index
        self.prov_index = prov_index
        self.view_cfg = view_cfg or ViewConfig()
        self.augment = augment
        self.seed = seed
        self.eval_mode = eval_mode
        self.task = task
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        self.keys = [k for k, v in split_assignments.items()
                     if v == split and k in samples]
        self.keys.sort()
        if not self.keys:
            raise ValueError(f"划分 {split} 下没有任何样本")

        self._idx = None
        self._meta = samples

    def _index(self):
        """延迟构建：DataLoader worker 里各自建各自的句柄。

        open_index 按布局自动选读取器（tar 分片或解包后的散文件目录）。
        """
        if self._idx is None:
            self._idx = open_index(self.data_dir)
        return self._idx

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        rng = sample_rng(self.seed, key, int(self._epoch) if self.augment else 0)
        blob = self._index().read(key)
        # 视图是环绕切的，每个 90° 视图占源宽的 1/4。源宽达到 4×输出尺寸
        # 就够了，再大只是白白让采样多搬数据。四川的 2048 宽会走半尺寸
        # 解码，全国铺底的 1024 宽本来就刚好，不会被降。
        pano = decode_jpeg(blob, min_width=4 * self.view_cfg.size)
        if self.task == "vehicle":
            from data.vehicle import car_view
            views = car_view(pano, self.view_cfg, rng, self.augment)
            vmask = np.ones(1, dtype=bool)
        else:
            views, vmask = make_sample(pano, rng, self.view_cfg, augment_on=self.augment,
                                       eval_mode=self.eval_mode)

        return {
            "views": torch.from_numpy(views).permute(0, 3, 1, 2).contiguous(),
            "vmask": torch.from_numpy(vmask),
            "county": torch.tensor(self.class_index.get(key, -1), dtype=torch.long),
            "city": torch.tensor(self.city_index.get(key, -1), dtype=torch.long),
            "prov": torch.tensor(self.prov_index.get(key, -1), dtype=torch.long),
            "coord": torch.tensor(self.coord_index.get(key, (0.0, 0.0)),
                                  dtype=torch.float32),
        }


def make_loader(dataset, batch_size, shuffle, num_workers=2, seed=0,
                sampler=None, distributed=False):
    """sampler 非空时用它分片（DDP），此时不能再传 shuffle。

    分片模式下每个进程只看到自己那一份，且必须 drop_last——各进程的批数
    不一致会让集合通信互相等待到死锁。
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=shuffle or distributed,
        persistent_workers=num_workers > 0 and (shuffle or distributed),
        pin_memory=torch.cuda.is_available(),
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
        generator=torch.Generator().manual_seed(seed) if (shuffle and sampler is None) else None,
    )
