"""分片索引器的测试。

按偏移量 seek 读取最容易出的错是**偏移算错但只错一点点**——读出来的字节数
对得上，内容却是隔壁成员的尾巴。所以断言必须是逐字节相等，而不是长度相等。
"""
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.shards import ShardIndex, load_split  # noqa: E402

# 模块级：fork 出的子进程要用同一份期望值
EXPECTED = {
    "a": bytes(range(256)) * 3,
    "bb": b"\xff" * 1000,
    "ccc": b"JPEG-ish" + bytes([i % 251 for i in range(4096)]),
    "d": b"\x01\x02\x03",
}


def _read_worker(idx, keys, ok):
    """在子进程里反复读取并逐字节比对。模块级定义，否则无法被子进程导入。"""
    try:
        for _ in range(20):
            for k in keys:
                if idx.read(k) != EXPECTED[k]:
                    return
        ok.value = 1
    except Exception:
        pass


def build_tar(path, entries, pad=0):
    """entries: [(key, jpeg_bytes, meta_dict)]。pad 用于制造成员间的对齐差异。"""
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as tf:
        for key, blob, meta in entries:
            info = tarfile.TarInfo(f"{key}.jpg")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
            if meta is not None:
                b = json.dumps(meta).encode()
                mi = tarfile.TarInfo(f"{key}.json")
                mi.size = len(b)
                tf.addfile(mi, io.BytesIO(b))
            if pad:
                pb = b"\x00" * pad
                pi = tarfile.TarInfo(f"{key}.pad")
                pi.size = len(pb)
                tf.addfile(pi, io.BytesIO(pb))


class TestShardIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.payloads = EXPECTED
        # 用两个分片，并故意插入变长成员改变后续偏移
        s0 = [(k, self.payloads[k], {"adcode": "510105", "n": len(self.payloads[k])})
              for k in ("a", "bb")]
        s1 = [(k, self.payloads[k], {"adcode": "510104", "n": len(self.payloads[k])})
              for k in ("ccc", "d")]
        build_tar(self.tmp / "sv-0000.tar", s0, pad=7)
        build_tar(self.tmp / "sv-0001.tar", s1, pad=13)
        self.idx = ShardIndex(sorted(self.tmp.glob("sv-*.tar")))

    def tearDown(self):
        self.idx.close()

    def test_finds_all_keys(self):
        self.assertEqual(sorted(self.idx.keys), ["a", "bb", "ccc", "d"])
        self.assertEqual(len(self.idx), 4)

    def test_reads_exact_bytes(self):
        """逐字节相等——长度对得上但内容错位是这里最典型的 bug。"""
        for k, want in self.payloads.items():
            self.assertEqual(self.idx.read(k), want, f"{k} 内容不符")

    def test_reads_across_shards(self):
        self.assertIn("a", self.idx)
        self.assertIn("ccc", self.idx)
        self.assertEqual(self.idx.read("ccc"), self.payloads["ccc"])

    def test_metadata_sidecar(self):
        """侧车 JSON 是选择性读取的，要用就得显式打开。"""
        idx = ShardIndex(sorted(self.tmp.glob("sv-*.tar")), want_json=True)
        self.assertEqual(idx.meta("a")["adcode"], "510105")
        self.assertEqual(idx.meta("ccc")["adcode"], "510104")
        idx.close()

    def test_json_not_read_by_default(self):
        """默认不读侧车 JSON。

        两万多个成员逐个解出来解析会占启动时间的大头，而训练用不到
        ——元数据来自 samples.jsonl。这条断言钉住这个默认值，避免以后
        有人顺手改回去又把启动拖慢。
        """
        self.assertEqual(self.idx.meta("a"), {})
        self.assertEqual(len(self.idx), 4, "索引本身仍应完整")

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            self.idx.read("nope")
        self.assertNotIn("nope", self.idx)

    def test_repeated_reads_consistent(self):
        """反复读取必须一致——句柄复用不能把文件位置搞乱。"""
        for _ in range(3):
            for k, want in self.payloads.items():
                self.assertEqual(self.idx.read(k), want)

    def test_read_does_not_move_file_offset(self):
        """read() 不得改变文件偏移量。

        这条断言直接锁定一个会静默损坏数据的缺陷：DataLoader 的 worker 是
        fork 出来的，**文件偏移量属于 open file description，跨进程共享**。
        若用 seek+read，两个 worker 会互相把对方的文件位置挪走，读出别人的
        字节——训练照常跑，指标毫无意义。pread 带偏移读取，不动位置。

        断言偏移量而非靠多进程竞态碰运气：竞态不一定复现，而偏移量是确定的。
        """
        fd = self.idx._fd(0)
        before = os.lseek(fd, 0, os.SEEK_CUR)
        for k in ("a", "bb", "a", "bb"):
            self.idx.read(k)
        after = os.lseek(fd, 0, os.SEEK_CUR)
        self.assertEqual(before, after, "read() 挪动了文件偏移量")

    def test_concurrent_workers_read_correct_bytes(self):
        """多个 fork 出的进程并发读，每个都必须拿到自己的字节。

        必须显式用 fork 启动方式：DataLoader 的 worker 就是这样来的，
        也只有 fork 会让子进程继承父进程已打开的文件描述符、从而共享
        文件偏移量。forkserver/spawn 会重新打开文件，测不出这个问题。
        """
        idx2 = ShardIndex(sorted(self.tmp.glob("sv-*.tar")))
        ctx = mp.get_context("fork")
        ok = ctx.Value("i", 0)
        procs = [ctx.Process(target=_read_worker, args=(idx2, tuple(self.payloads), ok))
                 for _ in range(3)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
        self.assertEqual(ok.value, 1, "并发读取拿到了错误的字节")
        idx2.close()

    def test_empty_glob_rejected(self):
        with self.assertRaises(ValueError):
            ShardIndex([])

    def test_truncated_shard_is_detected_loudly(self):
        """截断的分片必须被检出，而不是静默产出坏数据。

        不指定在哪一层报错：tarfile 对截断的容忍程度取决于截断落点——
        落在填充区时它能照常枚举成员名（实测截到 40% 仍列出全部成员），
        落在头部之间时它会在枚举阶段就抛错。两条路径都算检出。
        """
        p = self.tmp / "sv-0000.tar"
        blob = p.read_bytes()
        p.write_bytes(blob[: int(len(blob) * 0.4)])   # 截进成员数据内部
        detected = False
        try:
            idx = ShardIndex([p])
            for k in list(idx.keys):
                idx.read(k)
            idx.close()
        except Exception:
            detected = True
        self.assertTrue(detected, "截断的分片未被检出，会静默产出坏数据")

    def test_short_read_guard(self):
        """夸大索引长度到越界，验证读长度校验生效。

        夸大的幅度必须真正越过文件末尾：tar 尾部有零填充，只多读几十字节
        会读到填充而非越界，校验不会触发。
        """
        idx = ShardIndex(sorted(self.tmp.glob("sv-*.tar")))
        si, off, size = idx._index["a"]
        idx._index["a"] = (si, off, size + 10_000_000)
        with self.assertRaises(IOError):
            idx.read("a")
        idx.close()


class TestLoadSplit(unittest.TestCase):
    def test_roundtrip(self):
        tmp = Path(tempfile.mkdtemp()) / "split.json"
        tmp.write_text(json.dumps({
            "assignments": {"p1": "train", "p2": "test_county"},
            "holdout_adcodes": ["510105"],
        }), encoding="utf-8")
        a, payload = load_split(tmp)
        self.assertEqual(a["p1"], "train")
        self.assertEqual(payload["holdout_adcodes"], ["510105"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
