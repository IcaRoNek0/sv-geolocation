"""划分逻辑的测试。

最关键的一条：**(车辆, 日期) 分组不得跨越训练与评估**。同一辆车同一天拍下
的相邻帧几乎重复，一旦被分到两边，验证集里就出现了训练集见过的地点，指标
虚高却不报错。

用合成数据跑真实的 make_splits.py，而不是复制一份划分逻辑到测试里——
复制出来的逻辑会各自演化，测试就失去意义。
"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = AI_ROOT / "collect" / "make_splits.py"


def synth_samples(n_counties=30, per_county=60, seed=1):
    """造出含跨县车辆分组的合成样本。"""
    rng = random.Random(seed)
    rows = []
    # 故意让一部分分组横跨两个县：真实街景车一天会开过县界
    for ci in range(n_counties):
        adcode = f"51{ci:04d}"
        for i in range(per_county):
            # 每 10 条共用一个跨县分组
            if i % 10 == 0:
                vehicle = f"{ci:03d}_0000_A1"
            else:
                vehicle = f"{ci:03d}_{i:04d}_B2"
            rows.append({
                "panoid": f"{ci:04d}{i:05d}",
                "adcode": adcode,
                "part": "sichuan",
                "lng": 104.0 + ci * 0.01,
                "lat": 30.0 + i * 0.001,
                "vehicle": vehicle,
                "date": "140725",
                "city": f"{ci:03d}",
            })
    # 补一批全国铺底样本
    for i in range(400):
        rows.append({
            "panoid": f"9{i:06d}",
            "adcode": f"{10 + i % 30:02d}0000",
            "part": "national",
            "lng": 110.0 + i * 0.01,
            "lat": 32.0 + i * 0.01,
            "vehicle": f"9{i:03d}_0000_N9",
            "date": "150101",
            "city": "900",
        })
    return rows


class TestSplitIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.samples_path = cls.tmp / "samples.jsonl"
        rows = synth_samples()
        with cls.samples_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        cls.out = cls.tmp / "split.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--samples", str(cls.samples_path),
             "--out", str(cls.out),
             "--holdout-counties", "5",
             "--per-county", "12"],
            capture_output=True, text=True,
        )
        cls.proc = proc
        cls.payload = json.loads(cls.out.read_text()) if cls.out.exists() else None
        cls.rows = [json.loads(l) for l in cls.samples_path.read_text(encoding="utf-8").splitlines()]

    def test_script_succeeds(self):
        self.assertEqual(self.proc.returncode, 0,
                         f"脚本失败：{self.proc.stdout}\n{self.proc.stderr}")

    def test_no_group_spans_train_and_eval(self):
        """核心不变式：任何 (车辆, 日期) 分组不得横跨训练与评估。"""
        self.assertIsNotNone(self.payload)
        rows = self.rows
        assignments = self.payload["assignments"]

        by_group = defaultdict(set)
        for r in rows:
            g = f"{r['vehicle']}|{r['date']}"
            by_group[g].add(assignments[r["panoid"]])

        bad = {g: k for g, k in by_group.items()
               if "train" in k and len(k) > 1}
        self.assertEqual(bad, {}, f"这些分组跨越了训练与评估：{list(bad)[:5]}")

    def test_every_sample_assigned(self):
        rows = self.rows
        a = self.payload["assignments"]
        self.assertEqual(len(a), len(rows))
        for r in rows:
            self.assertIn(a[r["panoid"]],
                          {"train", "val_same", "val_national", "test_county"})

    def test_holdout_counties_fully_removed_from_train(self):
        """整县留出的县，训练集里不得有任何该县样本。"""
        rows = self.rows
        holdout = set(self.payload["holdout_adcodes"])
        a = self.payload["assignments"]
        self.assertTrue(holdout)
        for r in rows:
            if r["adcode"] in holdout:
                self.assertNotEqual(a[r["panoid"]], "train",
                                    f"{r['panoid']} 属于留出县却进了训练集")

    def test_per_county_val_budget_respected(self):
        """同县留出的量应接近 --per-county，允许因整组进出而超出。"""
        rows = self.rows
        a = self.payload["assignments"]
        holdout = set(self.payload["holdout_adcodes"])
        per = defaultdict(int)
        for r in rows:
            if a[r["panoid"]] == "val_same":
                per[r["adcode"]] += 1
        for c, n in per.items():
            self.assertLessEqual(n, 12 + 10,
                                 f"{c} 留出了 {n} 条，超出预算过多")

    def test_selection_is_deterministic(self):
        """同样输入必须给出同样划分——否则不同 session 的指标不可比。"""
        out2 = self.tmp / "split2.json"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--samples", str(self.samples_path),
             "--out", str(out2),
             "--holdout-counties", "5", "--per-county", "12"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(self.out.read_text(), out2.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
