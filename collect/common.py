"""采集端共享工具。

路径约定、panoID 解析、网格抽稀、只读数据库连接。
"""
import math
import sqlite3
from pathlib import Path

# ── 路径 ────────────────────────────────────────────────────────────────
AI_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = AI_ROOT.parent
MAIN_DB = WORKSPACE / "qsresult" / "all_streetviews.sqlite3"
OUT_DIR = AI_ROOT / "data" / "pool"

# ── 采样参数 ────────────────────────────────────────────────────────────
SICHUAN_PREFIX = "51"
SICHUAN_PER_COUNTY = 109        # 183 县 × 109 ≈ 20 000
NATIONAL_PER_PROVINCE = 303     # 33 省 × 303 ≈ 10 000
SAMPLE_GRID_M = 300.0           # 训练样本抽稀：同县内 300 米内只留一点
POINTS_PER_COUNTY = 500         # 点位表上限
POINTS_GRID_M = 1000.0          # 点位表抽稀网格

METERS_PER_DEG_LAT = 111_320.0


def parse_panoid(pid):
    """27 位 panoID -> 车辆/日期/城市。

    `09006900001410191436379165M` -> 车辆 `090_0000_5M`、日期 `141019`、城市 `069`。
    字段位置与 svc_trajectory/refer/doc.py 一致。
    """
    if not pid or len(pid) < 16:
        return None
    return {
        "vehicle": f"{pid[:3]}_{pid[6:10]}_{pid[-2:]}",
        "date": pid[10:16],
        "city": pid[3:6],
    }


def province_of(adcode):
    """6 位县级 adcode -> 6 位省级 adcode（`510104` -> `510000`）。"""
    if not adcode or len(adcode) < 2:
        return None
    return adcode[:2] + "0000"


def _cell(lng, lat, size_m):
    """经纬度 -> 指定米制网格的格子编号。"""
    cos_lat = math.cos(math.radians(lat))
    if cos_lat < 1e-6:
        cos_lat = 1e-6
    gx = int(lng * METERS_PER_DEG_LAT * cos_lat / size_m)
    gy = int(lat * METERS_PER_DEG_LAT / size_m)
    return (gx, gy)


class GridThinner:
    """按网格抽稀：同一格子只接受第一个点。

    格子用单个整数编码，避免元组开销——主库三千万行时这里的内存是硬约束。
    """

    __slots__ = ("size_m", "_seen")

    def __init__(self, size_m):
        self.size_m = size_m
        self._seen = set()

    def accept(self, lng, lat):
        gx, gy = _cell(lng, lat, self.size_m)
        key = (gx << 32) ^ gy
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def __len__(self):
        return len(self._seen)


def open_main_db():
    """以只读方式打开主库。

    主库约 6.3 GB，任何写入都是事故，因此固定 mode=ro；
    调用方负责 close。
    """
    if not MAIN_DB.exists():
        raise FileNotFoundError(f"主库不存在：{MAIN_DB}")
    uri = f"file:{MAIN_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only = ON")
    return conn
