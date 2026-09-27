"""行政区划名称与点位归属。

名称表直接读 trajectory 项目的 assets，不复制一份过来——那样两边会各自
演化。两个表都要用：

    divisions-2023.json    3209 条，全量
    divisions-legacy.json    51 条，2020 年后改名的区划（沙县区、龙海区、
                           会理市等）与港澳分区

实测两表零重叠，所以合并顺序无关；仍按"legacy 覆盖 2023"合并，这样将来
若出现同名条目，以更新的一方为准。
"""
import json
from pathlib import Path

ASSETS = (Path(__file__).resolve().parent.parent.parent
          / "trajectory" / "src" / "main" / "assets")
COUNTY_ROOT = Path(__file__).resolve().parent.parent.parent / "geojson" / "counties"


def load_names():
    """合并两张表，返回 {adcode: 名称}。"""
    names = {}
    for fname in ("divisions-2023.json", "divisions-legacy.json"):
        p = ASSETS / fname
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}（trajectory 项目的行政区划表）")
        names.update(json.loads(p.read_text(encoding="utf-8")))
    return names


def city_code(adcode):
    """县级 adcode → 所属地级市。

    与 trajectory 的 County.kt 保持一致：直辖市（11/12/31/50 开头）归到
    xx0000，省直辖县级单位（第 3-4 位为 90）保持自身，其余取前 4 位 + 00。
    注意直辖市这一条——按"前 4 位 + 00"会得到 110100，而表里没有这个码。
    """
    if not adcode or len(adcode) < 6:
        return adcode
    if adcode[:2] in ("11", "12", "31", "50"):
        return adcode[:2] + "0000"
    if adcode[2:4] == "90":
        return adcode
    return adcode[:4] + "00"


def province_code(adcode):
    return adcode[:2] + "0000" if adcode and len(adcode) >= 2 else adcode


def describe(adcode, names, sep=" "):
    """adcode → "四川省 德阳市 广汉市"，缺哪级跳过哪级。

    直辖市会把省市两级解析成同一个码（北京 110101 → 110000/110000），
    靠去重避免重复显示。
    """
    if not adcode:
        return "?"
    parts = []
    for code in (province_code(adcode), city_code(adcode), adcode):
        n = names.get(code)
        if n and n not in parts:
            parts.append(n)
    return sep.join(parts) if parts else adcode


class CountyLocator:
    """按坐标判定落在哪个县。只为屏幕上几个点做判定，不建空间索引。"""

    def __init__(self, root=COUNTY_ROOT):
        from shapely.geometry import shape
        self.entries = []
        for f in sorted(Path(root).glob("*_full.json")):
            if f.name == "100000_full.json":
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for feat in d.get("features", []):
                props = feat.get("properties", {})
                if props.get("level") != "district":
                    continue
                try:
                    self.entries.append((str(props.get("adcode")),
                                         shape(feat["geometry"])))
                except Exception:
                    continue

    def __len__(self):
        return len(self.entries)

    def at(self, lon, lat):
        """返回该坐标所属的县级 adcode，落在边界外则 None。"""
        from shapely.geometry import Point
        pt = Point(lon, lat)
        for code, geom in self.entries:
            if geom.contains(pt):
                return code
        return None
