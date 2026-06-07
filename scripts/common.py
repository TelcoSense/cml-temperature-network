#!/usr/bin/env python3
import configparser
import math
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cfg.read(path, encoding="utf-8")
    return cfg


def cfg_path(cfg: configparser.ConfigParser, section: str, key: str) -> str:
    return cfg[section][key].strip()


def cfg_list(cfg: configparser.ConfigParser, section: str, key: str) -> List[str]:
    raw = cfg[section].get(key, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def cfg_float_list(cfg: configparser.ConfigParser, section: str, key: str) -> List[float]:
    return [float(x) for x in cfg_list(cfg, section, key)]


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(a))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def month_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.strftime("%Y-%m")


def cyclic(series: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    angle = 2.0 * math.pi * pd.to_numeric(series, errors="coerce") / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(angle),
            f"{prefix}_cos": np.cos(angle),
        },
        index=series.index,
    )


def iter_csv_files(input_dir: str, pattern: str) -> Iterable[str]:
    return sorted(str(p) for p in Path(input_dir).glob(pattern))
