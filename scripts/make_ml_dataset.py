#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd

from common import cfg_path, cyclic, ensure_parent, iter_csv_files, load_config, month_key


def load_raw_dataset(input_dir: str, pattern: str) -> pd.DataFrame:
    files = list(iter_csv_files(input_dir, pattern))
    if not files:
        raise RuntimeError(f"No raw dataset files found in {input_dir} matching {pattern}")
    parts = []
    for fp in files:
        part = pd.read_csv(fp)
        part["source_file"] = fp
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def add_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df.copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["time_utc"]).copy()
    out["month"] = month_key(out["time_utc"])
    out["doy"] = out["time_utc"].dt.dayofyear.astype(int)
    out["hour_float"] = out["time_utc"].dt.hour + out["time_utc"].dt.minute / 60.0

    if cfg.getboolean("features", "include_clock_features", fallback=True):
        out = pd.concat(
            [
                out,
                cyclic(out["doy"], 366.0, "doy"),
                cyclic(out["hour_float"], 24.0, "hour"),
            ],
            axis=1,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    selected_csv = cfg_path(cfg, "paths", "selected_candidates_csv")
    raw_dir = cfg_path(cfg, "paths", "raw_dataset_dir")
    raw_pattern = cfg["paths"].get("raw_dataset_pattern", "merged_10m_*.csv")
    out_csv = cfg_path(cfg, "paths", "ml_dataset_csv")

    start = pd.Timestamp(cfg["dataset"]["start_utc"])
    stop = pd.Timestamp(cfg["dataset"]["stop_utc"])
    target_col = cfg["dataset"].get("target_col", "t_ref")
    unit_col = cfg["dataset"].get("unit_temp_col", "temp_unit")
    max_abs_temp = cfg.getfloat("dataset", "max_abs_temp_c", fallback=80.0)
    min_samples = cfg.getint("selection", "min_samples_per_cml", fallback=1000)

    selected = pd.read_csv(selected_csv)
    keep_cml = set(selected["cml_id"].astype(str))
    roof_map = selected[["cml_id", "roof_id"]].drop_duplicates()
    roof_map["cml_id"] = roof_map["cml_id"].astype(str)
    roof_map["roof_id"] = roof_map["roof_id"].astype(str)

    df = load_raw_dataset(raw_dir, raw_pattern)
    if "cml_id" not in df.columns:
        raise RuntimeError("Raw dataset needs cml_id.")
    df["cml_id"] = df["cml_id"].astype(str)
    df = df[df["cml_id"].isin(keep_cml)].copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df[(df["time_utc"] >= start) & (df["time_utc"] < stop)].copy()
    df = df.merge(roof_map, on="cml_id", how="left")

    required = ["time_utc", "cml_id", "roof_id", unit_col, target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"ML dataset missing required columns after merge: {missing}")

    for col in [unit_col, target_col, "azimuth", "altitude", "trsl", "rsl", "tsl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    if cfg.getboolean("dataset", "drop_missing", fallback=True):
        df = df.dropna(subset=[unit_col, target_col, "roof_id"]).copy()
    df = df[df[unit_col].abs() <= max_abs_temp].copy()
    df = df[df[target_col].abs() <= max_abs_temp].copy()

    counts = df.groupby("cml_id")["time_utc"].size()
    good_cml = counts[counts >= min_samples].index
    df = df[df["cml_id"].isin(good_cml)].copy()

    df = add_features(df, cfg)
    df["offset_raw"] = df[unit_col] - df[target_col]
    df = df.sort_values(["time_utc", "roof_id", "cml_id"])

    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}: {len(df)} rows, {df['cml_id'].nunique()} CMLs, {df['roof_id'].nunique()} roofs")
    print(f"Dropped rows during cleaning/subsetting: {before - len(df)}")
    print(f"Months: {', '.join(sorted(df['month'].unique()))}")


if __name__ == "__main__":
    main()
