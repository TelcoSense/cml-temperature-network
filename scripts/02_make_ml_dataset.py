#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd

from common import cfg_path, cyclic, ensure_parent, iter_csv_files, load_config, month_key


def optional_path(cfg, section: str, key: str, default: str) -> str:
    if cfg.has_section(section) and cfg.has_option(section, key):
        return cfg[section].get(key, default).strip()
    return default


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


def apply_post_reset_filter(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict[str, object]]:
    if not cfg.has_section("quality_control"):
        return df, {
            "mode": "unspecified",
            "status": "not_configured",
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_removed": 0,
            "reset_events": 0,
            "note": "No [quality_control] section was configured.",
        }

    mode = cfg["quality_control"].get("post_reset_filter_mode", "prefiltered").strip().lower()
    exclude_minutes = cfg.getfloat("quality_control", "post_reset_exclude_minutes", fallback=20.0)
    uptime_col = cfg["quality_control"].get("uptime_col", "uptime").strip()
    group_col = cfg["quality_control"].get("uptime_group_col", "cml_id").strip()
    time_col = cfg["quality_control"].get("time_col", "time_utc").strip()
    uptime_scale = cfg.getfloat("quality_control", "uptime_scale_to_seconds", fallback=1.0)
    sample_minutes = cfg.getfloat("quality_control", "sample_interval_minutes", fallback=10.0)

    before = len(df)
    base = {
        "mode": mode,
        "rows_before": before,
        "rows_after": before,
        "rows_removed": 0,
        "reset_events": 0,
        "exclude_minutes": exclude_minutes,
        "uptime_col": uptime_col,
        "group_col": group_col,
    }
    if mode in {"off", "none", "false", "0"}:
        return df, {**base, "status": "disabled", "note": "Post-reset filtering disabled."}
    if uptime_col not in df.columns:
        if mode in {"prefiltered", "auto"}:
            return df, {
                **base,
                "status": "assumed_prefiltered",
                "note": (
                    f"Column {uptime_col!r} is absent. Input data are assumed to have "
                    "been filtered upstream before export."
                ),
            }
        raise RuntimeError(
            f"Configured post-reset filtering needs column {uptime_col!r}, but it is absent. "
            "Set post_reset_filter_mode=prefiltered only if the exported input was already filtered."
        )
    if group_col not in df.columns:
        raise RuntimeError(f"Configured post-reset filtering needs group column {group_col!r}.")
    if time_col not in df.columns:
        raise RuntimeError(f"Configured post-reset filtering needs time column {time_col!r}.")

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    window_rows = max(0, int(np.ceil(exclude_minutes / max(sample_minutes, 1e-9))))
    threshold_sec = exclude_minutes * 60.0
    keep_parts = []
    reset_events = 0
    for _, g in work.sort_values([group_col, time_col]).groupby(group_col, sort=False):
        g = g.reset_index()
        u_sec = pd.to_numeric(g[uptime_col], errors="coerce") * uptime_scale
        reset = (u_sec.diff() < 0) | (u_sec <= threshold_sec)
        reset = reset.fillna(False).to_numpy(dtype=bool)
        reset_events += int(reset.sum())
        invalid = np.zeros(len(g), dtype=bool)
        for i0 in np.where(reset)[0]:
            invalid[i0 : min(len(g), i0 + window_rows + 1)] = True
        keep_parts.append(g.loc[~invalid, "index"])

    if keep_parts:
        keep_idx = pd.concat(keep_parts).to_numpy()
        out = work.loc[keep_idx].copy()
    else:
        out = work.iloc[0:0].copy()
    after = len(out)
    return out, {
        **base,
        "status": "applied",
        "rows_after": after,
        "rows_removed": before - after,
        "reset_events": reset_events,
        "note": "Post-reset/warm-up rows excluded from the ML dataset.",
    }


def write_qc_summary(path: str, summary: dict[str, object]) -> None:
    ensure_parent(path)
    lines = [f"{key}: {value}" for key, value in summary.items()]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


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
    qc_summary_txt = optional_path(cfg, "paths", "qc_summary_txt", "outputs/ml_dataset_qc_summary.txt")

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
    df, reset_qc = apply_post_reset_filter(df, cfg)

    counts = df.groupby("cml_id")["time_utc"].size()
    good_cml = counts[counts >= min_samples].index
    df = df[df["cml_id"].isin(good_cml)].copy()

    df = add_features(df, cfg)
    df["offset_raw"] = df[unit_col] - df[target_col]
    df = df.sort_values(["time_utc", "roof_id", "cml_id"])

    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)
    write_qc_summary(
        qc_summary_txt,
        {
            "rows_loaded_after_selection": before,
            "rows_final": len(df),
            "rows_dropped_cleaning_subsetting": before - len(df),
            **{f"post_reset_{k}": v for k, v in reset_qc.items()},
        },
    )

    print(f"Wrote {out_csv}: {len(df)} rows, {df['cml_id'].nunique()} CMLs, {df['roof_id'].nunique()} roofs")
    print(f"Dropped rows during cleaning/subsetting: {before - len(df)}")
    print(f"Post-reset QC: {reset_qc['status']} ({reset_qc['note']})")
    print(f"Months: {', '.join(sorted(df['month'].unique()))}")


if __name__ == "__main__":
    main()
