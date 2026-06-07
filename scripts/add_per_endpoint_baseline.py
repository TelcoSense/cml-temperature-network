#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import cfg_list, cfg_path, ensure_parent, load_config


RUN_MODELS_PATH = Path(__file__).with_name("run_models.py")
spec = importlib.util.spec_from_file_location("run_models", RUN_MODELS_PATH)
run_models = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = run_models
spec.loader.exec_module(run_models)

experiment_frames = run_models.experiment_frames
make_splits = run_models.make_splits
score = run_models.score


MODEL_NAME = "per_endpoint_offset"


def per_endpoint_offset_predict(train: pd.DataFrame, test: pd.DataFrame, unit_col: str, target_col: str) -> np.ndarray:
    delta = pd.to_numeric(train[unit_col], errors="coerce") - pd.to_numeric(train[target_col], errors="coerce")
    offsets = delta.groupby(train["cml_id"].astype(str)).median()
    global_offset = float(np.nanmedian(delta))
    test_offset = test["cml_id"].astype(str).map(offsets).fillna(global_offset).to_numpy()
    return pd.to_numeric(test[unit_col], errors="coerce").to_numpy() - test_offset


def drop_existing_model(path: str, model_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    if df.empty or "model" not in df.columns:
        return df
    return df[df["model"].astype(str) != model_name].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_csv = cfg_path(cfg, "paths", "ml_dataset_csv")
    metrics_csv = cfg_path(cfg, "paths", "metrics_csv")
    predictions_csv = cfg_path(cfg, "paths", "predictions_csv")

    df = pd.read_csv(dataset_csv)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    target_col = cfg["dataset"].get("target_col", "t_ref")
    unit_col = cfg["dataset"].get("unit_temp_col", "temp_unit")
    protocols = cfg_list(cfg, "validation", "protocols")
    min_train = cfg.getint("validation", "min_train_samples", fallback=1000)
    min_test = cfg.getint("validation", "min_test_samples", fallback=100)

    metric_rows = []
    pred_rows = []
    for scope, scope_df in experiment_frames(df, cfg):
        scope_df = scope_df.reset_index(drop=True).copy()
        scope_df["row_id"] = np.arange(len(scope_df))
        splits = make_splits(scope_df, protocols, min_train=min_train, min_test=min_test)
        for split in splits:
            if split.protocol == "leave_one_roof":
                continue
            train = scope_df.loc[split.train_idx].copy()
            test = scope_df.loc[split.test_idx].copy()
            pred = per_endpoint_offset_predict(train, test, unit_col=unit_col, target_col=target_col)
            y_test = test[target_col]
            metrics = score(y_test, pred)
            metric_rows.append(
                {
                    "scope": scope,
                    "protocol": split.protocol,
                    "fold": split.fold,
                    "model": MODEL_NAME,
                    "n_train": len(train),
                    "n_fit": len(train),
                    "n_test": len(test),
                    "n_train_months": train["month"].nunique(),
                    "n_test_months": test["month"].nunique(),
                    "n_train_roofs": train["roof_id"].nunique(),
                    "n_test_roofs": test["roof_id"].nunique(),
                    "n_train_technologies": train["technology"].nunique(),
                    "n_test_technologies": test["technology"].nunique(),
                    "best_params_json": "{}",
                    **metrics,
                }
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "scope": scope,
                        "protocol": split.protocol,
                        "fold": split.fold,
                        "model": MODEL_NAME,
                        "row_id": test["row_id"].to_numpy(),
                        "time_utc": test["time_utc"].astype(str).to_numpy(),
                        "cml_id": test["cml_id"].astype(str).to_numpy(),
                        "roof_id": test["roof_id"].astype(str).to_numpy(),
                        "technology": test["technology"].astype(str).to_numpy(),
                        "y_true": y_test.to_numpy(),
                        "y_pred": pred,
                    }
                )
            )

    if not metric_rows:
        raise RuntimeError("No temporal splits available for the per-endpoint baseline.")

    metrics = pd.concat([drop_existing_model(metrics_csv, MODEL_NAME), pd.DataFrame(metric_rows)], ignore_index=True)
    preds = pd.concat([drop_existing_model(predictions_csv, MODEL_NAME), pd.concat(pred_rows, ignore_index=True)], ignore_index=True)

    ensure_parent(metrics_csv)
    ensure_parent(predictions_csv)
    metrics.to_csv(metrics_csv, index=False)
    preds.to_csv(predictions_csv, index=False)
    print(f"Wrote {metrics_csv} with {len(metric_rows)} {MODEL_NAME} rows")
    print(f"Wrote {predictions_csv} with {sum(len(p) for p in pred_rows)} {MODEL_NAME} predictions")


if __name__ == "__main__":
    main()
