#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from common import cfg_list, cfg_path, ensure_parent, load_config


RUN_MODELS_PATH = Path(__file__).with_name("03_run_models.py")
spec = importlib.util.spec_from_file_location("run_models", RUN_MODELS_PATH)
run_models = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = run_models
spec.loader.exec_module(run_models)


MODEL_LABELS = {
    "hist_gradient_boosting": "HistGBT",
    "xgboost": "XGBoost",
}


def choose_split(splits, protocol: str):
    candidates = [s for s in splits if s.protocol == protocol]
    if not candidates:
        raise RuntimeError(f"No split found for protocol={protocol!r}")
    return sorted(candidates, key=lambda s: s.fold)[-1]


def feature_label(name: str) -> str:
    labels = {
        "temp_unit": "$t_{\\mathrm{MU}}$",
        "doy_sin": "seasonal sine",
        "doy_cos": "seasonal cosine",
        "hour_sin": "diurnal sine",
        "hour_cos": "diurnal cosine",
        "sun": "daylight",
        "azimuth": "azimuth",
        "altitude": "altitude",
        "trsl": "transmit level",
        "rsl": "receive level",
        "tsl": "transmit power",
        "technology": "technology",
        "cml_id": "endpoint ID",
        "roof_id": "roof ID",
    }
    return labels.get(name, name.replace("_", " "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    ap.add_argument("--scope", default="all_technologies")
    ap.add_argument("--protocol", default="last_month")
    ap.add_argument("--model", default="hist_gradient_boosting", choices=["hist_gradient_boosting", "xgboost"])
    ap.add_argument("--n-repeats", type=int, default=8)
    ap.add_argument("--max-test-samples", type=int, default=50000)
    ap.add_argument("--out-csv", default="outputs/ml_feature_importance.csv")
    ap.add_argument("--out-pdf", default="plots/ml_feature_importance.pdf")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg_path(cfg, "paths", "ml_dataset_csv"))
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    target_col = cfg["dataset"].get("target_col", "t_ref")
    protocols = cfg_list(cfg, "validation", "protocols")
    if args.protocol not in protocols:
        protocols = protocols + [args.protocol]

    scopes = dict(run_models.experiment_frames(df, cfg))
    if args.scope not in scopes:
        raise RuntimeError(f"Scope {args.scope!r} not available. Available: {sorted(scopes)}")

    scope_df = scopes[args.scope].reset_index(drop=True).copy()
    scope_df["row_id"] = np.arange(len(scope_df))
    numeric, categorical = run_models.choose_features(scope_df, cfg)
    features = sorted(set(numeric + categorical))
    splits = run_models.make_splits(
        scope_df,
        protocols,
        min_train=cfg.getint("validation", "min_train_samples", fallback=1000),
        min_test=cfg.getint("validation", "min_test_samples", fallback=100),
    )
    split = choose_split(splits, args.protocol)

    train = scope_df.loc[split.train_idx].copy()
    test = scope_df.loc[split.test_idx].copy()
    if args.max_test_samples > 0 and len(test) > args.max_test_samples:
        test = test.sample(n=args.max_test_samples, random_state=cfg.getint("models", "random_state", fallback=42))

    models = run_models.build_models(cfg, numeric, categorical)
    estimator, grid = models[args.model]
    max_fit = cfg.getint("models", "max_train_samples_per_fold", fallback=30000)
    if args.model == "xgboost":
        max_fit = cfg.getint("models", "max_xgb_train_samples_per_fold", fallback=max_fit)
    fit_train = run_models.sample_training_frame(
        train,
        max_samples=max_fit,
        random_state=cfg.getint("models", "random_state", fallback=42),
    )

    fitted, params = run_models.fit_model(
        args.model,
        estimator,
        grid,
        fit_train[features],
        fit_train[target_col],
        fit_train,
        cfg.getint("models", "inner_cv_splits", fallback=2),
    )
    result = permutation_importance(
        fitted,
        test[features],
        test[target_col],
        n_repeats=args.n_repeats,
        scoring="neg_mean_absolute_error",
        random_state=cfg.getint("models", "random_state", fallback=42),
        n_jobs=1,
    )
    out = pd.DataFrame(
        {
            "scope": args.scope,
            "protocol": split.protocol,
            "fold": split.fold,
            "model": args.model,
            "feature": features,
            "label": [feature_label(f) for f in features],
            "importance_mae_c": result.importances_mean,
            "importance_std_c": result.importances_std,
            "n_fit": len(fit_train),
            "n_test": len(test),
            "best_params_json": str(params),
        }
    ).sort_values("importance_mae_c", ascending=False)

    ensure_parent(args.out_csv)
    out.to_csv(args.out_csv, index=False)

    top = out.head(10).sort_values("importance_mae_c")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.barh(top["label"], top["importance_mae_c"], xerr=top["importance_std_c"], color="#4C78A8")
    ax.set_xlabel("Increase in MAE after permutation ($^{\\circ}$C)")
    ax.set_ylabel("")
    ax.grid(True, axis="x", color="0.88", linewidth=0.5)
    fig.tight_layout()
    ensure_parent(args.out_pdf)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
