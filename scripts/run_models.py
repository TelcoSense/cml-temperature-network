#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge

from common import cfg_float_list, cfg_list, cfg_path, ensure_parent, load_config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


@dataclass
class Split:
    protocol: str
    fold: str
    train_idx: np.ndarray
    test_idx: np.ndarray


class ConstantOffsetRegressor:
    def fit(self, x: pd.DataFrame, y: pd.Series):
        unit = pd.to_numeric(x["temp_unit"], errors="coerce")
        target = pd.to_numeric(y, errors="coerce")
        self.offset_ = float(np.nanmedian(unit - target))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(x["temp_unit"], errors="coerce").to_numpy() - self.offset_

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


class PerEndpointOffsetRegressor:
    def fit(self, x: pd.DataFrame, y: pd.Series):
        unit = pd.to_numeric(x["temp_unit"], errors="coerce")
        target = pd.to_numeric(y, errors="coerce")
        delta = unit - target
        endpoint = x["cml_id"].astype(str)
        self.offset_by_endpoint_ = delta.groupby(endpoint).median()
        self.global_offset_ = float(np.nanmedian(delta))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        unit = pd.to_numeric(x["temp_unit"], errors="coerce").to_numpy()
        offset = (
            x["cml_id"]
            .astype(str)
            .map(self.offset_by_endpoint_)
            .fillna(self.global_offset_)
            .to_numpy()
        )
        return unit - offset

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


def make_splits(df: pd.DataFrame, protocols: List[str], min_train: int, min_test: int) -> List[Split]:
    splits: List[Split] = []
    months = sorted(df["month"].dropna().unique())
    for protocol in protocols:
        if protocol == "last_month":
            if len(months) < 2:
                continue
            test_month = months[-1]
            train_idx = df.index[df["month"] != test_month].to_numpy()
            test_idx = df.index[df["month"] == test_month].to_numpy()
            splits.append(Split(protocol, f"test_{test_month}", train_idx, test_idx))
        elif protocol == "leave_one_month":
            for m in months:
                train_idx = df.index[df["month"] != m].to_numpy()
                test_idx = df.index[df["month"] == m].to_numpy()
                splits.append(Split(protocol, f"test_{m}", train_idx, test_idx))
        elif protocol == "leave_one_roof":
            for roof in sorted(df["roof_id"].astype(str).unique()):
                train_idx = df.index[df["roof_id"].astype(str) != roof].to_numpy()
                test_idx = df.index[df["roof_id"].astype(str) == roof].to_numpy()
                splits.append(Split(protocol, f"test_roof_{roof}", train_idx, test_idx))
        else:
            raise RuntimeError(f"Unknown validation protocol: {protocol}")

    return [
        s
        for s in splits
        if len(s.train_idx) >= min_train and len(s.test_idx) >= min_test
    ]


def choose_features(df: pd.DataFrame, cfg) -> Tuple[List[str], List[str]]:
    numeric = ["temp_unit"]
    if cfg.getboolean("features", "include_clock_features", fallback=True):
        numeric += [c for c in ["doy_sin", "doy_cos", "hour_sin", "hour_cos", "sun"] if c in df.columns]
    if cfg.getboolean("features", "include_geometry_features", fallback=True):
        numeric += [c for c in ["azimuth", "altitude"] if c in df.columns]
    if cfg.getboolean("features", "include_power_features", fallback=True):
        numeric += [c for c in ["trsl", "rsl", "tsl"] if c in df.columns]

    categorical = []
    if cfg.getboolean("features", "include_categorical_ids", fallback=True):
        categorical += [c for c in ["technology", "cml_id", "roof_id"] if c in df.columns]
    return numeric, categorical


def preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
    )


def spline_preprocessor(numeric: List[str], categorical: List[str], n_knots: int, degree: int) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num_spline",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("spline", SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
    )


def build_models(cfg, numeric: List[str], categorical: List[str]) -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    prep = preprocessor(numeric, categorical)
    c_values = cfg_float_list(cfg, "models", "svr_c_values")
    gamma_values = cfg_list(cfg, "models", "svr_gamma_values")
    random_state = cfg.getint("models", "random_state", fallback=42)
    spline_knots = cfg.getint("models", "spline_n_knots", fallback=5)
    spline_degree = cfg.getint("models", "spline_degree", fallback=3)

    models: Dict[str, Tuple[object, Dict[str, List[object]]]] = {
        "constant_offset": (ConstantOffsetRegressor(), {}),
        "per_endpoint_offset": (PerEndpointOffsetRegressor(), {}),
        "clock_only_ridge": (
            Pipeline(
                [
                    ("prep", preprocessor([c for c in ["doy_sin", "doy_cos", "hour_sin", "hour_cos", "sun"] if c in numeric], [])),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            {},
        ),
        "svr_linear": (
            Pipeline(
                [
                    ("prep", prep),
                    (
                        "model",
                        LinearSVR(
                            random_state=random_state,
                            dual="auto",
                            max_iter=5000,
                        ),
                    ),
                ]
            ),
            {"model__C": c_values},
        ),
        "svr_rbf": (
            Pipeline([("prep", prep), ("model", SVR(kernel="rbf"))]),
            {"model__C": c_values, "model__gamma": gamma_values},
        ),
    }
    if cfg.getboolean("models", "run_spline_ridge", fallback=True):
        models["spline_ridge"] = (
            Pipeline(
                [
                    ("prep", spline_preprocessor(numeric, categorical, n_knots=spline_knots, degree=spline_degree)),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            {},
        )
    if cfg.getboolean("models", "run_hist_gradient_boosting", fallback=True):
        models["hist_gradient_boosting"] = (
            Pipeline(
                [
                    ("prep", prep),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=cfg.getint("models", "hgb_max_iter", fallback=250),
                            learning_rate=cfg.getfloat("models", "hgb_learning_rate", fallback=0.05),
                            max_leaf_nodes=cfg.getint("models", "hgb_max_leaf_nodes", fallback=15),
                            min_samples_leaf=cfg.getint("models", "hgb_min_samples_leaf", fallback=50),
                            l2_regularization=cfg.getfloat("models", "hgb_l2_regularization", fallback=0.1),
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {},
        )
    if cfg.getboolean("models", "run_xgboost", fallback=False):
        if XGBRegressor is None:
            raise RuntimeError(
                "run_xgboost=true but xgboost is not installed in the active Python environment."
            )
        models["xgboost"] = (
            Pipeline(
                [
                    ("prep", prep),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=cfg.getint("models", "xgb_n_estimators", fallback=350),
                            learning_rate=cfg.getfloat("models", "xgb_learning_rate", fallback=0.04),
                            max_depth=cfg.getint("models", "xgb_max_depth", fallback=4),
                            min_child_weight=cfg.getfloat("models", "xgb_min_child_weight", fallback=10.0),
                            subsample=cfg.getfloat("models", "xgb_subsample", fallback=0.85),
                            colsample_bytree=cfg.getfloat("models", "xgb_colsample_bytree", fallback=0.85),
                            reg_lambda=cfg.getfloat("models", "xgb_reg_lambda", fallback=5.0),
                            reg_alpha=cfg.getfloat("models", "xgb_reg_alpha", fallback=0.1),
                            objective="reg:squarederror",
                            tree_method=cfg["models"].get("xgb_tree_method", "hist"),
                            random_state=random_state,
                            n_jobs=cfg.getint("models", "xgb_n_jobs", fallback=1),
                        ),
                    ),
                ]
            ),
            {},
        )
    return models


def grouped_cv_for_train(train_df: pd.DataFrame):
    return grouped_cv_for_train_with_n(train_df, n_splits=3)


def grouped_cv_for_train_with_n(train_df: pd.DataFrame, n_splits: int):
    groups = train_df["month"].astype(str)
    n_groups = groups.nunique()
    if n_groups >= 2:
        return GroupKFold(n_splits=min(max(2, n_splits), n_groups)), groups
    return max(2, n_splits), None


def fit_model(
    name: str,
    estimator,
    grid: Dict[str, List[object]],
    x_train,
    y_train,
    train_df,
    inner_cv_splits: int,
):
    if not grid:
        return estimator.fit(x_train, y_train), {}

    cv, groups = grouped_cv_for_train_with_n(train_df, n_splits=inner_cv_splits)
    search = GridSearchCV(
        estimator,
        grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=None,
        refit=True,
    )
    if groups is None:
        search.fit(x_train, y_train)
    else:
        search.fit(x_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_


def sample_training_frame(
    train: pd.DataFrame, max_samples: int, random_state: int
) -> pd.DataFrame:
    if max_samples <= 0 or len(train) <= max_samples:
        return train

    work = train.copy()
    work["sample_group"] = work["month"].astype(str) + "|roof_" + work["roof_id"].astype(str)
    n_groups = work["sample_group"].nunique()
    per_group = max(1, int(np.ceil(max_samples / max(n_groups, 1))))
    sampled_parts = []
    for _, group in work.groupby("sample_group", sort=False):
        sampled_parts.append(
            group.sample(
                n=min(len(group), per_group),
                random_state=random_state,
            )
        )
    sampled = pd.concat(sampled_parts, axis=0).drop(columns=["sample_group"])
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=random_state)
    return sampled.sort_values("time_utc")


def score(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae_c": float(mean_absolute_error(y_true, y_pred)),
        "rmse_c": float(rmse),
        "bias_c": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def experiment_frames(df: pd.DataFrame, cfg) -> List[Tuple[str, pd.DataFrame]]:
    frames: List[Tuple[str, pd.DataFrame]] = []
    run_all = cfg.getboolean("experiments", "run_all_technologies", fallback=True)
    run_single = cfg.getboolean("experiments", "run_single_technology", fallback=False)

    if run_all:
        frames.append(("all_technologies", df.copy()))

    if run_single:
        requested = cfg["experiments"].get("single_technology", "auto").strip()
        if requested.lower() == "auto":
            technology = str(df["technology"].value_counts().index[0])
        else:
            technology = requested
        one = df[df["technology"].astype(str) == technology].copy()
        if one.empty:
            raise RuntimeError(f"No samples found for single_technology={technology!r}")
        frames.append((f"single_technology_{technology}", one))

    if not frames:
        raise RuntimeError("No experiment scopes enabled in [experiments].")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg_path(cfg, "paths", "ml_dataset_csv"))
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    target_col = cfg["dataset"].get("target_col", "t_ref")
    protocols = cfg_list(cfg, "validation", "protocols")
    min_train = cfg.getint("validation", "min_train_samples", fallback=1000)
    min_test = cfg.getint("validation", "min_test_samples", fallback=100)
    max_fit = cfg.getint("models", "max_train_samples_per_fold", fallback=30000)
    max_rbf_fit = cfg.getint("models", "max_rbf_train_samples_per_fold", fallback=1500)
    max_xgb_fit = cfg.getint("models", "max_xgb_train_samples_per_fold", fallback=max_fit)
    inner_cv_splits = cfg.getint("models", "inner_cv_splits", fallback=2)
    random_state = cfg.getint("models", "random_state", fallback=42)
    metric_rows = []
    pred_rows = []

    for scope, scope_df in experiment_frames(df, cfg):
        scope_df = scope_df.reset_index(drop=True).copy()
        scope_df["row_id"] = np.arange(len(scope_df))
        numeric, categorical = choose_features(scope_df, cfg)
        features = sorted(set(numeric + categorical))
        splits = make_splits(scope_df, protocols, min_train=min_train, min_test=min_test)
        if not splits:
            print(f"SKIP {scope}: no validation splits possible with configured limits", flush=True)
            continue
        models = build_models(cfg, numeric, categorical)

        tech_summary = (
            scope_df.groupby("technology")["cml_id"]
            .agg(rows="size", cmls="nunique")
            .reset_index()
            .to_dict(orient="records")
        )
        print(f"SCOPE {scope}: rows={len(scope_df)}, technologies={tech_summary}", flush=True)

        for split in splits:
            train = scope_df.loc[split.train_idx].copy()
            test = scope_df.loc[split.test_idx].copy()
            x_test = test[features]
            y_test = test[target_col]

            for name, (estimator, grid) in models.items():
                if name == "per_endpoint_offset" and split.protocol == "leave_one_roof":
                    continue
                print(
                    f"START {scope}/{split.protocol}/{split.fold}/{name} "
                    f"(train={len(train)}, test={len(test)})",
                    flush=True,
                )
                fit_train = train
                if name not in {"constant_offset", "clock_only_ridge"}:
                    if name == "svr_rbf":
                        model_max_fit = max_rbf_fit
                    elif name == "xgboost":
                        model_max_fit = max_xgb_fit
                    else:
                        model_max_fit = max_fit
                    fit_train = sample_training_frame(
                        train, max_samples=model_max_fit, random_state=random_state
                    )
                x_train = fit_train[features]
                y_train = fit_train[target_col]
                fitted, params = fit_model(
                    name,
                    estimator,
                    grid,
                    x_train,
                    y_train,
                    fit_train,
                    inner_cv_splits=inner_cv_splits,
                )
                pred = fitted.predict(x_test)
                metrics = score(y_test, pred)
                metric_rows.append(
                    {
                        "scope": scope,
                        "protocol": split.protocol,
                        "fold": split.fold,
                        "model": name,
                        "n_train": len(train),
                        "n_fit": len(fit_train),
                        "n_test": len(test),
                        "n_train_months": train["month"].nunique(),
                        "n_test_months": test["month"].nunique(),
                        "n_train_roofs": train["roof_id"].nunique(),
                        "n_test_roofs": test["roof_id"].nunique(),
                        "n_train_technologies": train["technology"].nunique(),
                        "n_test_technologies": test["technology"].nunique(),
                        "best_params_json": json.dumps(params, sort_keys=True),
                        **metrics,
                    }
                )
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "scope": scope,
                            "protocol": split.protocol,
                            "fold": split.fold,
                            "model": name,
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
                print(f"DONE {scope}/{split.protocol}/{split.fold}/{name}: MAE={metrics['mae_c']:.3f} C", flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True)
    metrics_csv = cfg_path(cfg, "paths", "metrics_csv")
    preds_csv = cfg_path(cfg, "paths", "predictions_csv")
    ensure_parent(metrics_csv)
    ensure_parent(preds_csv)
    metrics_df.to_csv(metrics_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {preds_csv}")


if __name__ == "__main__":
    main()
