#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import cfg_path, ensure_parent, load_config

REPORTED_MODELS = ["constant_offset", "clock_only_ridge", "svr_linear", "hist_gradient_boosting", "xgboost"]


def fmt_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    return df.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def primary_scope(metrics: pd.DataFrame) -> str:
    if "scope" not in metrics.columns:
        return ""
    scopes = set(metrics["scope"].dropna().astype(str))
    if "all_technologies" in scopes:
        return "all_technologies"
    return sorted(scopes)[0]


def technology_aliases(dataset: pd.DataFrame) -> dict[str, str]:
    counts = dataset["technology"].astype(str).value_counts()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {tech: f"Technology {letters[i]}" for i, tech in enumerate(counts.index)}


def scope_label(scope: str, aliases: dict[str, str]) -> str:
    if scope == "all_technologies":
        return "all technologies"
    prefix = "single_technology_"
    if scope.startswith(prefix):
        raw = scope[len(prefix) :]
        return f"{aliases.get(raw, 'single technology')} only"
    return scope.replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = pd.read_csv(cfg_path(cfg, "paths", "metrics_csv"))
    dataset = pd.read_csv(cfg_path(cfg, "paths", "ml_dataset_csv"))
    report_path = cfg_path(cfg, "paths", "report_txt")

    aliases = technology_aliases(dataset)
    scope = primary_scope(metrics)
    primary_metrics = metrics if not scope else metrics[metrics["scope"].astype(str) == scope].copy()
    reported_metrics = metrics[metrics["model"].isin(REPORTED_MODELS)].copy()
    reported_primary_metrics = primary_metrics[primary_metrics["model"].isin(REPORTED_MODELS)].copy()
    scope_summary = pd.DataFrame()
    if "scope" in metrics.columns:
        scope_summary = (
            reported_metrics[reported_metrics["protocol"] == "leave_one_month"]
            .groupby(["scope", "model"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                mae_mean_c=("mae_c", "mean"),
                rmse_mean_c=("rmse_c", "mean"),
                r2_mean=("r2", "mean"),
            )
            .sort_values(["scope", "mae_mean_c", "model"])
        )
        scope_summary["scope"] = scope_summary["scope"].astype(str).map(
            lambda s: scope_label(s, aliases)
        )

    summary = (
        reported_primary_metrics.groupby(["protocol", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae_mean_c=("mae_c", "mean"),
            mae_std_c=("mae_c", "std"),
            rmse_mean_c=("rmse_c", "mean"),
            bias_mean_c=("bias_c", "mean"),
            r2_mean=("r2", "mean"),
        )
        .sort_values(["protocol", "mae_mean_c", "model"])
    )
    offset = dataset["offset_raw"].describe(percentiles=[0.05, 0.5, 0.95])
    tech_summary = (
        dataset.groupby("technology")
        .agg(rows=("technology", "size"), cmls=("cml_id", "nunique"), roofs=("roof_id", "nunique"))
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    tech_summary["technology"] = tech_summary["technology"].astype(str).map(
        lambda t: aliases.get(t, "Technology")
    )

    text = f"""ML revision report
==================

Purpose
-------
This experiment rebuilds the manuscript ML section around reviewer-robust claims:
no random shuffled split, blocked temporal and roof-based validation, and a
compact reported model set consisting of a constant-offset correction, a
clock-only ridge baseline, linear SVM, histogram gradient-boosted trees, and
XGBoost.

Dataset
-------
Rows: {len(dataset)}
CML endpoints: {dataset['cml_id'].nunique()}
Roofs: {dataset['roof_id'].nunique()}
Months: {', '.join(sorted(dataset['month'].astype(str).unique()))}

Technology coverage:
{fmt_table(tech_summary)}

Raw unit-reference offset, temp_unit - t_ref:
{offset.to_string(float_format=lambda x: f"{x:.3f}")}

Validation and models
---------------------
The main estimates are based on temporal or group-blocked validation:
last-month holdout, leave-one-month-out, and leave-one-roof-out when possible.
These protocols directly address leakage caused by randomly shuffling autocorrelated
10-minute samples.

Models:
- constant_offset: t_est = temp_unit - median(temp_unit - t_ref) fitted on training data only.
- clock_only_ridge: seasonal and diurnal features only; checks whether the model is just learning climatology.
- svr_linear: scaled linear support-vector regression.
- hist_gradient_boosting: regularized histogram gradient-boosted trees for tabular nonlinear effects.
- xgboost: regularized gradient-boosted trees using the XGBoost implementation.

The main reported model set is intentionally limited to two baselines and three
learned models.

Metric summary
--------------
Primary scope: {scope_label(scope, aliases) if scope else 'not specified'}

{fmt_table(summary)}

Technology-scope comparison (leave-one-month)
---------------------------------------------
{fmt_table(scope_summary)}

Fold-level metrics for primary scope
------------------------------------
{fmt_table(reported_primary_metrics.sort_values(['protocol', 'fold', 'mae_c', 'model']))}

Recommended manuscript wording
------------------------------
We estimate agreement with the remote reference station, not the true local
roof-level air temperature. The reference station is spatially separated from
the CML units, so MAE values include both model error and site mismatch.

Avoid saying that any model achieves local air-temperature accuracy unless
an independent co-located reference is added. A safer claim is:

"Using one year of 10-minute observations from five roof clusters, CML unit
temperature explained most of the variance in the reference-station air
temperature. Under blocked temporal validation, the linear SVM gave the best
temporal generalization, the constant-offset correction quantified the
uncorrected hardware bias, the clock-only baseline quantified seasonal and
diurnal skill, and boosted-tree models were most relevant for the separate
roof-transfer test."

Reviewer-risk checklist
-----------------------
- Keep the reported ML table focused on constant offset, clock-only ridge, linear SVM, HistGBT, and XGBoost.
- Put the validation protocol in the table caption: no randomly shuffled 80/20 split.
- Show results by fold/month/roof so the best number is not presented alone.
- Keep mRMR out unless the score definition and scale are explained; use model tables and
  permutation/ablation evidence instead if feature importance is needed.
- Phrase the study as feasibility at one site/five roofs unless more geographically
  distributed roofs are added.
"""

    ensure_parent(report_path)
    Path(report_path).write_text(text, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
