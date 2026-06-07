#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import cfg_path, ensure_parent, load_config


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


MODEL_LABELS = {
    "constant_offset": "Constant offset",
    "per_endpoint_offset": "Per-endpoint offset",
    "clock_only_ridge": "Clock-only ridge",
    "svr_linear": "Linear SVM",
    "hist_gradient_boosting": "HistGBT",
    "xgboost": "XGBoost",
}

REPORTED_MODELS = [
    "constant_offset",
    "per_endpoint_offset",
    "clock_only_ridge",
    "svr_linear",
    "hist_gradient_boosting",
    "xgboost",
]
SCATTER_MODELS = ["constant_offset", "svr_linear", "xgboost"]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " "))


def technology_aliases_from_frame(df: pd.DataFrame) -> dict[str, str]:
    if "technology" not in df.columns:
        return {}
    counts = df["technology"].astype(str).value_counts()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {tech: f"Technology {letters[i]}" for i, tech in enumerate(counts.index)}


def scope_label(scope: str, aliases: dict[str, str]) -> str:
    if scope == "all_technologies":
        return "All technologies"
    prefix = "single_technology_"
    if scope.startswith(prefix):
        raw = scope[len(prefix) :]
        return f"{aliases.get(raw, 'Single technology')} only"
    return scope.replace("_", " ")


def primary_scope(df: pd.DataFrame) -> str:
    if "scope" not in df.columns:
        return ""
    scopes = set(df["scope"].dropna().astype(str))
    if "all_technologies" in scopes:
        return "all_technologies"
    return sorted(scopes)[0]


def with_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if not scope or "scope" not in df.columns:
        return df.copy()
    return df[df["scope"].astype(str) == scope].copy()


def preferred_last_month_fold(preds: pd.DataFrame) -> tuple[str, str]:
    last = preds[preds["protocol"] == "last_month"]
    if not last.empty:
        return "last_month", sorted(last["fold"].dropna().unique())[-1]
    lom = preds[preds["protocol"] == "leave_one_month"]
    if lom.empty:
        raise RuntimeError("No last_month or leave_one_month predictions found.")
    return "leave_one_month", sorted(lom["fold"].dropna().unique())[-1]


def sample_predictions(
    preds: pd.DataFrame, protocol: str, fold: str, model: str, n: int, seed: int
) -> pd.DataFrame:
    part = preds[
        (preds["protocol"] == protocol)
        & (preds["fold"] == fold)
        & (preds["model"] == model)
    ].copy()
    if n > 0 and len(part) > n:
        part = part.sample(n=n, random_state=seed)
    return part


def common_prediction_sample(
    preds: pd.DataFrame, protocol: str, fold: str, models: list[str], n: int, seed: int
) -> pd.DataFrame:
    part = preds[
        (preds["protocol"] == protocol)
        & (preds["fold"] == fold)
        & (preds["model"].isin(models))
    ].copy()
    if part.empty or "row_id" not in part.columns:
        return part

    counts = part.groupby("row_id")["model"].nunique()
    common_ids = counts[counts == len(models)].index.to_series()
    if n > 0 and len(common_ids) > n:
        common_ids = common_ids.sample(n=n, random_state=seed)
    return part[part["row_id"].isin(common_ids)]


def make_scatter_figure(
    preds: pd.DataFrame, out_pdf: str, aliases: dict[str, str], seed: int = 42
) -> None:
    scope = primary_scope(preds)
    preds = with_scope(preds, scope)
    protocol, fold = preferred_last_month_fold(preds)
    models = SCATTER_MODELS
    available = [
        model
        for model in models
        if (
            (preds["protocol"] == protocol)
            & (preds["fold"] == fold)
            & (preds["model"] == model)
        ).any()
    ]

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(4.2 * len(available), 3.7),
        sharex=True,
        sharey=True,
    )
    if len(available) == 1:
        axes = [axes]

    all_part = preds[(preds["protocol"] == protocol) & (preds["fold"] == fold)]
    values = np.r_[all_part["y_true"].to_numpy(), all_part["y_pred"].to_numpy()]
    vmin = float(np.nanpercentile(values, 0.5)) - 0.5
    vmax = float(np.nanpercentile(values, 99.5)) + 0.5

    sampled = common_prediction_sample(preds, protocol, fold, available, n=25000, seed=seed)
    hb = None
    for ax, model in zip(axes, available):
        part = sampled[sampled["model"] == model]
        hb = ax.hexbin(
            part["y_true"],
            part["y_pred"],
            gridsize=55,
            mincnt=1,
            cmap="viridis",
            linewidths=0.0,
        )
        ax.plot([vmin, vmax], [vmin, vmax], color="crimson", linestyle="--", linewidth=1.0)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.88", linewidth=0.5)
        ax.set_xlabel("Reference temperature ($^{\\circ}$C)")

    axes[0].set_ylabel("Estimated temperature ($^{\\circ}$C)")
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.17, top=0.95, wspace=0.12)
    if hb is not None:
        cax = fig.add_axes([0.915, 0.17, 0.014, 0.78])
        fig.colorbar(hb, cax=cax, label="Samples per bin")

    ensure_parent(out_pdf)
    fig.savefig(out_pdf)
    plt.close(fig)


def make_technology_scatter_figure(
    preds: pd.DataFrame, out_pdf: str, aliases: dict[str, str], seed: int = 42
) -> None:
    scope = primary_scope(preds)
    preds = with_scope(preds, scope)
    protocol, fold = preferred_last_month_fold(preds)
    part = sample_predictions(preds, protocol, fold, "svr_linear", n=0, seed=seed)
    if part.empty or "technology" not in part.columns:
        return

    technologies = sorted(part["technology"].astype(str).unique(), key=lambda tech: aliases.get(tech, tech))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]
    color_by_technology = {tech: palette[i % len(palette)] for i, tech in enumerate(technologies)}
    values = np.r_[part["y_true"].to_numpy(), part["y_pred"].to_numpy()]
    vmin = float(np.nanpercentile(values, 0.5)) - 0.5
    vmax = float(np.nanpercentile(values, 99.5)) + 0.5

    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    for tech in technologies:
        g = part[part["technology"].astype(str) == tech]
        ax.scatter(
            g["y_true"],
            g["y_pred"],
            s=4,
            alpha=0.18,
            linewidths=0,
            color=color_by_technology[tech],
            rasterized=True,
            zorder=2,
        )
    ax.plot([vmin, vmax], [vmin, vmax], color="black", linestyle="--", linewidth=1.0, zorder=3)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference temperature ($^{\\circ}$C)")
    ax.set_ylabel("Estimated temperature ($^{\\circ}$C)")
    ax.grid(True, color="0.88", linewidth=0.5)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color_by_technology[tech],
            label=aliases.get(tech, "Technology"),
            markersize=5,
        )
        for tech in technologies
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    ensure_parent(out_pdf)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=600)
    plt.close(fig)


def make_mae_figure(metrics: pd.DataFrame, out_pdf: str) -> None:
    scope = primary_scope(metrics)
    metrics = with_scope(metrics, scope)
    order = REPORTED_MODELS
    protocols = [
        protocol
        for protocol in ["last_month", "leave_one_month", "leave_one_roof"]
        if protocol in set(metrics["protocol"])
    ]
    summary = (
        metrics.groupby(["protocol", "model"], as_index=False)
        .agg(mae=("mae_c", "mean"), mae_std=("mae_c", "std"))
    )

    fig, axes = plt.subplots(1, len(protocols), figsize=(4.2 * len(protocols), 3.4), sharey=True)
    if len(protocols) == 1:
        axes = [axes]
    colors = {
        "constant_offset": "#777777",
        "per_endpoint_offset": "#66A61E",
        "clock_only_ridge": "#4C78A8",
        "svr_linear": "#F58518",
        "hist_gradient_boosting": "#E45756",
        "xgboost": "#B279A2",
    }
    for ax, protocol in zip(axes, protocols):
        part = summary[summary["protocol"] == protocol].copy()
        part = part[part["model"].isin(order)]
        part["model"] = pd.Categorical(part["model"], categories=order, ordered=True)
        part = part.sort_values("model")
        x = np.arange(len(part))
        ax.bar(
            x,
            part["mae"],
            yerr=part["mae_std"].fillna(0.0),
            color=[colors.get(str(model), "0.5") for model in part["model"]],
            capsize=3,
            linewidth=0.5,
            edgecolor="0.25",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([model_label(str(model)) for model in part["model"]], rotation=35, ha="right")
        ax.grid(True, axis="y", color="0.88", linewidth=0.5)
    axes[0].set_ylabel("MAE ($^{\\circ}$C)")
    fig.tight_layout()
    ensure_parent(out_pdf)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_scope_comparison_figure(
    metrics: pd.DataFrame, out_pdf: str, aliases: dict[str, str]
) -> None:
    if "scope" not in metrics.columns or metrics["scope"].nunique() < 2:
        return
    protocol = "leave_one_month"
    part = metrics[metrics["protocol"] == protocol].copy()
    if part.empty:
        return
    summary = (
        part.groupby(["scope", "model"], as_index=False)
        .agg(mae=("mae_c", "mean"), mae_std=("mae_c", "std"))
    )
    order = REPORTED_MODELS
    scopes = sorted(summary["scope"].unique(), key=lambda s: (s != "all_technologies", s))
    width = 0.35 if len(scopes) == 2 else 0.25
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, scope in enumerate(scopes):
        rows = summary[summary["scope"] == scope].set_index("model")
        vals = [rows.loc[model, "mae"] if model in rows.index else np.nan for model in order]
        errs = [rows.loc[model, "mae_std"] if model in rows.index else 0.0 for model in order]
        offset = (i - (len(scopes) - 1) / 2.0) * width
        label = scope_label(scope, aliases)
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(model) for model in order], rotation=30, ha="right")
    ax.set_ylabel("MAE ($^{\\circ}$C)")
    ax.grid(True, axis="y", color="0.88", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_parent(out_pdf)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_timeseries_figure(preds: pd.DataFrame, out_pdf: str) -> None:
    scope = primary_scope(preds)
    preds = with_scope(preds, scope)
    protocol, fold = preferred_last_month_fold(preds)
    candidates = preds[(preds["protocol"] == protocol) & (preds["fold"] == fold)]
    if candidates.empty:
        return
    cml_id = candidates["cml_id"].value_counts().index[0]
    part = candidates[candidates["cml_id"] == cml_id].copy()
    part["time_utc"] = pd.to_datetime(part["time_utc"], utc=True, errors="coerce")
    t0 = part["time_utc"].min()
    t1 = t0 + pd.Timedelta(days=7)
    part = part[(part["time_utc"] >= t0) & (part["time_utc"] < t1)]
    pivot = part.pivot_table(index="time_utc", columns="model", values="y_pred", aggfunc="first")
    truth = part.groupby("time_utc")["y_true"].first()

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(truth.index, truth.values, color="black", linewidth=1.3, label="Reference")
    for model, color in [
        ("constant_offset", "0.55"),
        ("clock_only_ridge", "#4C78A8"),
        ("svr_linear", "#F58518"),
        ("hist_gradient_boosting", "#E45756"),
        ("xgboost", "#B279A2"),
    ]:
        if model in pivot.columns:
            ax.plot(pivot.index, pivot[model], linewidth=1.0, label=model_label(model), color=color)
    ax.set_ylabel("Temperature ($^{\\circ}$C)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, color="0.88", linewidth=0.5)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    ensure_parent(out_pdf)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary_csvs(
    metrics: pd.DataFrame,
    out_summary: str,
    out_months: str,
    out_scope: str,
    aliases: dict[str, str],
) -> None:
    primary = with_scope(metrics, primary_scope(metrics))
    summary = (
        primary.groupby(["protocol", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae=("mae_c", "mean"),
            rmse=("rmse_c", "mean"),
            bias=("bias_c", "mean"),
            r2=("r2", "mean"),
        )
    )
    summary = summary[summary["model"].isin(REPORTED_MODELS)].copy()
    ensure_parent(out_summary)
    summary["model_label"] = summary["model"].map(model_label)
    summary.to_csv(out_summary, index=False)

    month = primary[primary["protocol"] == "leave_one_month"].copy()
    month = month[month["model"].isin(REPORTED_MODELS)]
    month["month"] = month["fold"].str.replace("test_", "", regex=False)
    pivot = month.pivot_table(index="month", columns="model", values="mae_c", aggfunc="first")
    ensure_parent(out_months)
    pivot.sort_index().reset_index().to_csv(out_months, index=False)

    if "scope" in metrics.columns:
        scope_summary = (
            metrics[metrics["protocol"] == "leave_one_month"]
            .groupby(["scope", "model"], as_index=False)
            .agg(folds=("fold", "nunique"), mae=("mae_c", "mean"), rmse=("rmse_c", "mean"), r2=("r2", "mean"))
        )
        scope_summary = scope_summary[scope_summary["model"].isin(REPORTED_MODELS)].copy()
        scope_summary["scope_label"] = scope_summary["scope"].map(lambda s: scope_label(str(s), aliases))
        scope_summary["model_label"] = scope_summary["model"].map(model_label)
        ensure_parent(out_scope)
        scope_summary.to_csv(out_scope, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = pd.read_csv(cfg_path(cfg, "paths", "metrics_csv"))
    preds = pd.read_csv(cfg_path(cfg, "paths", "predictions_csv"))
    aliases = {}
    if "scope" in preds.columns:
        aliases = technology_aliases_from_frame(preds[preds["scope"] == "all_technologies"])
    if not aliases:
        aliases = technology_aliases_from_frame(preds)

    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_scatter_figure(preds, str(plot_dir / "ml_blocked_scatter_three_panel.pdf"), aliases)
    make_technology_scatter_figure(preds, str(plot_dir / "ml_technology_scatter.pdf"), aliases)
    make_mae_figure(metrics, str(plot_dir / "ml_validation_mae.pdf"))
    make_scope_comparison_figure(metrics, str(plot_dir / "ml_scope_comparison.pdf"), aliases)
    make_timeseries_figure(preds, str(plot_dir / "ml_last_month_timeseries.pdf"))
    write_summary_csvs(
        metrics,
        "outputs/ml_summary_by_protocol.csv",
        "outputs/ml_leave_one_month_mae.csv",
        "outputs/ml_scope_summary.csv",
        aliases,
    )

    print("Wrote ML vector figures to", plot_dir)
    print("Wrote ML summary CSV files to outputs/")


if __name__ == "__main__":
    main()
