#!/usr/bin/env python3
import argparse

import pandas as pd

from common import cfg_path, ensure_parent, load_config


MODEL_LABELS = {
    "constant_offset": "Constant offset",
    "clock_only_ridge": "Clock-only ridge",
    "svr_linear": "Linear SVM",
    "hist_gradient_boosting": "HistGBT",
    "xgboost": "XGBoost",
}


def technology_aliases(df: pd.DataFrame) -> dict[str, str]:
    counts = df["technology"].astype(str).value_counts()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {tech: f"Technology {letters[i]}" for i, tech in enumerate(counts.index)}


def write_lines(path: str, lines: list[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    ap.add_argument("--offset-csv", default="outputs/endpoint_offset_by_technology.csv")
    ap.add_argument("--offset-tex", default="outputs/endpoint_offset_by_technology_rows.tex")
    ap.add_argument("--roof-csv", default="outputs/leave_one_roof_fold_mae.csv")
    ap.add_argument("--roof-tex", default="outputs/leave_one_roof_fold_mae_rows.tex")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg_path(cfg, "paths", "ml_dataset_csv"))
    metrics = pd.read_csv(cfg_path(cfg, "paths", "metrics_csv"))
    aliases = technology_aliases(df)

    endpoint = (
        df.assign(offset_c=lambda x: x["temp_unit"] - x["t_ref"])
        .groupby(["technology", "cml_id"], dropna=False)
        .agg(
            median_offset_c=("offset_c", "median"),
            mean_offset_c=("offset_c", "mean"),
            n_samples=("offset_c", "size"),
        )
        .reset_index()
    )
    summary = (
        endpoint.groupby("technology", dropna=False)
        .agg(
            n_endpoints=("cml_id", "nunique"),
            median_endpoint_offset_c=("median_offset_c", "median"),
            p10_endpoint_offset_c=("median_offset_c", lambda x: x.quantile(0.10)),
            p90_endpoint_offset_c=("median_offset_c", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    summary["technology_alias"] = summary["technology"].astype(str).map(aliases)
    summary = summary.sort_values("technology_alias")
    ensure_parent(args.offset_csv)
    summary[
        [
            "technology_alias",
            "n_endpoints",
            "median_endpoint_offset_c",
            "p10_endpoint_offset_c",
            "p90_endpoint_offset_c",
        ]
    ].to_csv(args.offset_csv, index=False)
    write_lines(
        args.offset_tex,
        [
            (
                f"{r.technology_alias} & {int(r.n_endpoints)} & "
                f"{r.median_endpoint_offset_c:.1f} & "
                f"{r.p10_endpoint_offset_c:.1f}--{r.p90_endpoint_offset_c:.1f} \\\\"
            )
            for r in summary.itertuples(index=False)
        ],
    )

    roof = metrics[
        (metrics["scope"] == "all_technologies")
        & (metrics["protocol"] == "leave_one_roof")
        & (metrics["model"].isin(MODEL_LABELS))
    ].copy()
    roof["model_label"] = roof["model"].map(MODEL_LABELS)
    roof_summary = (
        roof.groupby(["model", "model_label"], dropna=False)
        .agg(
            mean_mae_c=("mae_c", "mean"),
            min_mae_c=("mae_c", "min"),
            max_mae_c=("mae_c", "max"),
            sd_mae_c=("mae_c", "std"),
            n_folds=("mae_c", "size"),
        )
        .reset_index()
    )
    order = {name: i for i, name in enumerate(MODEL_LABELS)}
    roof_summary["order"] = roof_summary["model"].map(order)
    roof_summary = roof_summary.sort_values("order")
    ensure_parent(args.roof_csv)
    roof_summary[
        ["model_label", "n_folds", "mean_mae_c", "min_mae_c", "max_mae_c", "sd_mae_c"]
    ].to_csv(args.roof_csv, index=False)
    write_lines(
        args.roof_tex,
        [
            (
                f"{r.model_label} & {int(r.n_folds)} & {r.mean_mae_c:.2f} & "
                f"{r.min_mae_c:.2f}--{r.max_mae_c:.2f} & {r.sd_mae_c:.2f} \\\\"
            )
            for r in roof_summary.itertuples(index=False)
        ],
    )

    print(f"Wrote {args.offset_csv}")
    print(f"Wrote {args.offset_tex}")
    print(f"Wrote {args.roof_csv}")
    print(f"Wrote {args.roof_tex}")


if __name__ == "__main__":
    main()
