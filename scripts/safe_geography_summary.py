#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import haversine_m, ensure_parent


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-csv", default="outputs/selected_candidates.csv")
    ap.add_argument("--roof-csv", default="outputs/roof_selection.csv")
    ap.add_argument("--out-md", default="outputs/privacy_safe_geography_summary.md")
    args = ap.parse_args()

    selected = pd.read_csv(args.selected_csv)
    roofs = pd.read_csv(args.roof_csv)
    roofs = roofs[roofs.get("selected", True).astype(bool)].copy() if "selected" in roofs.columns else roofs.copy()

    if "roof_id" not in selected.columns or "roof_id" not in roofs.columns:
        raise RuntimeError("Expected roof_id in selected and roof summary CSVs.")

    cml_counts = selected.groupby("roof_id")["cml_id"].nunique() if "cml_id" in selected.columns else None
    tech_counts = selected.groupby("roof_id")["technology"].nunique() if "technology" in selected.columns else None
    station_col = first_existing(selected.columns, ["reference_station_id", "station_id", "ws_id", "station"])
    station_count = selected[station_col].nunique() if station_col else None

    dist_col = first_existing(roofs.columns, ["mean_station_distance_m", "distance_m", "station_distance_m"])
    lat_col = first_existing(roofs.columns, ["roof_lat", "lat_cml", "lat"])
    lon_col = first_existing(roofs.columns, ["roof_lon", "lon_cml", "lon"])

    span_km = np.nan
    if lat_col and lon_col and len(roofs) >= 2:
        coords = roofs[[lat_col, lon_col]].dropna().to_numpy(dtype=float)
        max_dist = 0.0
        for i in range(len(coords)):
            d = haversine_m(coords[i, 0], coords[i, 1], coords[i + 1 :, 0], coords[i + 1 :, 1])
            if len(d):
                max_dist = max(max_dist, float(np.max(d)))
        span_km = max_dist / 1000.0

    safe = roofs[["roof_id"]].copy()
    safe["roof_label"] = [f"R{i + 1}" for i in range(len(safe))]
    if cml_counts is not None:
        safe["n_endpoints"] = safe["roof_id"].map(cml_counts).fillna(0).astype(int)
    if station_col:
        roof_station = selected.groupby("roof_id")[station_col].agg(lambda x: sorted(set(map(str, x)))[0])
        station_aliases = {station: f"S{i + 1}" for i, station in enumerate(sorted(set(roof_station)))}
        safe["reference_station"] = safe["roof_id"].map(roof_station).map(station_aliases)
    if dist_col:
        safe["distance_to_reference_m"] = pd.to_numeric(roofs[dist_col], errors="coerce").round(0).astype("Int64")

    md = []
    md.append("# Privacy-safe geography summary")
    md.append("")
    md.append(f"- Selected roof clusters: {len(safe)}")
    if cml_counts is not None:
        md.append(f"- Selected CML endpoints: {int(cml_counts.sum())}")
    if station_count is not None:
        md.append(f"- Distinct reference stations in metadata: {station_count}")
    else:
        md.append("- Distinct reference stations in metadata: not available in selected CSV")
    if dist_col:
        d = safe["distance_to_reference_m"].dropna().astype(float)
        md.append(f"- Reference-distance range: {int(d.min())}-{int(d.max())} m")
    if not np.isnan(span_km):
        md.append(f"- Maximum separation among selected roof centroids: {span_km:.1f} km")
    md.append("")
    md.append("No coordinates, IP addresses, physical place names, or real technology names are written by this summary.")
    md.append("")
    headers = list(safe.columns)
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in safe.iterrows():
        vals = ["" if pd.isna(row[h]) else str(row[h]) for h in headers]
        md.append("| " + " | ".join(vals) + " |")

    ensure_parent(args.out_md)
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
