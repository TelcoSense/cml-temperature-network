#!/usr/bin/env python3
import argparse

import pandas as pd

from common import UnionFind, cfg_path, ensure_parent, haversine_m, load_config


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cml_id" not in out.columns:
        if {"link_id", "side"}.issubset(out.columns):
            out["cml_id"] = out["link_id"].astype(str) + "_" + out["side"].astype(str)
        elif "ip" in out.columns:
            out["cml_id"] = out["ip"].astype(str)
        else:
            raise RuntimeError("Candidate CSV needs cml_id, link_id+side, or ip.")
    required = {"cml_id", "lon_cml", "lat_cml"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"Candidate CSV missing required columns: {sorted(missing)}")
    return out.dropna(subset=["cml_id", "lon_cml", "lat_cml"]).copy()


def assign_roofs(df: pd.DataFrame, roof_radius_m: float) -> pd.DataFrame:
    if "roof_id" in df.columns:
        out = df.copy()
        out["roof_id"] = out["roof_id"].astype(str)
        return out

    sites = (
        df[["lon_cml", "lat_cml"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename_axis("site_idx")
        .reset_index()
    )
    uf = UnionFind(len(sites))
    lats = sites["lat_cml"].to_numpy()
    lons = sites["lon_cml"].to_numpy()
    for i in range(len(sites)):
        dist = haversine_m(lats[i], lons[i], lats[i + 1 :], lons[i + 1 :])
        for j_rel in dist.nonzero()[0]:
            if dist[j_rel] <= roof_radius_m:
                uf.union(i, i + 1 + int(j_rel))

    roots = [uf.find(i) for i in range(len(sites))]
    remap = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    sites["roof_id"] = [remap[root] for root in roots]
    out = df.merge(sites, on=["lon_cml", "lat_cml"], how="left")
    out["roof_id"] = out["roof_id"].astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    candidates_csv = cfg_path(cfg, "paths", "candidates_csv")
    selected_csv = cfg_path(cfg, "paths", "selected_candidates_csv")
    roof_selection_csv = cfg_path(cfg, "paths", "roof_selection_csv")
    roof_radius_m = cfg.getfloat("selection", "roof_radius_m", fallback=20.0)
    max_roofs = cfg.getint("selection", "max_roofs", fallback=5)
    prefer_many = cfg.getboolean("selection", "prefer_roofs_with_many_cmls", fallback=True)

    cand = normalize_candidates(pd.read_csv(candidates_csv))
    cand = assign_roofs(cand, roof_radius_m=roof_radius_m)

    roof_stats = (
        cand.groupby("roof_id", as_index=False)
        .agg(
            n_cml=("cml_id", "nunique"),
            roof_lon=("lon_cml", "median"),
            roof_lat=("lat_cml", "median"),
            mean_station_distance_m=("distance_m", "mean") if "distance_m" in cand.columns else ("lat_cml", "size"),
        )
        .copy()
    )
    sort_cols = ["n_cml", "roof_id"] if prefer_many else ["roof_id"]
    ascending = [False, True] if prefer_many else [True]
    roof_stats = roof_stats.sort_values(sort_cols, ascending=ascending)
    selected_roofs = roof_stats.head(max_roofs)["roof_id"].astype(str).tolist()

    selected = cand[cand["roof_id"].astype(str).isin(selected_roofs)].copy()
    selected = selected.sort_values(["roof_id", "cml_id"])
    roof_stats["selected"] = roof_stats["roof_id"].astype(str).isin(selected_roofs)

    ensure_parent(selected_csv)
    ensure_parent(roof_selection_csv)
    selected.to_csv(selected_csv, index=False)
    roof_stats.to_csv(roof_selection_csv, index=False)

    print(f"Wrote {selected_csv}: {selected['cml_id'].nunique()} CML endpoints on {len(selected_roofs)} roofs")
    print(f"Wrote {roof_selection_csv}")


if __name__ == "__main__":
    main()
