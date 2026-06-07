#!/usr/bin/env python3
import argparse
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common import ensure_parent


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings = []
    for si in root.findall("x:si", NS):
        texts = [t.text or "" for t in si.findall(".//x:t", NS)]
        strings.append("".join(texts))
    return strings


def read_first_sheet_xlsx(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        shared = read_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find("x:sheets/x:sheet", NS)
        if first_sheet is None:
            raise RuntimeError(f"No sheets in {path}")
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels:
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise RuntimeError(f"Cannot resolve first sheet in {path}")
        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))

    rows = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            idx = column_index(ref)
            value = cell.find("x:v", NS)
            if value is None:
                values[idx] = np.nan
                continue
            raw = value.text
            if cell.attrib.get("t") == "s":
                values[idx] = shared[int(raw)]
            else:
                try:
                    values[idx] = float(raw)
                except (TypeError, ValueError):
                    values[idx] = raw
        if values:
            rows.append([values.get(i, np.nan) for i in range(max(values) + 1)])
    max_len = max(len(r) for r in rows)
    rows = [r + [np.nan] * (max_len - len(r)) for r in rows]
    header = [str(x) for x in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def warmup(t_sec, t_inf, t0, tau_sec):
    return t_inf - (t_inf - t0) * np.exp(-t_sec / tau_sec)


def fit_series(name: str, temp: pd.Series, power_w: float | None):
    y = pd.to_numeric(temp, errors="coerce").dropna().to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float) * 60.0
    p0 = [float(np.nanmax(y)), float(y[0]), 300.0]
    bounds = ([-50.0, -50.0, 1.0], [150.0, 150.0, 7200.0])
    popt, pcov = curve_fit(warmup, t, y, p0=p0, bounds=bounds, maxfev=50000)
    sigma = np.sqrt(np.diag(pcov))
    pred = warmup(t, *popt)
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    t_inf, t0, tau = popt
    tau_sigma = sigma[2]
    # The laboratory units were AC-equilibrated before start-up; use fitted t0 as a proxy
    # for room temperature when no separate room-temperature logger is available.
    g_th = np.nan
    c_th = np.nan
    if power_w is not None and abs(t_inf - t0) > 1e-9:
        g_th = power_w / (t_inf - t0)
        c_th = tau * g_th
    return {
        "series": name,
        "n_samples": len(y),
        "t0_c": t0,
        "t_inf_c": t_inf,
        "tau_min": tau / 60.0,
        "tau_sigma_min": tau_sigma / 60.0,
        "r2": r2,
        "power_w": power_w if power_w is not None else np.nan,
        "g_th_w_k": g_th,
        "c_th_j_k": c_th,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="data/private/cold_start_measurement.xlsx",
        help="Path to the laboratory cold-start workbook. Not included in the public repository.",
    )
    ap.add_argument(
        "--reset-csv",
        default="data/private/field_reset.csv",
        help="Path to a field reset CSV with a temp column. Not included in the public repository.",
    )
    ap.add_argument("--out-csv", default="outputs/cold_start_fit_summary.csv")
    ap.add_argument("--out-tex", default="outputs/cold_start_fit_rows.tex")
    args = ap.parse_args()

    df = read_first_sheet_xlsx(args.excel)
    reset = pd.read_csv(args.reset_csv)
    cml_temp = pd.to_numeric(reset["temp"], errors="coerce").iloc[::2].reset_index(drop=True)
    cml_temp = cml_temp.iloc[: len(df)]

    rows = [
        fit_series("LA", df["LA unit"], power_w=20.5),
        fit_series("HA", df["HA unit"], power_w=20.8),
        fit_series("field CML", cml_temp, power_w=None),
    ]
    out = pd.DataFrame(rows)
    ensure_parent(args.out_csv)
    out.to_csv(args.out_csv, index=False)

    tex_rows = []
    for r in rows:
        g = "--" if math.isnan(r["g_th_w_k"]) else f"{r['g_th_w_k']:.2f}"
        c = "--" if math.isnan(r["c_th_j_k"]) else f"{r['c_th_j_k']:.0f}"
        tex_rows.append(
            f"{r['series']} & {r['t0_c']:.1f} & {r['t_inf_c']:.1f} & "
            f"{r['tau_min']:.1f} $\\pm$ {r['tau_sigma_min']:.1f} & "
            f"{r['r2']:.3f} & {g} & {c} \\\\"
        )
    ensure_parent(args.out_tex)
    Path(args.out_tex).write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
