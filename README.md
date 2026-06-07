# CML Temperature Network Code Supplement

This repository contains reproducible analysis code for the CML
temperature-network study. It includes the data-preparation, modelling,
validation, plotting, and diagnostic scripts needed to reproduce the computational
workflow from private or anonymized input data.

The repository does not include raw CML telemetry, exact link locations, IP
addresses, station names, or real technology names.

## Expected Input

Point `configs/config.ini` at a private or anonymized monthly 10-minute dataset.
The expected analysis columns are:

`time_utc, cml_id, technology, temp_unit, t_ref, azimuth, altitude, sun, day, hour`

Optional columns such as `trsl`, `rsl`, and `tsl` are used automatically when present.
For privacy-safe publication, `cml_id`, `technology`, roof identifiers, and station
identifiers should be anonymized before data are placed in or near this repository.
Coordinates are needed only for the optional roof-selection step and should not be
committed.

## Quick Start

1. Copy `configs/config.ini.dist` to `configs/config.ini`.
2. Edit paths, year, and model settings for your local private/anonymized data.
3. Run all steps:

```powershell
.\run_pipeline.ps1 -Config configs/config.ini
```

Selected steps:

```powershell
.\run_pipeline.ps1 -Config configs/config.ini -Steps select,dataset,models,plots
```

## Pipeline Steps

- `select_roofs.py` selects up to five roof clusters from a candidate metadata CSV.
- `make_ml_dataset.py` filters one year of exported monthly data and adds derived features.
- `run_models.py` evaluates constant offset, clock-only ridge, linear SVR, histogram gradient boosting, and XGBoost. Additional diagnostic model variants can remain configured internally when needed. A spline-ridge experiment is available in the model runner but disabled by default because it is less stable in blocked spatial transfer.
- `make_plots.py` writes vector figures to `plots/` and compact CSV summaries to `outputs/`.
- `add_per_endpoint_baseline.py` backfills the per-endpoint constant-offset baseline into existing metrics and prediction CSVs without re-training the supervised models.
- `feature_importance.py` computes held-out permutation importance for the selected boosted-tree model.
- `fit_coldstarts.py` fits first-order warm-up curves from local cold-start files. The data files are not distributed.
- `safe_geography_summary.py` writes a privacy-safe roof/station summary from local metadata.

The default strategy is conservative: use one year, five roofs, no random shuffled split,
and save all numbers by validation protocol. Kernel SVR is trained on a fixed,
stratified subset per fold and evaluated on the full held-out block; increase
`max_rbf_train_samples_per_fold` only after the baseline run is stable.

The model runner includes two offset baselines. `constant_offset` estimates one global median offset
from the training fold. `per_endpoint_offset` estimates a separate median offset for each CML
endpoint and is saved only for temporal protocols; it is undefined for leave-one-roof validation
because all endpoints on the held-out roof are unseen during training.

The default ML run has two experiment scopes:

- `all_technologies`: all selected CML technologies pooled, with technology used as a categorical predictor.
- `single_technology_<name>`: the largest technology group by default (`single_technology = auto`) to separate hardware heterogeneity from validation effects.

Generated figures are PDF vector graphics only and are written to `plots/`.
Generated tabular summaries are CSV files written to `outputs/`.

## Privacy Boundary

The following files are intentionally ignored:

- `configs/config.ini`
- `data/private/`
- generated CSV/parquet/pickle outputs
- logs and caches
