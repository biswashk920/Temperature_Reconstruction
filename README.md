# Temperature Reconstruction

A machine learning pipeline for **bias-correcting ERA5-Land reanalysis temperature data using ground station observations**, built with XGBoost, and applied to reconstruct a long-term (1950–2024) station-quality temperature series.

## Overview

Reanalysis products like ERA5-Land provide global, gridded climate data, but at coarse spatial resolution (~9 km), which introduces systematic bias relative to point-based station measurements — especially in complex terrain. This project trains a machine learning model to learn and correct that bias, using paired station and ERA5-Land observations as training data.

The pipeline:

1. **Data preparation** — loads hourly station temperature records and co-located ERA5-Land temperature data, aligns them by timestamp, and engineers cyclical time features (hour-of-day and day-of-year, encoded as sine/cosine pairs) to capture diurnal and seasonal cycles.
2. **Model training** — trains an XGBoost regressor to predict station temperature from ERA5-Land temperature and time features, using early stopping to select the optimal number of boosting rounds.
3. **Model diagnostics** — evaluates performance via learning curves, k-fold cross-validation, feature importance, permutation importance, and SHAP value analysis for model interpretability.
4. **Out-of-domain validation** — tests generalization by applying the trained model to station data withheld from training.
5. **Long-term reconstruction** — applies the trained model to decades of historical ERA5-Land data to produce a bias-corrected temperature time series, followed by trend analysis (linear regression, OLS with confidence intervals, anomaly detection relative to a historical baseline, and seasonal trend decomposition).

## Repository Contents

- `ML_Temp_Recons.ipynb` — the full analysis notebook, from data loading through model training, validation, and long-term trend reconstruction.

> **Note:** Raw data files (station and ERA5-Land CSVs) are not included in this repository. See [Data](#data) below.

## Setup

### Environment

This project uses Python 3.11 in a dedicated conda environment.

```bash
conda create -n MLtemp python=3.11 -y
conda activate MLtemp
conda install -c conda-forge numpy pandas matplotlib scikit-learn seaborn scipy statsmodels jupyter ipykernel -y
pip install "numpy<2" xgboost==2.1.4 shap==0.45.1
```

> **Note:** `shap==0.45.1` requires NumPy 1.x; installing NumPy 2.x will cause import errors.

### Data

The notebook expects the following files in its working directory (not included in this repo):

- Station temperature CSVs (with `Date` and temperature columns)
- Co-located ERA5-Land hourly CSVs for the same stations
- A folder of long-term ERA5-Land CSVs (with `date` and `temperature` columns) covering the historical reconstruction period, for the trend analysis section

Update the relevant file paths in the notebook to point to your local data locations before running.

### Running

Open `ML_Temp_Recons.ipynb` in Jupyter or VS Code, select the `MLtemp` kernel, and run cells sequentially from top to bottom (Run All is recommended, since some variables are reused/reassigned later in the notebook).

## Methods

- **Model:** XGBoost regression (`xgboost==2.1.4`)
- **Interpretability:** SHAP values (`shap==0.45.1`)
- **Validation:** k-fold cross-validation, out-of-domain station testing
- **Trend analysis:** Ordinary least squares regression with confidence intervals, anomaly analysis relative to a historical baseline period

