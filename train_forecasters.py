#!/usr/bin/env python3
"""
Train XGBoost & Random-Forest forecasters for

 • PM₂.₅, NO₂  (horizons = 1 h, 3 h)
 • Wind-speed, Temperature, Humidity (Randwick, Earlwood, Macquarie Park)

Outputs → models/
  rf_<target>_t+<H>.pkl         (all targets)
  xgb_<target>_t+<H>.pkl        (pollutants only)

Metrics
  reports/metrics.csv           ← pollutants only
  reports/metrics_all.csv       ← pollutants + meteorology
  reports/aqi_forecast.csv      ← quick 1 h & 3 h AQI forecast

Set environment variable **OPTUNA_TRIALS** to control how many hyper-parameter
trials to run (0 = skip Optuna and use default XGB params for fastest run).
"""
from pathlib import Path
import warnings, re, os, joblib, json  # json kept for potential future logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Monkey-patch for SHAP API changes (v0.47+ removed summary_plot)
# -----------------------------------------------------------------------------
if not hasattr(shap, "summary_plot"):
    shap.summary_plot = shap.plots.bar  # graceful fallback

# -----------------------------------------------------------------------------
# Optuna toggle (default 0 = no hyper-tuning → fastest)
# -----------------------------------------------------------------------------
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "0"))

# ───────────── AQI helpers ─────────────
AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "no2": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300),
        (1250, 1649, 301, 400), (1650, 2049, 401, 500)]
}

def _aqi_piecewise(pollutant: str, conc: float) -> float:
    """Linear interpolation between AQI break-points."""
    for lo, hi, ilo, ihi in AQI_BREAKPOINTS[pollutant]:
        if lo <= conc <= hi:
            return (ihi - ilo) / (hi - lo) * (conc - lo) + ilo
    return np.nan

def overall_aqi(pm: float, no2: float):
    """Return overall AQI + sub-indices dict."""
    a = _aqi_piecewise("pm25", pm)
    b = _aqi_piecewise("no2",  no2)
    return max(a, b), {"pm25": a, "no2": b}

# ───────────── Paths ─────────────
ROOT   = Path(__file__).parent
MODELS = ROOT / "models";  MODELS.mkdir(exist_ok=True)
REPORT = ROOT / "reports"; REPORT.mkdir(exist_ok=True)
DATA   = ROOT

# ───────────── Load & clean data ─────────────

def clean_name(raw: str) -> str:
    raw = raw.lower().replace("pm2.5", "pm25")
    raw = re.sub(r"[^0-9a-z_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw

print("Loading CSVs …")
dfs = [
    pd.read_csv(DATA / "Final_Wind.csv",   index_col=0, parse_dates=True),
    pd.read_csv(DATA / "humtemp_all.csv",  index_col=0, parse_dates=True),
    pd.read_csv(DATA / "pm25_no2_all.csv", index_col=0, parse_dates=True),
]
wide = pd.concat(dfs, axis=1).sort_index().dropna(how="all")
wide.columns = [clean_name(c) for c in wide.columns]

# city-wide pollutant targets
wide["pm25"] = wide[[c for c in wide.columns if "pm25" in c]].mean(axis=1)
wide["no2"]  = wide[[c for c in wide.columns if "no2"  in c]].mean(axis=1)
# drop redundant site-specific pollutant cols
poll_cols = [c for c in wide.columns if ("pm25" in c or "no2" in c) and "_" in c]
wide.drop(columns=[c for c in poll_cols if c not in ("pm25", "no2")], inplace=True)

# basic cyclic time features + lags/rolling means
wide["hr_sin"]  = np.sin(2 * np.pi * wide.index.hour / 24)
wide["hr_cos"]  = np.cos(2 * np.pi * wide.index.hour / 24)
wide["dow_sin"] = np.sin(2 * np.pi * wide.index.dayofweek / 7)
wide["dow_cos"] = np.cos(2 * np.pi * wide.index.dayofweek / 7)
for p in ("pm25", "no2"):
    for L in (1, 3, 6):  wide[f"{p}_lag{L}"]  = wide[p].shift(L)
    for W in (3, 6, 12): wide[f"{p}_roll{W}"] = wide[p].rolling(W, 1).mean()

# ───────────── Targets ─────────────
SUBURBS = ["randwick", "earlwood", "macquarie_park"]
MET_TARGETS = [
    c for c in wide.columns
    if any(c.startswith(f"{sb}_") for sb in SUBURBS)
    and (
        ("_wsp_" in c and c.endswith("_m_s")) or
        ("_temp_" in c and "average" in c)   or
        ("_humid_" in c and c.endswith("_%"))
    )
]

HORIZONS = [1, 3]
WF_CV = TimeSeriesSplit(n_splits=6)   # walk-forward CV for metrics
(REPORT / "shap").mkdir(exist_ok=True)

rows, rows_all = [], []

print("Training models … (N_TRIALS=%d)" % N_TRIALS)

for tgt in ["pm25", "no2"] + MET_TARGETS:
    for H in HORIZONS:
        y = wide[tgt].shift(-H)
        X = wide.drop(columns=[tgt]).dropna()
        df0 = pd.concat([X, y.rename("y")], axis=1).dropna()
        X0, y0 = df0.drop(columns="y"), df0["y"]
        split = int(len(X0) * 0.8)
        Xtr, Xte = X0.iloc[:split], X0.iloc[split:]
        ytr, yte = y0.iloc[:split], y0.iloc[split:]

        # --- Random-Forest (all targets) ------------------------------
        rf = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(Xtr, ytr)
        rmse_rf = np.sqrt(mean_squared_error(yte, rf.predict(Xte)))
        mae_rf  = mean_absolute_error(yte, rf.predict(Xte))
        joblib.dump({"model": rf, "features": list(Xtr.columns)},
                    MODELS / f"rf_{tgt}_t+{H}.pkl")

        # --- XGBoost (pollutants only) -------------------------------
        rmse_xgb = mae_xgb = np.nan
        if tgt in ("pm25", "no2"):

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600, 100),
                    "max_depth":    trial.suggest_int("max_depth", 4, 10),
                    "eta":          trial.suggest_float("eta", 0.01, 0.3, log=True),
                    "subsample":    trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "objective": "reg:squarederror",
                    "n_jobs": -1,
                    "random_state": 42,
                }
                mdl = XGBRegressor(**params)
                rmses, maes = [], []
                for tr_idx, te_idx in WF_CV.split(X0):
                    mdl.fit(X0.values[tr_idx], y0.values[tr_idx])
                    pred = mdl.predict(X0.values[te_idx])
                    rmses.append(np.sqrt(mean_squared_error(y0.values[te_idx], pred)))
                    maes.append(mean_absolute_error(y0.values[te_idx], pred))
                trial.set_user_attr("mae", float(np.mean(maes)))
                return np.mean(rmses)

            study = optuna.create_study(direction="minimize")

            if N_TRIALS > 0:
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            else:  # quick path with hand-picked defaults
                study.enqueue_trial({
                    "n_estimators": 400,
                    "max_depth": 6,
                    "eta": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.96,
                })
                study.optimize(objective, n_trials=1, show_progress_bar=False)

            best_params = study.best_params
            best_params.update({"objective": "reg:squarederror", "n_jobs": -1, "random_state": 42})
            best = XGBRegressor(**best_params)
            best.fit(Xtr.values, ytr)
            rmse_xgb = np.sqrt(mean_squared_error(yte, best.predict(Xte.values)))
            mae_xgb  = mean_absolute_error(yte, best.predict(Xte.values))

            # SHAP explainability
            try:
                expl = shap.TreeExplainer(best)
                shap_vals = expl.shap_values(Xtr.sample(min(4000, len(Xtr))).values)
                shap.plots.bar(shap_vals, max_display=20, show=False)
                shap_path = REPORT / f"shap/shap_{tgt}_t+{H}.png"
                plt.tight_layout()
                plt.savefig(shap_path, dpi=180)
                plt.close()
            except Exception as e:
                print(f"[SHAP] skipped for {tgt} t+{H}:", e)

            joblib.dump({"model": best, "features": list(Xtr.columns)},
                        MODELS / f"xgb_{tgt}_t+{H}.pkl")

        rows_all.append({"target": tgt, "horizon": H,
                         "rf_rmse": rmse_rf, "xgb_rmse": rmse_xgb})
        if tgt in ("pm25", "no2"):
            rows.append({"pollutant": tgt, "horizon": H,
                         "rf_rmse": rmse_rf, "xgb_rmse": rmse_xgb,
                         "rf_mae": mae_rf, "xgb_mae": mae_xgb})

# Save metric tables
pd.DataFrame(rows).to_csv(REPORT / "metrics.csv", index=False)
pd.DataFrame(rows_all).to_csv(REPORT / "metrics_all.csv", index=False)
print("✓ metrics saved")

# ───────────── Quick AQI forecast (for last timestamp) ─────────────

def quick_pred(path: Path, row: pd.DataFrame) -> float:
    obj = joblib.load(path)
    feat = row.reindex(columns=obj["features"], fill_value=0).values
    return float(obj["model"].predict(feat)[0])

latest = wide.iloc[[-1]].copy()  # keep ALL cols for feature alignment
out = []
for H in (1, 3):
    pm  = quick_pred(MODELS / f"xgb_pm25_t+{H}.pkl", latest)
    no2 = quick_pred(MODELS / f"xgb_no2_t+{H}.pkl",  latest)
    aqi, subs = overall_aqi(pm, no2)
    out.append({"horizon": H,
                "pm25_pred": pm,
                "no2_pred": no2,
                "aqi": aqi,
                "pm25_index": subs["pm25"],
                "no2_index": subs["no2"]})

pd.DataFrame(out).to_csv(REPORT / "aqi_forecast.csv", index=False)
print("✓ AQI forecast saved → reports/aqi_forecast.csv")
