#!/usr/bin/env python3
"""
Train XGBoost & Random-Forest forecasters for PM₂.₅ / NO₂
horizons = {1, 3, 6} hours ahead.

Outputs
• models/xgb_<pollutant>_t+<h>.pkl
• models/rf_<pollutant>_t+<h>.pkl
• reports/metrics.csv
"""
from pathlib import Path
import warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")

# ─────────────────── AQI helpers ────────────────────
AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0,   0,  50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4,101, 150),
        (55.5,150.4,151, 200),
        (150.5,250.4,201, 300),
        (250.5,350.4,301, 400),
        (350.5,500.4,401, 500),
    ],
    "no2": [
        (0,   53,     0,  50),
        (54,  100,   51, 100),
        (101, 360,  101, 150),
        (361, 649,  151, 200),
        (650,1249,  201, 300),
        (1250,1649,301, 400),
        (1650,2049,401, 500),
    ],
}

def _aqi_piecewise(poll: str, conc: float) -> float:
    """Return AQI sub‑index for a given concentration."""
    for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS[poll]:
        if c_lo <= conc <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo
    return float("nan")

def overall_aqi(pm25: float, no2: float):
    """Return overall AQI (max of sub‑indices) + the individual sub‑indices."""
    pm_sub = _aqi_piecewise("pm25", pm25)
    no2_sub = _aqi_piecewise("no2",  no2)
    return max(pm_sub, no2_sub), {"pm25": pm_sub, "no2": no2_sub}

# ───────────────────────────── paths ────────────────────────────
ROOT   = Path(__file__).parent
DATA   = ROOT
MODELS = ROOT / "models" ; MODELS.mkdir(exist_ok=True)
REPORT = ROOT / "reports" ; REPORT.mkdir(exist_ok=True)

# ──────────────────────────── load data ─────────────────────────
dfs = [
    pd.read_csv(DATA / "Final_Wind.csv",   index_col=0, parse_dates=True),
    pd.read_csv(DATA / "humtemp_all.csv",  index_col=0, parse_dates=True),
    pd.read_csv(DATA / "pm25_no2_all.csv", index_col=0, parse_dates=True),
]
wide = pd.concat(dfs, axis=1).dropna(how="all").sort_index()

# ─────────────────────── feature engineering ────────────────────
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hr_sin"]  = np.sin(2*np.pi*idx.hour/24)
    df["hr_cos"]  = np.cos(2*np.pi*idx.hour/24)
    df["dow_sin"] = np.sin(2*np.pi*idx.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
    return df

def add_lag_roll(df: pd.DataFrame, cols, lags=(1,3,6), wins=(3,6,12)):
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        for W in wins:
            df[f"{c}_roll{W}"] = df[c].rolling(W, min_periods=1).mean()
    return df

POLLUTANTS = ["pm25", "no2"]

# normalize column names: keep suburb + pollutant + metric
wide.columns = [
    c.lower()
     .replace("pm2.5", "pm25")
     .replace("no2",   "no2")
     .replace(" ", "_")
     .replace("[", "")
     .replace("]", "")
     .replace("/", "_")
    for c in wide.columns
]

# ── consolidate site columns into a single average per pollutant ──
pm25_cols = [c for c in wide.columns if "pm25" in c]
no2_cols  = [c for c in wide.columns if "no2"  in c]

# create city‑wide mean series
wide["pm25"] = wide[pm25_cols].mean(axis=1, skipna=True)
wide["no2"]  = wide[no2_cols ].mean(axis=1, skipna=True)

# drop the individual site columns (keep only the consolidated target + exogenous features)
wide = wide.drop(columns=[c for c in pm25_cols + no2_cols if c not in ["pm25", "no2"]])

# fill one-off gaps (already cleaned, but safe)
wide = wide.interpolate("time", limit=1)

wide = add_time_feats(wide)
wide = add_lag_roll(wide, POLLUTANTS)

# ───────────────────────── training loop ────────────────────────
TARGETS  = {"pm25":"pm25", "no2":"no2"}
HORIZONS = [1, 3]                # hours ahead (drop 6 h for now)
cv       = TimeSeriesSplit(n_splits=4)
rows     = []

for target in TARGETS.values():
    for H in HORIZONS:
        y = wide[target].shift(-H)          # t+H label
        X = wide.drop(columns=POLLUTANTS)  # features ≠ raw targets
        df = pd.concat([X, y.rename("y")], axis=1).dropna()

        X, y = df.drop(columns="y"), df["y"]
        split = int(len(df)*0.8)   # chronological split
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # ── Random Forest baseline ──
        rf = RandomForestRegressor(n_estimators=400,
                                   max_depth=None,
                                   min_samples_leaf=2,
                                   n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)
        rf_rmse = np.sqrt(mean_squared_error(y_te, rf_pred))
        joblib.dump(
            {"model": rf, "features": list(X.columns)},
            MODELS / f"rf_{target}_t+{H}.pkl"
        )

        # ── XGBoost (small RS search) ──
        xgb_base = XGBRegressor(objective="reg:squarederror",
                                n_estimators=600,
                                random_state=42, n_jobs=-1)
        param_grid = {
            "max_depth":     [4,6,8],
            "learning_rate": [0.03,0.05,0.1],
            "subsample":     [0.7,0.9,1.0],
            "colsample_bytree":[0.5,0.8,1.0]
        }
        rs = RandomizedSearchCV(
                xgb_base, param_grid, n_iter=15, cv=cv,
                scoring="neg_root_mean_squared_error",
                random_state=42, verbose=0, n_jobs=-1)
        rs.fit(X_tr, y_tr)
        xgb = rs.best_estimator_
        xgb_pred = xgb.predict(X_te)
        xgb_rmse = np.sqrt(mean_squared_error(y_te, xgb_pred))
        joblib.dump(
            {"model": xgb, "features": list(X.columns)},
            MODELS / f"xgb_{target}_t+{H}.pkl"
        )

        rows.append(dict(
            pollutant=target, horizon=H,
            rf_rmse=rf_rmse, xgb_rmse=xgb_rmse
        ))
        print(f"{target} t+{H:>2}h  RF RMSE={rf_rmse:5.2f}  XGB RMSE={xgb_rmse:5.2f}")

pd.DataFrame(rows).to_csv(REPORT/"metrics.csv", index=False)
print("\n✓ training complete — metrics.csv written")

# ───────────────────── real‑time AQI forecast ─────────────────────
rows_aqi = []
current_feats = wide.drop(columns=POLLUTANTS).iloc[-1:].copy()

for H in HORIZONS:
    pm_obj  = joblib.load(MODELS / f"xgb_pm25_t+{H}.pkl")
    no2_obj = joblib.load(MODELS / f"xgb_no2_t+{H}.pkl")

    pm_X  = current_feats.reindex(columns=pm_obj["features"]).fillna(method="ffill", axis=1)
    no2_X = current_feats.reindex(columns=no2_obj["features"]).fillna(method="ffill", axis=1)

    pm_pred  = float(pm_obj["model"].predict(pm_X)[0])
    no2_pred = float(no2_obj["model"].predict(no2_X)[0])

    aqi_val, subs = overall_aqi(pm_pred, no2_pred)
    print(f"\nt+{H}h forecast  ➜  PM2.5={pm_pred:5.1f} µg/m³  "
          f"NO₂={no2_pred:5.1f} ppb  →  AQI={aqi_val:4.0f}")

    rows_aqi.append({
        "horizon":     H,
        "pm25_pred":   pm_pred,
        "no2_pred":    no2_pred,
        "aqi":         aqi_val,
        "pm25_index":  subs["pm25"],
        "no2_index":   subs["no2"],
    })

pd.DataFrame(rows_aqi).to_csv(REPORT / "aqi_forecast.csv", index=False)
print("✓ real‑time AQI forecast saved ➜ reports/aqi_forecast.csv")