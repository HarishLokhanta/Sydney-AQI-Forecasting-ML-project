#!/usr/bin/env python3
"""
ENGG2112-style forecasters (Linear, Ridge, KNN, Decision-Tree, Random-Forest)
for PM₂.₅, NO₂, and basic meteorological variables.

 • Data: humtemp_all.csv, pm25_no2_all.csv, Final_Wind.csv
 • Horizons: 1 h and 3 h ahead
 • 80 % / 20 % chronological split
 • Metrics: Mean-Squared-Error (MSE) and Coefficient of Determination (R²)

Outputs
 ├─ models/
 │   └─ <model>_<target>_t+{1|3}.pkl
 └─ reports/
     ├─ metrics_all.csv
     ├─ metrics_pollutants_all_models.csv
     └─ aqi_forecast.csv
"""
# ──────────────────────────────────────────────────────────────
from pathlib import Path
import re, os, warnings, joblib
import numpy as np, pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ══════════════════ Helper paths ═════════════════════════════
ROOT   = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA   = ROOT / "data"
MODELS = (ROOT / "models").mkdir(exist_ok=True) or ROOT / "models"
REPORT = (ROOT / "reports").mkdir(exist_ok=True) or ROOT / "reports"

# ═══════════ Load & combine the three CSVs ═══════════════════
def clean(col: str) -> str:
    col = col.lower().replace("pm2.5", "pm25")
    col = re.sub(r"[^0-9a-z_]+", "_", col)
    return re.sub(r"_+", "_", col).strip("_")

dfs = [pd.read_csv(DATA / f, index_col=0, parse_dates=True)
       for f in ("Final_Wind.csv", "humtemp_all.csv", "pm25_no2_all.csv")]

wide = (pd.concat(dfs, axis=1)
          .sort_index()
          .dropna(how="all"))
wide.columns = [clean(c) for c in wide.columns]

# city-wide pollutant means (single target each)
wide["pm25"] = wide[[c for c in wide if "pm25" in c]].mean(axis=1)
wide["no2"]  = wide[[c for c in wide if "no2" in c]].mean(axis=1)
wide.drop(columns=[c for c in wide
                   if ("pm25" in c or "no2" in c) and "_" in c],
          inplace=True)

# ═══════════ Simple temporal / lag / roll features ═══════════
wide["hr_sin"]  = np.sin(2 * np.pi * wide.index.hour      / 24)
wide["hr_cos"]  = np.cos(2 * np.pi * wide.index.hour      / 24)
wide["dow_sin"] = np.sin(2 * np.pi * wide.index.dayofweek / 7)
wide["dow_cos"] = np.cos(2 * np.pi * wide.index.dayofweek / 7)

for p in ("pm25", "no2"):
    wide[f"{p}_lag1"]  = wide[p].shift(1)
    wide[f"{p}_lag3"]  = wide[p].shift(3)
    wide[f"{p}_roll3"] = wide[p].rolling(3, min_periods=1).mean()
    wide[f"{p}_roll6"] = wide[p].rolling(6, min_periods=1).mean()

# ═════════════════ Target lists ══════════════════════════════
SUBURBS = ["randwick", "earlwood", "macquarie_park"]
MET_TARGETS = [c for c in wide.columns
               if any(c.startswith(sb + "_") for sb in SUBURBS)
               and any(k in c for k in ("_wsp_", "_temp_", "_humid_"))
               and c.endswith(("_m_s", "average_c", "_%"))]

TARGETS  = ["pm25", "no2"] + MET_TARGETS
HORIZONS = [1, 3]

# ════════════════ Model zoo from lectures ════════════════════
MODEL_SPECS = {
    "lin":   LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "knn":   KNeighborsRegressor(n_neighbors=5, weights="distance"),
    "tree":  DecisionTreeRegressor(max_depth=12, random_state=42),
    "rf":    RandomForestRegressor(n_estimators=300,
                                   max_depth=18,
                                   min_samples_leaf=2,
                                   n_jobs=-1,
                                   random_state=42),
}

# ══════════════ Training / evaluation loop ═══════════════════
rows_all = []

for tgt in TARGETS:
    for H in HORIZONS:
        # shift target
        y_shifted = wide[tgt].shift(-H)
        X = wide.drop(columns=[tgt]).dropna()
        df = pd.concat([X, y_shifted.rename("y")], axis=1).dropna()

        X_all, y_all = df.drop(columns="y"), df["y"]
        split = int(len(X_all) * 0.8)          # chronological split

        X_tr, X_te = X_all.iloc[:split], X_all.iloc[split:]
        y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]

        for tag, model in MODEL_SPECS.items():
            mdl = model.fit(X_tr, y_tr)
            y_hat = mdl.predict(X_te)
            mse   = mean_squared_error(y_te, y_hat)
            r2    = r2_score(y_te, y_hat)

            rows_all.append({
                "model": tag,
                "target": tgt,
                "horizon": H,
                "mse": mse,
                "r2": r2,
            })

            # save model + feature list
            joblib.dump({"model": mdl, "features": list(X_tr)},
                        MODELS / f"{tag}_{tgt}_t+{H}.pkl")

            print(f"✓ {tag.upper():5s}  {tgt:25s}  t+{H}: MSE = {mse:.3f}, R² = {r2:.3f}")

# ═════════════ Write out the metrics tables ══════════════════
metrics_df = pd.DataFrame(rows_all)
metrics_df.to_csv(REPORT / "metrics_all.csv", index=False)
# pollutant-only subset
metrics_df[metrics_df.target.isin(["pm25", "no2"])].to_csv(REPORT / "metrics_pollutants_all_models.csv", index=False)
print("✓ metrics saved  → reports/*.csv")

# ═════════════ Quick AQI forecast with RF models ═════════════
def quick(path: Path, row: pd.Series) -> float:
    obj   = joblib.load(path)
    feats = row.reindex(columns=obj["features"], fill_value=0).values
    return float(obj["model"].predict(feats.reshape(1, -1))[0])

latest = wide.dropna().iloc[[-1]]
out = []
for H in HORIZONS:
    pm  = quick(MODELS / f"rf_pm25_t+{H}.pkl", latest)
    no2 = quick(MODELS / f"rf_no2_t+{H}.pkl",  latest)

    # very simple Australian AQI indices
    def aqi_idx(val, pollutant="pm25"):
        bp = ((0,12,0,50), (12.1,35.4,51,100), (35.5,55.4,101,150),
              (55.5,150.4,151,200), (150.5,250.4,201,300),
              (250.5,350.4,301,400), (350.5,500.4,401,500))
        for lo, hi, a, b in bp:
            if lo <= val <= hi:
                return (b - a) / (hi - lo) * (val - lo) + a
        return np.nan

    aqi_pm  = aqi_idx(pm,  "pm25")
    aqi_no2 = aqi_idx(no2, "no2")
    out.append(dict(horizon=H, pm25_pred=pm, no2_pred=no2,
                    pm25_index=aqi_pm, no2_index=aqi_no2,
                    aqi=max(aqi_pm, aqi_no2)))

pd.DataFrame(out).to_csv(REPORT / "aqi_forecast.csv", index=False)
print("✓ AQI forecast saved → reports/aqi_forecast.csv")