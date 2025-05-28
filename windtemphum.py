#!/usr/bin/env python3
"""
Random-Forest meteorology forecaster (ENGG2112 style)

✓ Detects **all** suburb-level wind-speed / temperature / humidity columns
  whose names match  *_wsp_*, *_temp_*, *_humid_*.
✓ Trains horizon-wise RF models   (H ∈ {1, 3}  hours).
✓ Adds city-wide “Sydney-avg” targets (mean of all suburbs).
✓ Saves:
       models/rf_<target>_t+{H}.pkl
       reports/metrics_meteo_rf.csv     (MSE + R² + scope=suburb/city)
       reports/aqi_forecast_all.csv     (merged with existing LIN PM/NO₂)
"""
# ─────────────────────────────────────────────────────────────
from pathlib import Path
import re, warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics   import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------- paths ------------------------------------------------------------
ROOT   = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA   = ROOT / "data"
MODELS = ROOT / "models"   ; MODELS.mkdir(exist_ok=True)
REPORT = ROOT / "reports"  ; REPORT.mkdir(exist_ok=True)

# ---------- helpers ----------------------------------------------------------
def clean(col: str) -> str:
    col = col.lower().replace("pm2.5", "pm25")
    col = re.sub(r"[^0-9a-z_]+", "_", col)
    return re.sub(r"_+", "_", col).strip("_")

def seasonal_features(df: pd.DataFrame) -> None:
    df["hr_sin"]  = np.sin(2*np.pi*df.index.hour      / 24)
    df["hr_cos"]  = np.cos(2*np.pi*df.index.hour      / 24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek / 7)

def lag_roll(df: pd.DataFrame, p: str) -> None:
    df[f"{p}_lag1"]  = df[p].shift(1)
    df[f"{p}_lag3"]  = df[p].shift(3)
    df[f"{p}_roll3"] = df[p].rolling(3, 1).mean()
    df[f"{p}_roll6"] = df[p].rolling(6, 1).mean()

# ---------- load & tidy ------------------------------------------------------
dfs = [pd.read_csv(DATA / f, index_col=0, parse_dates=True)
       for f in ("Final_Wind.csv", "humtemp_all.csv", "pm25_no2_all.csv")]

wide = pd.concat(dfs, axis=1).sort_index().dropna(how="all")
wide.columns = [clean(c) for c in wide.columns]

# overall PM / NO2 (for AQI later)
wide["pm25"] = wide[[c for c in wide if "pm25" in c]].mean(axis=1)
wide["no2"]  = wide[[c for c in wide if "no2"  in c]].mean(axis=1)
wide.drop(columns=[c for c in wide if ("pm25" in c or "no2" in c) and "_" in c],
          inplace=True)

seasonal_features(wide)
for p in ("pm25", "no2"):
    lag_roll(wide, p)

# only generate & train city-wide ("Sydney") meteorology targets
for stub in ("wsp_1h_average_m_s", "temp_1h_average_c", "humid_1h_average_%"):
    # compute Sydney-avg directly
    cols = [c for c in wide.columns if c.endswith(stub)]
    if cols:  # skip if, e.g., no humidity in data
        wide[f"sydney_{stub}"] = wide[cols].mean(axis=1)

# now only train on those three city-wide columns:
met_columns = [
    f"sydney_{stub}"
    for stub in ("wsp_1h_average_m_s", "temp_1h_average_c", "humid_1h_average_%")
    if f"sydney_{stub}" in wide.columns
]
# ---------- training -----------------------------------------
HORIZONS = (1, 3)
RF_KW    = dict(n_estimators=300, max_depth=18,
                min_samples_leaf=2, n_jobs=-1, random_state=42)

metrics = []

for tgt in met_columns:
    for H in HORIZONS:
        y = wide[tgt].shift(-H)
        df = pd.concat([wide.drop(columns=[tgt]), y.rename("y")], axis=1).dropna()
        X, y_all = df.drop(columns="y"), df["y"]

        split = int(len(X) * 0.8)                       # chrono 80/20
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]

        mdl = RandomForestRegressor(**RF_KW).fit(X_tr, y_tr)
        y_hat = mdl.predict(X_te)
        mse   = mean_squared_error(y_te, y_hat)
        r2    = r2_score(y_te, y_hat)

        scope = "city" if tgt.startswith("sydney_") else "suburb"
        metrics.append(dict(target=tgt, horizon=H, mse=mse, r2=r2, scope=scope))

        joblib.dump({"model": mdl, "features": list(X_tr)},
                    MODELS / f"rf_{tgt}_t+{H}.pkl")

        print(f"✓ RF {tgt:35s} t+{H}:  MSE={mse:7.3f}   R²={r2:6.3f}")

# save metrics
pd.DataFrame(metrics).to_csv(REPORT / "metrics_meteo_rf.csv", index=False)

# ---------- helper for latest one-shot prediction ---------------------------
def predict(pkl: Path, row: pd.Series) -> float:
    obj   = joblib.load(pkl)
    feats = row.reindex(columns=obj["features"], fill_value=0).values
    return float(obj["model"].predict(feats.reshape(1, -1))[0])

# ---------- AQI merge (unchanged logic) ------------------------------------
def aqi_idx(v, p="pm25"):
    bp_pm = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
             (55.5,150.4,151,200),(150.5,250.4,201,300),
             (250.5,350.4,301,400),(350.5,500.4,401,500)]
    bp_no = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
             (361,649,151,200),(650,1249,201,300),
             (1250,1649,301,400),(1650,2049,401,500)]
    bp = bp_pm if p == "pm25" else bp_no
    for lo, hi, a, b in bp:
        if lo <= v <= hi:
            return (b - a) / (hi - lo) * (v - lo) + a
    return np.nan

latest = wide.dropna().iloc[[-1]]
rows   = []

for H in HORIZONS:
    row = {"horizon": H}
    row["pm25_pred"] = predict(MODELS / f"lin_sydney_pm25_t+{H}.pkl", latest)
    row["no2_pred"]  = predict(MODELS / f"lin_sydney_no2_t+{H}.pkl",  latest)

    # add met predictions (all suburbs + city)
    for tgt in met_columns:
        row[f"{tgt}_pred"] = predict(MODELS / f"rf_{tgt}_t+{H}.pkl", latest)

    row["pm25_index"] = aqi_idx(row["pm25_pred"], "pm25")
    row["no2_index"]  = aqi_idx(row["no2_pred"],  "no2")
    row["aqi"]        = max(row["pm25_index"], row["no2_index"])
    rows.append(row)

pd.DataFrame(rows).to_csv(REPORT / "aqi_forecast_all.csv", index=False)
print("✓ AQI + meteo forecast saved → reports/aqi_forecast_all.csv")