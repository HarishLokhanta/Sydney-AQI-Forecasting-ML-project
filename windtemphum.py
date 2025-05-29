#!/usr/bin/env python3
"""
Random-Forest meteorology forecaster (ENGG2112 style)

✓ Trains horizon-wise RF models   (H ∈ {1, 3}  hours) on Sydney-wide averages
✓ Saves:
       models/rf_sydney_<stub>_t+{H}.pkl
       reports/metrics_meteo_rf.csv     (MSE + R² for each meteo target)
       reports/aqi_forecast_all.csv     (merged with existing LIN PM/NO₂)
"""
from pathlib import Path
import re, warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------- paths ------------------------------------------------------------
ROOT   = Path(__file__).parent
DATA   = ROOT / "data"
MODELS = ROOT / "models"  ; MODELS.mkdir(exist_ok=True)
REPORT = ROOT / "reports" ; REPORT.mkdir(exist_ok=True)

# ---------- helpers ----------------------------------------------------------
def clean(col: str) -> str:
    col = col.lower().replace("pm2.5","pm25")
    col = re.sub(r"[^0-9a-z_]+","_", col)
    return re.sub(r"_+","_", col).strip("_")

def seasonal_features(df: pd.DataFrame) -> None:
    df["hr_sin"]  = np.sin(2*np.pi*df.index.hour      / 24)
    df["hr_cos"]  = np.cos(2*np.pi*df.index.hour      / 24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek / 7)

def lag_roll(df: pd.DataFrame, p: str) -> None:
    df[f"{p}_lag1"]  = df[p].shift(1)
    df[f"{p}_lag3"]  = df[p].shift(3)
    df[f"{p}_roll3"] = df[p].rolling(3,1).mean()
    df[f"{p}_roll6"] = df[p].rolling(6,1).mean()

# ---------- load & preprocess ------------------------------------------------
dfs = [pd.read_csv(DATA/f, index_col=0, parse_dates=True)
       for f in ("Final_Wind.csv","humtemp_all.csv","pm25_no2_all.csv")]
wide = pd.concat(dfs, axis=1).sort_index().dropna(how="all")
wide.columns = [clean(c) for c in wide.columns]

# add overall pollutant columns for AQI merge later
wide["pm25"] = wide[[c for c in wide if c.startswith("pm25")]].mean(axis=1)
wide["no2"]  = wide[[c for c in wide if c.startswith("no2")]].mean(axis=1)
# drop older pollutant cols
wide.drop(columns=[c for c in wide if (c.startswith("pm25") or c.startswith("no2")) and '_' in c], inplace=True)

# date/time features + lags for pollutants
seasonal_features(wide)
for p in ("pm25","no2"):
    lag_roll(wide, p)

# ---------- compute Sydney-wide meteo targets --------------------------------
stubs = ["wsp_1h_average_m_s","temp_1h_average_c","humid_1h_average_%"]
met_columns = []
for stub in stubs:
    cols = [c for c in wide.columns if c.endswith(stub)]
    if cols:
        wide[f"sydney_{stub}"] = wide[cols].mean(axis=1)
        met_columns.append(f"sydney_{stub}")

# ---------- train RF models --------------------------------------------------
HORIZONS = (1,3)
RF_KW    = dict(n_estimators=300, max_depth=18,
                min_samples_leaf=2, n_jobs=-1, random_state=42)
metrics = []
for tgt in met_columns:
    for H in HORIZONS:
        y = wide[tgt].shift(-H)
        df = pd.concat([wide.drop(columns=[tgt]), y.rename("y")], axis=1).dropna()
        X, y_all = df.drop(columns="y"), df["y"]
        split = int(len(X)*0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]
        mdl = RandomForestRegressor(**RF_KW).fit(X_tr, y_tr)
        y_hat = mdl.predict(X_te)
        mse = mean_squared_error(y_te, y_hat)
        r2  = r2_score(y_te, y_hat)
        metrics.append({"target":tgt,"horizon":H,"mse":mse,"r2":r2})
        joblib.dump({"model":mdl,"features":list(X_tr)}, MODELS/f"rf_{tgt}_t+{H}.pkl")
        print(f"Trained RF {tgt} t+{H}: MSE={mse:.3f}, R²={r2:.3f}")
# save metrics
pd.DataFrame(metrics).to_csv(REPORT/"metrics_meteo_rf.csv",index=False)

# ---------- one-shot AQI+meteo merge -----------------------------------------
def predict(pkl: Path, row: pd.Series) -> float:
    obj = joblib.load(pkl)
    vals = row.reindex(columns=obj["features"],fill_value=0).values.reshape(1,-1)
    return float(obj["model"].predict(vals)[0])

def aqi_idx(v, p="pm25") -> float:
    bp_pm = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    bp_no = [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]
    bp = bp_pm if p=="pm25" else bp_no
    for lo,hi,a,b in bp:
        if lo<=v<=hi: return (b-a)/(hi-lo)*(v-lo)+a
    return np.nan

latest = wide.dropna().iloc[[-1]]
rows = []
for H in HORIZONS:
    r = {"horizon":H}
    r["pm25_pred"] = predict(MODELS/f"lin_sydney_pm25_t+{H}.pkl", latest)
    r["no2_pred"]  = predict(MODELS/f"lin_sydney_no2_t+{H}.pkl",  latest)
    for tgt in met_columns:
        r[f"{tgt}_pred"] = predict(MODELS/f"rf_{tgt}_t+{H}.pkl", latest)
    r["pm25_index"] = aqi_idx(r["pm25_pred"],"pm25")
    r["no2_index"]  = aqi_idx(r["no2_pred"],"no2")
    r["aqi"]        = max(r["pm25_index"], r["no2_index"])
    rows.append(r)
pd.DataFrame(rows).to_csv(REPORT/"aqi_forecast_all.csv",index=False)
print("✓ AQI+meteo forecast saved → reports/aqi_forecast_all.csv")