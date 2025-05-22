#!/usr/bin/env python3
"""
ENGG2112 – Linear-Regression AQI forecaster (multi-suburb edition)
------------------------------------------------------------------
* Trains one LinearRegression per suburb for every pollutant that
  exists (PM₂.₅, NO₂). A city-wide “sydney” average is also trained.
* Horizons  : 1 h  &  3 h ahead
* Train/test: 80 % / 20 % chronological split  (Week-4)
* Features  : seasonality + lag / rolling means (Week-7)
* Metrics   : MSE + R²                         (Week-8)

Outputs
-------
models/lin_<suburb|sydney>_<pollutant>_t+{1|3}.pkl
reports/metrics_linear_all.csv
reports/aqi_forecast_suburbs.csv
"""
# ────────────────────────────────────────────────────────────
from pathlib import Path
import re, warnings, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

# ───────── paths ────────────────────────────────────────────
ROOT   = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA   = ROOT / "data"
MODELS = ROOT / "models" ; MODELS.mkdir(exist_ok=True)
REPORT = ROOT / "reports"; REPORT.mkdir(exist_ok=True)

# ───────── helpers ──────────────────────────────────────────
def clean(col: str) -> str:
    col = col.lower().replace("pm2.5", "pm25")
    col = re.sub(r"[^0-9a-z_]+", "_", col)
    return re.sub(r"_+", "_", col).strip("_")

def add_seasonal(df: pd.DataFrame) -> None:
    df["hr_sin"]  = np.sin(2*np.pi*df.index.hour      / 24)
    df["hr_cos"]  = np.cos(2*np.pi*df.index.hour      / 24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek / 7)

def add_lag_roll(df: pd.DataFrame, col: str) -> None:
    df[f"{col}_lag1"]  = df[col].shift(1)
    df[f"{col}_lag3"]  = df[col].shift(3)
    df[f"{col}_roll3"] = df[col].rolling(3, 1).mean()
    df[f"{col}_roll6"] = df[col].rolling(6, 1).mean()

def aqi_idx(v, pollutant="pm25"):
    bp_pm = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
             (55.5,150.4,151,200),(150.5,250.4,201,300),
             (250.5,350.4,301,400),(350.5,500.4,401,500)]
    bp_no = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
             (361,649,151,200),(650,1249,201,300),
             (1250,1649,301,400),(1650,2049,401,500)]
    bp = bp_pm if pollutant == "pm25" else bp_no
    for lo, hi, a, b in bp:
        if lo <= v <= hi:
            return (b-a)/(hi-lo)*(v-lo)+a
    return np.nan

def maybe_predict(stub: str, H:int, latest_row: pd.Series):
    pkl = MODELS / f"{stub}_t+{H}.pkl"
    if not pkl.exists():
        return np.nan                     # no monitor at that suburb
    obj = joblib.load(pkl)
    vec = latest_row.reindex(columns=obj["features"], fill_value=0).values
    return float(obj["model"].predict(vec.reshape(1,-1))[0])

# ───────── 1. load & tidy raw data ───────────────────────────
dfs = [pd.read_csv(DATA/f, index_col=0, parse_dates=True)
       for f in ("Final_Wind.csv", "humtemp_all.csv", "pm25_no2_all.csv")]
wide = pd.concat(dfs, axis=1).sort_index().dropna(how="all")
wide.columns = [clean(c) for c in wide.columns]

pm_cols  = [c for c in wide if "pm25" in c and "_" in c]
no2_cols = [c for c in wide if "no2"  in c and "_" in c]

suburbs_pm  = {c.split("_")[0] for c in pm_cols}
suburbs_no2 = {c.split("_")[0] for c in no2_cols}
all_suburbs = sorted(suburbs_pm | suburbs_no2)
print("Detected suburbs:", ", ".join(all_suburbs))

# city-wide mean
wide["sydney_pm25"] = wide[pm_cols ].mean(axis=1)
wide["sydney_no2"]  = wide[no2_cols].mean(axis=1)

add_seasonal(wide)
for col in ("sydney_pm25", "sydney_no2"):
    add_lag_roll(wide, col)

# ───────── 2. train one model per available pollutant ───────
HORIZONS   = (1, 3)
metrics     = []

for sb in all_suburbs + ["sydney"]:
    for pollutant in ("pm25", "no2"):
        target_col = f"{sb}_{pollutant}" if sb != "sydney" else f"sydney_{pollutant}"
        if target_col not in wide.columns:
            continue        # pollutant not measured at this suburb

        for H in HORIZONS:
            y = wide[target_col].shift(-H)
            df_sup = pd.concat([wide.drop(columns=[target_col]),
                                y.rename("y")], axis=1).dropna()
            X_all, y_all = df_sup.drop(columns="y"), df_sup["y"]
            cut = int(len(X_all)*0.8)
            X_tr, X_te = X_all.iloc[:cut], X_all.iloc[cut:]
            y_tr, y_te = y_all.iloc[:cut], y_all.iloc[cut:]

            mdl = LinearRegression().fit(X_tr, y_tr)
            y_hat = mdl.predict(X_te)
            mse, r2 = mean_squared_error(y_te, y_hat), r2_score(y_te, y_hat)
            metrics.append(dict(suburb=sb, pollutant=pollutant,
                                horizon=H, mse=mse, r2=r2))
            joblib.dump({"model": mdl, "features": list(X_tr)},
                        MODELS / f"lin_{sb}_{pollutant}_t+{H}.pkl")
            print(f"✓ LIN {sb:<12s} {pollutant.upper():4s} t+{H}: "
                  f"MSE={mse:6.3f}  R²={r2:5.3f}")

pd.DataFrame(metrics).to_csv(REPORT/"metrics_linear_all.csv", index=False)

# ───────── 3. quick AQI forecast for every suburb ───────────
latest  = wide.dropna().iloc[[-1]]
records = []

for H in HORIZONS:
    for sb in all_suburbs + ["sydney"]:
        pm  = maybe_predict(f"lin_{sb}_pm25", H, latest)
        no2 = maybe_predict(f"lin_{sb}_no2" , H, latest)

        rec = dict(suburb=sb, horizon=H,
                   pm25_pred = pm,  no2_pred = no2,
                   pm25_index= aqi_idx(pm ,"pm25") if not np.isnan(pm ) else np.nan,
                   no2_index = aqi_idx(no2,"no2") if not np.isnan(no2) else np.nan)
        rec["aqi"] = np.nanmax([rec["pm25_index"], rec["no2_index"]])
        records.append(rec)

pd.DataFrame(records).to_csv(REPORT/"aqi_forecast_suburbs.csv", index=False)
print("✓ AQI forecast saved → reports/aqi_forecast_suburbs.csv")