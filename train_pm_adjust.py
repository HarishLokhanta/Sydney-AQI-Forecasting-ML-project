#!/usr/bin/env python3
"""
train_pm_adjust.py — Random-Forest adjustment for PM2.5
uses lin_pm25 + RF wind, temp (+ optional humidity) forecasts
"""

# ─────────────────────────────────────────────────────────────
from pathlib import Path
import re, warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent
DATA   = ROOT / "data"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

# ---------- 1. load & tidy raw data --------------------------
def clean(c):
    c = c.lower().replace("pm2.5", "pm25")
    c = re.sub(r"[^0-9a-z_]+", "_", c)
    return re.sub(r"_+", "_", c).strip("_")

dfs = [pd.read_csv(DATA/f, index_col=0, parse_dates=True)
       for f in ("Final_Wind.csv", "humtemp_all.csv", "pm25_no2_all.csv")]
wide = pd.concat(dfs, axis=1).sort_index().dropna(how="all")
wide.columns = [clean(c) for c in wide.columns]

wide["pm25"] = wide[[c for c in wide if "pm25" in c]].mean(axis=1)
wide.drop(columns=[c for c in wide if "pm25" in c and "_" in c], inplace=True)

# ---------- 2. seasonality + lag features (Week-7) -----------
wide["hr_sin"]  = np.sin(2*np.pi*wide.index.hour/24)
wide["hr_cos"]  = np.cos(2*np.pi*wide.index.hour/24)
wide["dow_sin"] = np.sin(2*np.pi*wide.index.dayofweek/7)
wide["dow_cos"] = np.cos(2*np.pi*wide.index.dayofweek/7)
for p in ("pm25",):
    wide[f"{p}_lag1"]  = wide[p].shift(1)
    wide[f"{p}_lag3"]  = wide[p].shift(3)
    wide[f"{p}_roll3"] = wide[p].rolling(3,1).mean()
    wide[f"{p}_roll6"] = wide[p].rolling(6,1).mean()

# ---------- 3. safe batch-predict ----------------------------
def safe_predict(pkl_stub: str, H: int, X: pd.DataFrame) -> pd.Series:
    pkl = MODELS / f"{pkl_stub}_t+{H}.pkl"
    if not pkl.exists():
        print(f"[warn] {pkl.name} missing → feature filled with 0")
        return pd.Series(0.0, index=X.index)
    obj  = joblib.load(pkl)
    Xsub = X.reindex(columns=obj["features"], fill_value=0).fillna(0)
    return pd.Series(obj["model"].predict(Xsub), index=X.index)

# ---------- 4. train adjustment RF (Week-8) ------------------
RF_KW = dict(n_estimators=300, max_depth=18,
             min_samples_leaf=2, n_jobs=-1, random_state=42)

for H in (1, 3):
    feats = pd.DataFrame(index=wide.index)
    feats["pm_lin"] = safe_predict("lin_pm25", H, wide)
    feats["wsp"]    = safe_predict("rf_randwick_wsp_1h_average_m_s", H, wide)
    feats["temp"]   = safe_predict("rf_randwick_temp_1h_average_c", H, wide)
    feats["humid"]  = safe_predict("rf_randwick_humid_1h_average_%", H, wide)

    data = pd.concat([feats, wide["pm25"].shift(-H).rename("y")], axis=1).dropna()
    X, y = data.drop(columns="y"), data["y"]
    cut  = int(len(X)*0.8)            # Week-4 chrono split
    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    adj = RandomForestRegressor(**RF_KW).fit(X_tr, y_tr)
    mse = mean_squared_error(y_te, adj.predict(X_te))
    print(f"H={H}: adj-RF MSE = {mse:.3f}")

    joblib.dump({"model": adj, "features": list(X_tr.columns)},
                MODELS / f"rf_pm_adjust_t+{H}.pkl")