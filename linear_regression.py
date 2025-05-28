#!/usr/bin/env python3
"""
Linear-Regression AQI forecaster (multi-suburb, never-blank)
* If a suburb lacks a pollutant model, fall back to the
  Sydney-average model for that pollutant / horizon.
* Adds seasonal + lag/rolling features
* Outputs:
    models/lin_<suburb|sydney>_<pollutant>_t+{1|3}.pkl
    reports/metrics_linear_all.csv
    reports/aqi_forecast_suburbs.csv
"""
# ───────────────────────────────────────────────────────────
from pathlib import Path
import re, warnings, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA   = ROOT / "data";   MODELS = ROOT / "models";  REPORT = ROOT / "reports"
MODELS.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────
def clean(c:str)->str:
    c=c.lower().replace("pm2.5","pm25"); c=re.sub(r"[^0-9a-z_]+","_",c)
    return re.sub(r"_+","_",c).strip("_")

def add_seasonal(df):
    df["hr_sin"]=np.sin(2*np.pi*df.index.hour/24)
    df["hr_cos"]=np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"]=np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"]=np.cos(2*np.pi*df.index.dayofweek/7)

def add_lag_roll(df,col):
    df[f"{col}_lag1"]=df[col].shift(1)
    df[f"{col}_lag3"]=df[col].shift(3)
    df[f"{col}_roll3"]=df[col].rolling(3,1).mean()
    df[f"{col}_roll6"]=df[col].rolling(6,1).mean()

def aqi_idx(v,p="pm25"):
    bp_pm=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),
           (250.5,350.4,301,400),(350.5,500.4,401,500)]
    bp_no=[(0,53,0,50),(54,100,51,100),(101,360,101,150),
           (361,649,151,200),(650,1249,201,300),
           (1250,1649,301,400),(1650,2049,401,500)]
    bp = bp_pm if p=="pm25" else bp_no
    for lo,hi,a,b in bp:
        if lo<=v<=hi: return (b-a)/(hi-lo)*(v-lo)+a
    return np.nan

def load_or_nan(stub,H):
    p = MODELS/f"{stub}_t+{H}.pkl"
    return None if not p.exists() else joblib.load(p)

def predict_from(obj,row):
    vec=row.fillna(0).reindex(columns=obj["features"],fill_value=0).values
    return float(obj["model"].predict(vec.reshape(1,-1))[0])

# ── 1. load data ───────────────────────────────────────────
dfs=[pd.read_csv(DATA/f,index_col=0,parse_dates=True)
     for f in ("Final_Wind.csv","humtemp_all.csv","pm25_no2_all.csv")]
wide=pd.concat(dfs,axis=1).sort_index().dropna(how="all")
wide.columns=[clean(c) for c in wide.columns]

pm_cols  =[c for c in wide if "_pm25" in c]
no2_cols =[c for c in wide if "_no2"  in c]
suburbs  =sorted({c.split("_pm25")[0].split("_no2")[0] for c in pm_cols+no2_cols})
print("Detected suburbs:",", ".join(suburbs))

# Sydney averages
wide["sydney_pm25"]=wide[pm_cols].mean(axis=1)
wide["sydney_no2"] =wide[no2_cols].mean(axis=1)

# 2. features
add_seasonal(wide)
for col in pm_cols+no2_cols+["sydney_pm25","sydney_no2"]:
    add_lag_roll(wide,col)

# 3. train
HORIZONS=(1,3); metrics=[]
for sb in suburbs+["sydney"]:
    for pol in ("pm25","no2"):
        target = (f"sydney_{pol}" if sb=="sydney"
                  else next((c for c in wide if c.startswith(f"{sb}_{pol}")),None))
        if not target: continue
        for H in HORIZONS:
            df = pd.concat([wide.drop(columns=[target]),
                            wide[target].shift(-H).rename("y")],axis=1).dropna(subset=["y"])
            if len(df)<2: continue
            X = df.drop(columns="y").fillna(0); y=df["y"]
            if len(df)>=10:
                cut=int(len(df)*0.8); X_tr,X_te=X[:cut],X[cut:]; y_tr,y_te=y[:cut],y[cut:]
            else:                     # micro-dataset
                X_tr,X_te,X_te = X,X,y*0    # fake test to keep shapes
                y_tr,y_te = y,y
            mdl=LinearRegression().fit(X_tr,y_tr)
            mse=r2=np.nan
            if len(df)>=10:
                y_hat=mdl.predict(X_te); mse=mean_squared_error(y_te,y_hat); r2=r2_score(y_te,y_hat)
            metrics.append(dict(suburb=sb,pollutant=pol,horizon=H,mse=mse,r2=r2))
            joblib.dump({"model":mdl,"features":list(X_tr.columns)},
                        MODELS/f"lin_{sb}_{pol}_t+{H}.pkl")
pd.DataFrame(metrics).to_csv(REPORT/"metrics_linear_all.csv",index=False)
metric_df=pd.DataFrame(metrics)

# 4. forecast with Sydney fallback
latest=wide.iloc[[-1]]
rows=[]
for H in HORIZONS:
    syd_pm = predict_from(load_or_nan("lin_sydney_pm25",H),latest)
    syd_no = predict_from(load_or_nan("lin_sydney_no2" ,H),latest)
    for sb in suburbs+["sydney"]:
        obj_pm = load_or_nan(f"lin_{sb}_pm25",H) or load_or_nan("lin_sydney_pm25",H)
        obj_no = load_or_nan(f"lin_{sb}_no2" ,H) or load_or_nan("lin_sydney_no2" ,H)
        pm  = predict_from(obj_pm,latest)
        no2 = predict_from(obj_no,latest)
        m_pm  = metric_df.query("suburb==@sb and pollutant=='pm25' and horizon==@H")
        m_no2 = metric_df.query("suburb==@sb and pollutant=='no2' and horizon==@H")
        rows.append(dict(
            suburb=sb,horizon=H,
            pm25_pred=pm,no2_pred=no2,
            pm25_index=aqi_idx(pm,"pm25"), no2_index=aqi_idx(no2,"no2"),
            pm25_mse =m_pm["mse"].iloc[0] if not m_pm.empty else np.nan,
            pm25_r2  =m_pm["r2"].iloc[0]  if not m_pm.empty else np.nan,
            no2_mse  =m_no2["mse"].iloc[0]if not m_no2.empty else np.nan,
            no2_r2   =m_no2["r2"].iloc[0] if not m_no2.empty else np.nan,
            aqi = max(aqi_idx(pm,"pm25"), aqi_idx(no2,"no2"))
        ))
pd.DataFrame(rows).to_csv(REPORT/"aqi_forecast_suburbs.csv",index=False)
print("✓ Forecast saved – every suburb now has values (fallback = Sydney average)")