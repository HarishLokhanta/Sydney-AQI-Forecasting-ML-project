import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from aqi_utils import aqi

# ─── Paths ─────────────────────────────────────
BASE    = Path(__file__).parent
MODELS  = BASE/"models"
REPORT  = BASE/"reports"
DATA    = BASE

# ─── Load metrics & AQI history ───────────────
metrics_df  = pd.read_csv(REPORT/"metrics.csv")
forecast_df = pd.read_csv(REPORT/"aqi_forecast.csv")

# ─── UI Setup ──────────────────────────────────
st.title("Sydney AQI Forecaster  🌬️")

suburb  = st.selectbox("Choose suburb",
            ["randwick","earlwood","macquarie_park"])
horizon = st.slider("Forecast horizon (hours)",1,6,3,step=2)
modeltyp= st.radio("Model",["XGBoost","Random Forest"])
st.markdown(f"**Selected:** {suburb.title()}, {horizon} h, {modeltyp}")

# ─── Helper to align features ─────────────────
def align_feats(mobj, df_row):
    return df_row.reindex(columns=mobj["features"], fill_value=0)

# ─── Predict button ───────────────────────────
if st.button("Predict"):

    # 1) Load + preprocess same as training
    df = pd.concat([
        pd.read_csv(DATA/"Final_Wind.csv",    index_col=0, parse_dates=True),
        pd.read_csv(DATA/"humtemp_all.csv",   index_col=0, parse_dates=True),
        pd.read_csv(DATA/"pm25_no2_all.csv",  index_col=0, parse_dates=True),
    ],axis=1).sort_index().dropna(how="all")

    df.columns = [c.lower().replace("pm2.5","pm25")
                         .replace(" ","_").replace("/","_")
                  for c in df.columns]
    df["pm25"] = df[[c for c in df if "pm25" in c]].mean(axis=1)
    df["no2" ] = df[[c for c in df if "no2"  in c]].mean(axis=1)

    # time + lag/roll
    df["hr_sin"]  = np.sin(2*np.pi*df.index.hour/24)
    df["hr_cos"]  = np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek/7)
    for p in ["pm25","no2"]:
        for L in (1,3,6): df[f"{p}_lag{L}"] = df[p].shift(L)
        for W in (3,6,12): df[f"{p}_roll{W}"] = df[p].rolling(W,1).mean()

    latest = df.dropna().iloc[[-1]]

    # 2) Pollutant forecasts
    tag_pm  = "xgb" if modeltyp=="XGBoost" else "rf"
    tag_no2 = tag_pm  # same choice for both

    pm_m  = joblib.load(MODELS/f"{tag_pm}_pm25_t+{horizon}.pkl")
    no2_m = joblib.load(MODELS/f"{tag_no2}_no2_t+{horizon}.pkl")

    pm_df  = align_feats(pm_m, latest)
    no2_df = align_feats(no2_m, latest)

    pm_pred  = float(pm_m["model"].predict(pm_df)[0])
    no2_pred = float(no2_m["model"].predict(no2_df)[0])
    aqi_val,_ = aqi(pm_pred, no2_pred)

    st.metric("PM₂.₅ [µg/m³]", f"{pm_pred:.1f}")
    st.metric("NO₂ [ppb]",     f"{no2_pred:.1f}")
    st.subheader(f"AQI = {aqi_val:.0f}")

    # 3) Meteorology forecasts via RF
    def predict_met(var):
        mp = MODELS/f"rf_{suburb}_{var}_t+{horizon}.pkl"
        if not mp.exists(): return None
        mo = joblib.load(mp)
        dfm = align_feats(mo, latest)
        return float(mo["model"].predict(dfm)[0])

    wsp = predict_met("wsp_1h_average_m_s")
    tmp = predict_met("temp_1h_average_°c")
    hum = predict_met("humid_1h_average_%")

    if wsp is not None: st.metric("Wind [m/s]", f"{wsp:.1f}")
    else:               st.write("No wind model for this suburb.")
    if tmp is not None: st.metric("Temp [°C]",  f"{tmp:.1f}")
    else:               st.write("No temp model for this suburb.")
    if hum is not None: st.metric("Humidity [%]", f"{hum:.1f}")
    else:               st.write("No humidity model for this suburb.")

# ─── Show past metrics & forecasts ───────────
st.markdown("---")
st.subheader("Pollutant RMSE")
st.table(metrics_df)
st.subheader("Latest AQI Forecast")
st.table(forecast_df)