#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
# Enhanced Streamlit dashboard for Sydney-wide AQI forecasting
# ──────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
from pathlib import Path

# ──────────── AQI helpers ─────────────────────────────────────
_PM25_BP = [
    (0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
    (55.5,150.4,151,200),(150.5,250.4,201,300),
    (250.5,350.4,301,400),(350.5,500.4,401,500),
]
_NO2_BP = [
    (0,53,0,50),(54,100,51,100),(101,360,101,150),
    (361,649,151,200),(650,1249,201,300),
    (1250,1649,301,400),(1650,2049,401,500),
]

def _interp(v,bp):
    for lo, hi, ilo, ihi in bp:
        if lo <= v <= hi:
            return (ihi-ilo)/(hi-lo)*(v-lo)+ilo
    return np.nan

def aqi_idx(pm,no2):
    return max(_interp(pm,_PM25_BP),
               _interp(no2*0.522,_NO2_BP))

def aqi_category(aqi: float):
    """Return (label,colour) for AQI badge (badge always green)."""
    if aqi <= 150:
        return "AQI", "#4CAF50"
    return "AQI", "#4CAF50"

# ─── meteorology corrections ───────────────────────────────────
ALPHA,BETA,GAMMA,H0 = 0.20,1.3,0.05,1000
pm_hum_corr  = lambda p,r: p/(1+ALPHA*(r/100)**BETA)
pm_wind_corr = lambda p,w: max(p*(1-GAMMA*w),0.5*p)
pm_temp_corr = lambda p,t: max(p*(H0/(100*t+150)),0.5*p)

# ─────────── utility ───────────────────────────────────────────
def clean_name(raw: str) -> str:
    s = raw.lower().replace("pm2.5","pm25")
    s = re.sub(r"[^0-9a-z_]+","_",s)
    return re.sub(r"_+","_",s).strip("_")

def load_raw_data(data_dir: Path) -> pd.DataFrame:
    df = pd.concat([
        pd.read_csv(data_dir/"Final_Wind.csv",   index_col=0, parse_dates=True),
        pd.read_csv(data_dir/"humtemp_all.csv",  index_col=0, parse_dates=True),
        pd.read_csv(data_dir/"pm25_no2_all.csv", index_col=0, parse_dates=True),
    ], axis=1).sort_index().dropna(how="all")
    df.columns = [clean_name(c) for c in df.columns]
    df["pm25"] = df[[c for c in df if "pm25" in c]].mean(axis=1)
    df["no2"]  = df[[c for c in df if "no2"  in c]].mean(axis=1)
    return df

def align(model_obj: dict, row: pd.Series) -> np.ndarray:
    return row.reindex(columns=model_obj["features"], fill_value=0).values

# ─────────── paths & load CSVs ─────────────────────────────────
BASE       = Path(__file__).resolve().parent
DATA_DIR   = BASE/"data"
REPORT_DIR = BASE/"reports"

metrics_df  = pd.read_csv(REPORT_DIR/"metrics_pollutants_all_models.csv")
forecast_df = pd.read_csv(REPORT_DIR/"aqi_forecast.csv")

# ─────────── Streamlit config & CSS ───────────────────────────
st.set_page_config(page_title="Sydney AQI Forecaster", layout="wide", page_icon="🌬️")
st.markdown("""
    <style>
      .stApp {background-color:#000;}         /* black page bg */
      .css-1d391kg {background-color:#ADD8E6;}   /* light-blue sidebar */
    </style>
""", unsafe_allow_html=True)

# --- ask for respiratory status once ----------------------------
if "health_status" not in st.session_state:
    def _set_health():
        st.session_state["health_status"] = st.session_state["health_temp"]
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<h2 style='text-align:center;'>Your respiratory status</h2>", unsafe_allow_html=True)
        st.radio("", ["Healthy","Asthma","COPD / Bronchitis","Other"],
                 key="health_temp", index=0, horizontal=True, label_visibility="collapsed")
        st.button("Confirm", on_click=_set_health, use_container_width=True)
    st.stop()

health_status = st.session_state["health_status"]

# ─────────── sidebar info ───────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About this app")
    st.write("Sydney-wide 1 h / 3 h AQI forecasts (2024–25 models).")
    st.write("**How AQI affects health**")
    st.markdown("""
      * Good (0–50): minimal risk  
      * Moderate (51–100): sensitive groups reduce exertion  
      * Unhealthy-for-SG (101–150): limit time outdoors  
      * Unhealthy+ (>150): avoid exertion  
    """)
    st.markdown("---")
    st.caption("Data: NSW EPA & BOM • Built with ❤️ & Streamlit")

# ─────────── main tabs ───────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Forecast","📈 Model metrics"])

# ---------------- FORECAST TAB --------------------------------
with tab1:
    st.header("Real-time Sydney-wide AQI forecast")
    cL, cR = st.columns([2,1])

    with cL:
        horizon = st.slider("Horizon (hrs)", 1, 3, 3, step=2)
        mdl     = st.radio("Pollutant model", ["Linear Regression","Random Forest"],
                           horizontal=True)

        if st.button("Run forecast"):
            raw    = load_raw_data(DATA_DIR)
            latest = raw.dropna().iloc[[-1]]

            prefix = "lin_sydney" if mdl == "Linear Regression" else "rf"

            # Load the correct PM2.5 model file
            if mdl == "Random Forest":
                pm_m = joblib.load(
                    BASE/"models"/f"{prefix}_pm_adjust_t+{horizon}.pkl"
                )
            else:
                pm_m = joblib.load(
                    BASE/"models"/f"{prefix}_pm25_t+{horizon}.pkl"
                )

            # NO₂ model remains unchanged
            no2_m = joblib.load(
                BASE/"models"/f"{prefix}_no2_t+{horizon}.pkl"
            )

            pm  = float(pm_m["model"].predict(align(pm_m, latest))[0])
            no2 = float(no2_m["model"].predict(align(no2_m, latest))[0])

            # city-wide meteorology corrections
            def pmet(stub):
                fp = BASE/"models"/f"rf_sydney_{stub}_t+{horizon}.pkl"
                if fp.exists():
                    m = joblib.load(fp)
                    return float(m["model"].predict(align(m, latest))[0])
                return None

            wsp, tmp, rh = (
                pmet("wsp_1h_average_m_s"),
                pmet("temp_1h_average_c"),
                pmet("humid_1h_average_%")
            )

            aqi_raw = aqi_idx(pm, no2)
            pm_adj  = pm
            if rh  is not None: pm_adj = pm_hum_corr(pm_adj, rh)
            if wsp is not None: pm_adj = pm_wind_corr(pm_adj, wsp)
            if tmp is not None: pm_adj = pm_temp_corr(pm_adj, tmp)
            aqi_adj = aqi_idx(pm_adj, no2)

            def advice(a,s):
                if s == "Healthy":
                    return "All good 😊" if a <= 100 else \
                           ("Lighten up" if a <= 150 else "Limit outdoor time")
                else:
                    return "Fine 👍" if a <= 50 else \
                           ("Reduce exertion" if a <= 100 else \
                           ("Limit time outside" if a <= 150 else "Stay indoors"))

            st.info(advice(aqi_adj, health_status))

            st.markdown(
                f"<div style='background:#4CAF50;color:black;padding:8px;"
                f"border-radius:6px;display:inline-block;'>"
                f"<b>AQI {aqi_adj:.0f}</b></div>",
                unsafe_allow_html=True
            )
            st.metric("PM₂.₅ (µg/m³)", f"{pm_adj:.1f}")
            st.metric("NO₂ (ppb)",     f"{no2:.1f}")

    with cR:
        st.subheader("Last 48 h observations")
        df48 = load_raw_data(DATA_DIR)[["pm25","no2"]].last("48H")
        fig  = px.line(df48, labels={"value":"conc.","index":"time"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Latest forecast")
        st.table(forecast_df)

# ---------------- METRICS TAB ---------------------------------
with tab2:
    st.header("Model performance (Sydney)")

    st.subheader("Mean Squared Error by Model")
    fig_mse = px.bar(
        metrics_df, x="model", y="mse",
        color="target", barmode="group",
        labels={"model":"Model","mse":"MSE","target":"Pollutant"},
        color_discrete_sequence=["#4CAF50","#ADD8E6"]
    )
    st.plotly_chart(fig_mse, use_container_width=True)

    st.subheader("Coefficient of Determination R² by Model")
    fig_r2 = px.bar(
        metrics_df, x="model", y="r2",
        color="target", barmode="group",
        labels={"model":"Model","r2":"R²","target":"Pollutant"},
        color_discrete_sequence=["#4CAF50","#ADD8E6"]
    ).update_yaxes(range=[0,1], dtick=0.1)
    st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("7-day AQI history")
    hist = load_raw_data(DATA_DIR)[["pm25","no2"]].resample("1H").mean()
    hist["AQI"] = hist.apply(lambda r: aqi_idx(r.pm25,r.no2), axis=1)
    wk = hist.last("7D")
    fig2 = px.area(wk, y="AQI", title="Hourly AQI – last 7 days",
                   color_discrete_sequence=["#FFFF00"])
    fig2.add_hline(y=100, line_dash="dot", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)