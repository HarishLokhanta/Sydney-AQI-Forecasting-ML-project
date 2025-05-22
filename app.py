#!/usr/bin/env python3
# -------------------------------------------------------------
#   Sydney-wide AQI dashboard  (static forecasts – no live ML)
# -------------------------------------------------------------
import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib   import Path
import datetime as dt

# ───────────────  paths & quick file check  ──────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
REP_DIR  = BASE / "reports"

FC_PATH  = REP_DIR / "aqi_forecast_all.csv"      # created by RF pipeline
if not FC_PATH.exists():
    st.stop()
forecast = pd.read_csv(FC_PATH)                 # two rows: 1 h & 3 h

hist_raw = pd.read_csv(DATA_DIR / "pm25_no2_all.csv",
                       index_col=0, parse_dates=True)
hist_raw.columns = [c.lower().replace("pm2.5", "pm25") for c in hist_raw.columns]
hist_raw["pm25"] = hist_raw[[c for c in hist_raw if "pm25" in c]].mean(axis=1)
hist_raw["no2"]  = hist_raw[[c for c in hist_raw if "no2"  in c]].mean(axis=1)

# ───────────────  tiny AQI helpers  ─────────────────────────
_PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
            (55.5,150.4,151,200),(150.5,250.4,201,300),
            (250.5,350.4,301,400),(350.5,500.4,401,500)]
_NO2_BP  = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
            (361,649,151,200),(650,1249,201,300),
            (1250,1649,301,400),(1650,2049,401,500)]
def _interp(v,bp):
    for lo,hi,a,b in bp:
        if lo<=v<=hi:return (b-a)/(hi-lo)*(v-lo)+a
    return np.nan
def aqi_idx(pm,no2_ppb):  # ppb → µg m-3 ≈ ×0.522
    return max(_interp(pm,_PM25_BP), _interp(no2_ppb*0.522,_NO2_BP))
def aqi_cat(aqi):
    if aqi<=50 :return"Good","#4CAF50"
    if aqi<=100:return"Moderate","#FFEB3B"
    if aqi<=150:return"Unhealthy-SG","#FF9800"
    if aqi<=200:return"Unhealthy","#F44336"
    if aqi<=300:return"Very Unhealthy","#9C27B0"
    return"Hazardous","#795548"

# ───────────────  Streamlit page config  ─────────────────────
st.set_page_config("Sydney AQI", "🌬️", layout="wide")

# ───────────────  sidebar  ───────────────────────────────────
st.sidebar.title("Sydney AQI")
resp_status = st.sidebar.radio("Respiratory status",
                               ["Healthy","Asthma","COPD / Bronchitis","Other"],
                               key="resp")
st.sidebar.caption(f"Local time {dt.datetime.now():%Y-%m-%d  %H:%M}")

# ───────────────  tabs  ─────────────────────────────────────-
tab_fc, tab_hist = st.tabs(["🔮 Forecast", "📈 History"])

# ============================================================
# 🔮  F O R E C A S T
# ============================================================
with tab_fc:
    st.header("City-wide forecast")

    colL, colR = st.columns([2,1], gap="small")

    with colL:
        hr = st.radio("Horizon (hours)", forecast["horizon"], index=1,
                      horizontal=True)
        if st.button("Get forecast"):
            row   = forecast.loc[forecast.horizon==hr].iloc[0]
            pm25  = row.pm25_pred
            no2   = row.no2_pred
            aqi   = row.aqi
            cat, col = aqi_cat(aqi)

            # advice text
            def advice(v, status):
                if status=="Healthy":
                    if v<=100:return"Air quality is acceptable."
                    if v<=150:return"Consider reducing heavy exertion."
                    return"Limit outdoor time & monitor symptoms."
                if v<=50 :return"Air is good for normal activity."
                if v<=100:return"Reduce strenuous activity."
                if v<=150:return"Limit time outdoors; keep medication handy."
                return"Avoid outdoor activity; stay indoors."
            st.info(advice(aqi, resp_status))

            st.markdown(f"<div style='background:{col};padding:6px 14px;"
                        f"border-radius:6px;width:max-content;font-size:1.1rem'>"
                        f"<b>AQI {aqi:.0f}</b> – {cat}"
                        f"</div>", unsafe_allow_html=True)
            m1,m2 = st.columns(2, gap="small")
            m1.metric("PM₂.₅ (µg m⁻³)", f"{pm25:.1f}")
            m2.metric("NO₂ (ppb)",      f"{no2:.1f}")

    with colR:
        last48 = hist_raw[["pm25","no2"]].last("48H")
        fig = px.line(last48, labels={"value":"Conc.","index":"time"},
                      title="Observed concentrations – 48 h",
                      height=280, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Stored forecast table")
        st.dataframe(forecast.style.format({"pm25_pred":"{:.1f}",
                                            "no2_pred":"{:.1f}",
                                            "aqi":"{:.0f}"}),
                     hide_index=True, use_container_width=True)

# ============================================================
# 📈  H I S T O R Y
# ============================================================
with tab_hist:
    st.header("City-wide AQI – last 7 days")

    h = hist_raw.resample("1H").mean()
    h["AQI"] = [aqi_idx(r.pm25, r.no2) for r in h.itertuples()]
    fig_h = px.area(h.last("7D"), y="AQI", labels={"index":"time"},
                    template="plotly_white")
    fig_h.add_hline(y=100, line_dash="dot", line_color="red")
    st.plotly_chart(fig_h, use_container_width=True)

    # optional metrics file
    if (REP_DIR/"metrics_meteo_rf.csv").exists():
        mets = (pd.read_csv(REP_DIR/"metrics_meteo_rf.csv")
                   .query("scope=='city'"))
        st.subheader("RF meteorology – hold-out performance")
        st.dataframe(mets[["target","horizon","mse","r2"]]
                     .style.format({"mse":"{:.3f}","r2":"{:.3f}"}),
                     use_container_width=True)