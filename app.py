#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
# Enhanced Streamlit dashboard for Sydney AQI forecasting
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np, joblib, re, plotly.express as px
import plotly.express as px  # already present but ensure only one import
from pathlib import Path
from datetime import datetime, timedelta

# ---------- AQI helpers -------------------------------------------------------
_PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
            (55.5,150.4,151,200),(150.5,250.4,201,300),
            (250.5,350.4,301,400),(350.5,500.4,401,500)]
_NO2_BP  = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
            (361,649,151,200),(650,1249,201,300),
            (1250,1649,301,400),(1650,2049,401,500)]

def _interp(val, bp):
    for lo, hi, ilo, ihi in bp:
        if lo <= val <= hi:
            return (ihi-ilo)/(hi-lo)*(val-lo)+ilo
    return np.nan

def aqi_idx(pm, no2):
    return max(_interp(pm, _PM25_BP),
               _interp(no2*0.522, _NO2_BP))

def aqi_category(aqi: float):
    """Return (text, colour) tuple for AQI badge."""
    if aqi <= 50:   return "Good",      "#4CAF50"
    if aqi <= 100:  return "Moderate",  "#FFEB3B"
    if aqi <= 150:  return "Unhealthy for SG", "#FF9800"
    if aqi <= 200:  return "Unhealthy", "#F44336"
    if aqi <= 300:  return "Very Unhealthy", "#9C27B0"
    return "Hazardous", "#795548"

# ---------- meteorology corrections ------------------------------------------
ALPHA, BETA = 0.20, 1.3     # RH → hygroscopic growth
GAMMA = 0.05                # wind dilution
H0 = 1_000                  # reference BL‑height proxy

def pm_hum_corr(pm, rh):   return pm / (1 + ALPHA*(rh/100)**BETA)
def pm_wind_corr(pm, wsp): return max(pm*(1-GAMMA*wsp), 0.5*pm)
def pm_temp_corr(pm, T):
    blh = 100*T + 150
    factor = H0 / blh
    return max(pm*factor, 0.5*pm)

# ---------- utility -----------------------------------------------------------
def clean_name(raw):
    raw = raw.lower().replace("pm2.5","pm25")
    raw = re.sub(r"[^0-9a-z_]+","_",raw)
    return re.sub(r"_+","_",raw).strip("_")

def load_raw_data(base: Path):
    df = pd.concat([
        pd.read_csv(base/"Final_Wind.csv",   index_col=0, parse_dates=True),
        pd.read_csv(base/"humtemp_all.csv",  index_col=0, parse_dates=True),
        pd.read_csv(base/"pm25_no2_all.csv", index_col=0, parse_dates=True),
    ], axis=1).sort_index().dropna(how="all")
    df.columns = [clean_name(c) for c in df.columns]
    df["pm25"] = df[[c for c in df if "pm25" in c]].mean(axis=1)
    df["no2"]  = df[[c for c in df if "no2"  in c]].mean(axis=1)
    return df

def align(model_obj, row):
    return row.reindex(columns=model_obj["features"], fill_value=0).values

# ---------- paths -------------------------------------------------------------
BASE    = Path(__file__).parent
MODELS  = BASE/"models"
REPORT  = BASE/"reports"

metrics_df  = pd.read_csv(REPORT/"metrics.csv")
forecast_df = pd.read_csv(REPORT/"aqi_forecast.csv")

# ---------- Streamlit page config --------------------------------------------
st.set_page_config(
    page_title="Sydney AQI Forecaster",
    layout="wide",
    page_icon="🌬️"
)

# Prompt for respiratory status on first load
if "health_status" not in st.session_state:
    def _set_health():
        st.session_state["health_status"] = st.session_state["health_temp"]

    # center everything in middle column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h2 style='text-align:center;margin-bottom:0.2em;'>Your respiratory status</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;margin-top:0;margin-bottom:1em;'>"
            "Please select your respiratory condition:"
            "</p>",
            unsafe_allow_html=True
        )
        st.radio(
            " ",
            ["Healthy", "Asthma", "COPD / Bronchitis", "Other respiratory condition"],
            key="health_temp",
            index=0,
            label_visibility="hidden"
        )
        st.button("Confirm", on_click=_set_health, use_container_width=True)
        with st.expander("Why your respiratory status matters"):
            st.write(
                "Your respiratory condition influences how sensitive you are to poor air quality. "
                "By letting us know if you have asthma, COPD, or other conditions, "
                "we can tailor health advice according to NSW Health guidelines."
            )
    st.stop()

health_status = st.session_state["health_status"]

# ---------- sidebar -----------------------------------------------------------
with st.sidebar:
    st.title("ℹ️  About this app")
    st.write(
        "Short‑term forecasts of PM₂.₅ & NO₂ for three suburbs in Sydney "
        "using machine‑learning models trained on 2024‑25 data."
    )
    st.write("**How AQI affects health**")
    st.markdown("""
* *Good* (0‑50): minimal risk  
* *Moderate* (51‑100): unusually sensitive people should consider reducing exertion  
* *Unhealthy* (>100): people with **asthma/COPD** may experience symptoms; limit outdoor activity  
* *Hazardous* (>300): everyone should avoid prolonged outdoor exertion  
    """)
    st.write(
        "Respiratory diseases such as **asthma** affect ~11 % of Australians, "
        "and hospital admissions rise measurably on high‑AQI days (Bureau of "
        "Meteorology, 2023)."
    )
    # --- user health status -------------------------------------------
    st.markdown("---")
    st.markdown("")
    st.caption("Made with ❤️  &  Streamlit")

# ---------- main layout -------------------------------------------------------
tab_forecast, tab_history = st.tabs(["🔮 Forecast", "📈 Historical / Model"])

# ==================== FORECAST TAB ============================================
with tab_forecast:
    st.header("Real‑time AQI forecast")
    colL, colR = st.columns([2,1])

    with colL:
        suburb  = st.selectbox("Suburb", ["randwick","earlwood","macquarie_park"])
        horizon = st.slider("Forecast horizon (hours)", 1, 3, 3, step=2)
        model   = st.radio("Model type", ["XGBoost", "Random Forest"], horizontal=True)
        if st.button("Run forecast"):
            raw = load_raw_data(BASE)
            latest = raw.dropna().iloc[[-1]]
            tag = "xgb" if model=="XGBoost" else "rf"

            pm_m  = joblib.load(MODELS/f"{tag}_pm25_t+{horizon}.pkl")
            no2_m = joblib.load(MODELS/f"{tag}_no2_t+{horizon}.pkl")
            pm  = float(pm_m["model"].predict(align(pm_m, latest))[0])
            no2 = float(no2_m["model"].predict(align(no2_m, latest))[0])

            # meteorology sub‑models
            def pmet(stub):
                fp = MODELS/f"rf_{suburb}_{stub}_t+{horizon}.pkl"
                if fp.exists():
                    mo = joblib.load(fp)
                    return float(mo["model"].predict(align(mo, latest))[0])
            wsp = pmet("wsp_1h_average_m_s")
            tmp = pmet("temp_1h_average_c")
            rh  = pmet("humid_1h_average_%")

            aqi_raw = aqi_idx(pm, no2)

            pm_adj = pm
            if rh  is not None: pm_adj = pm_hum_corr(pm_adj, rh)
            if wsp is not None: pm_adj = pm_wind_corr(pm_adj, wsp)
            if tmp is not None: pm_adj = pm_temp_corr(pm_adj, tmp)
            pm_adj = max(pm_adj, 0.5*pm)
            aqi_adj = aqi_idx(pm_adj, no2)
            st.markdown("")  # extra vertical spacing

            # ---- personalised advice ---------------------------------
            def advice_text(aqi_val: float, status: str) -> str:
                # thresholds adapted from NSW Health guidance
                if status == "Healthy":
                    if aqi_val <= 100:
                        return "Air quality is acceptable. Enjoy your outdoor activities! 😊"
                    elif aqi_val <= 150:
                        return "Consider reducing prolonged or heavy outdoor exertion."
                    else:
                        return "Limit time outdoors and monitor any symptoms."
                else:  # sensitive groups
                    if aqi_val <= 50:
                        return "Air quality is good. Normal outdoor activity is fine."
                    elif aqi_val <= 100:
                        return "Reduce or postpone strenuous outdoor activity."
                    elif aqi_val <= 150:
                        return "Limit time outdoors; keep medication handy."
                    else:
                        return "Avoid outdoor activity; stay indoors and follow your action plan."

            user_msg = advice_text(aqi_adj, health_status)
            st.info(user_msg)

            # --- display metrics ------------------------------------------------
            cat_raw, col_raw = aqi_category(aqi_raw)
            cat_adj, col_adj = aqi_category(aqi_adj)
            st.markdown(
                f"<div style='padding:0.3em 0.6em;"
                f"background:{col_raw};color:black;display:inline-block;"
                f"border-radius:6px'>Current AQI&nbsp;{aqi_raw:.0f} – {cat_raw}</div>",
                unsafe_allow_html=True
            )
            st.markdown("")  # visual gap before the metric rows
            st.metric("PM₂.₅ (µg/m³)", f"{pm:.1f}")
            st.metric("NO₂ (ppb)", f"{no2:.1f}")

            st.markdown(
                f"**Weather‑adjusted PM₂.₅**: {pm_adj:.1f} → "
                f"**Adjusted AQI** {aqi_adj:.0f} ({cat_adj})"
            )
            meta = []
            if wsp is not None: meta.append(f"Wind {wsp:.1f} m/s")
            if tmp is not None: meta.append(f"T {tmp:.1f} °C")
            if rh is not None:  meta.append(f"RH {rh:.0f}%")
            if meta: st.caption(" · ".join(meta))

    # right column → mini chart of last 48 h
    with colR:
        raw = load_raw_data(BASE)
        last48 = raw[["pm25","no2"]].last("48H").rename(columns={"pm25":"PM₂.₅","no2":"NO₂"})
        fig = px.line(last48, labels={"value":"Conc.", "index":"time"})
        fig.update_layout(title="Last 48 h observations", height=300, margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Latest Stored AQI Forecast (from CSV)")
        st.table(forecast_df)


# ==================== HISTORY TAB =============================================
with tab_history:
    st.header("Model performance & recent forecasts")

    st.subheader("RMSE by pollutant & horizon")
    st.dataframe(metrics_df, height=200)

    # bar chart of RMSE
    fig_rmse = px.bar(metrics_df, x="horizon", y="rf_rmse",
                      color="pollutant", barmode="group",
                      title="Random‑Forest RMSE (µg/m³ or ppb)")
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Latest quick AQI forecast (training script)")
    st.dataframe(forecast_df, height=180)

    # show last 7 d AQI
    raw = load_raw_data(BASE)
    daily = raw[["pm25","no2"]].resample("1H").mean()
    daily["AQI"] = [aqi_idx(r.pm25, r.no2) for r in daily.itertuples()]
    weekly = daily.last("7D")
    fig_week = px.area(weekly, y="AQI", title="Hourly AQI – last 7 days")
    fig_week.add_hline(y=100, line_dash="dot", line_color="red")
    st.plotly_chart(fig_week, use_container_width=True)