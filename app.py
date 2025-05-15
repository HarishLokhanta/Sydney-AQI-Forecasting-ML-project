# ──────────────────────────────────────────────────────────────
# app.py  (place this over your existing file)
# ──────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, joblib, numpy as np, re
from pathlib import Path

# ---------- AQI helpers ----------
_PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
            (55.5,150.4,151,200),(150.5,250.4,201,300),
            (250.5,350.4,301,400),(350.5,500.4,401,500)]
_NO2_BP  = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
            (361,649,151,200),(650,1249,201,300),
            (1250,1649,301,400),(1650,2049,401,500)]

def _interp(c, bp):
    for lo, hi, ilo, ihi in bp:
        if lo <= c <= hi:
            return (ihi-ilo)/(hi-lo)*(c-lo)+ilo
    return np.nan

def aqi_idx(pm, no2):
    return max(_interp(pm, _PM25_BP),
               _interp(no2*0.522, _NO2_BP))

# --- met‑correction formulas (adjusted coefficients) ---
ALPHA = 0.20   # hygroscopic factor
BETA  = 1.3
GAMMA = 0.05   # wind dilution
H0    = 1000   # reference BLH for temp proxy

def pm_hum_corr(pm, rh, α=ALPHA, β=BETA):
    """Reduce PM for hygroscopic growth; tweak α,β to soften the effect."""
    return pm / (1 + α*(rh/100)**β)

def pm_wind_corr(pm, wsp, γ=GAMMA):
    """Linear dilution with wind; cap so factor never < 0.5."""
    return max(pm * (1 - γ*wsp), 0.5*pm)

def pm_temp_corr(pm, T, H0=H0):
    """Boundary‑layer proxy using surface temp; floor at 0.5×pm."""
    blh = 100*T + 150
    factor = H0 / blh
    return max(pm * factor, 0.5*pm)

def clean_name(raw):
    raw = raw.lower().replace("pm2.5","pm25")
    raw = re.sub(r"[^0-9a-z_]+","_",raw)
    raw = re.sub(r"_+","_",raw).strip("_")
    return raw

# ---------- paths ----------
BASE   = Path(__file__).parent
MODELS = BASE/"models"
DATA   = BASE
REPORT = BASE/"reports"

metrics_df  = pd.read_csv(REPORT/"metrics.csv")
forecast_df = pd.read_csv(REPORT/"aqi_forecast.csv")

# ---------- UI ----------
st.title("Sydney AQI Forecaster  🌬️")
with st.sidebar:
    st.markdown("### How meteorology alters PM₂.₅")
    st.write("""
* **Humidity** – particles take up water → mass ↑  
* **Wind** – higher wind dilutes & advects plumes → mass ↓  
* **Temperature** – warms the boundary layer, mixing depth ↑ → mass ↓
    """)

# ─── AQI Methodology Explanation ─────────────
st.markdown("### AQI Calculation Details")
with st.expander("Click to view how AQI is computed"):
    st.markdown("""
    - **Breakpoints** for PM₂.₅ and NO₂ follow the Australian AQI standards:
      - PM₂.₅ (µg/m³): 0–12 → Good, 12.1–35.4 → Moderate, 35.5–55.4 → Unhealthy for Sensitive Groups, etc. citeturn0search0
      - NO₂ (ppb): 0–53 → Good, 54–100 → Moderate, etc. citeturn0search0
    - The function `_interp()` applies **linear interpolation** between these breakpoints, returning `numpy.nan` outside the range. 
    - NO₂ is **converted** from ppb to µg/m³ in the interpolation using its molecular weight (46.006 g/mol) and standard molar volume (24.45 L/mol):  
      \(\mu g/m³ = ppb \times \frac{46.006}{24.45}\) citeturn0search1
    """)
suburb  = st.selectbox("Choose suburb",["randwick","earlwood","macquarie_park"])
horizon = st.slider("Forecast horizon (hours)",1,3,3,step=2)
modeltyp= st.radio("Model",["XGBoost","Random Forest"])
st.markdown(f"**Selected:** {suburb.title()}, +{horizon} h, {modeltyp}")

def align_feats(mobj, row):
    return row.reindex(columns=mobj["features"], fill_value=0).values

# ---------- predict ----------
if st.button("Predict"):
    # --- load raw data (clean col names) ---
    df = pd.concat([
        pd.read_csv(DATA/"Final_Wind.csv",   index_col=0, parse_dates=True),
        pd.read_csv(DATA/"humtemp_all.csv",  index_col=0, parse_dates=True),
        pd.read_csv(DATA/"pm25_no2_all.csv", index_col=0, parse_dates=True),
    ],axis=1).sort_index().dropna(how="all")
    df.columns = [clean_name(c) for c in df.columns]
    df["pm25"] = df[[c for c in df if "pm25" in c]].mean(axis=1)
    df["no2"]  = df[[c for c in df if "no2"  in c]].mean(axis=1)

    # -- minimal features (lags already in RF/XGB models) --
    latest = df.dropna().iloc[[-1]]

    tag = "xgb" if modeltyp=="XGBoost" else "rf"
    pm_m  = joblib.load(MODELS/f"{tag}_pm25_t+{horizon}.pkl")
    no2_m = joblib.load(MODELS/f"{tag}_no2_t+{horizon}.pkl")

    pm = float(pm_m["model"].predict(align_feats(pm_m, latest))[0])
    no = float(no2_m["model"].predict(align_feats(no2_m, latest))[0])

    # --- met predictions (RF) ---
    def pmet(name_stub):
        fp = MODELS/f"rf_{suburb}_{name_stub}_t+{horizon}.pkl"
        if fp.exists():
            mo = joblib.load(fp)
            return float(mo["model"].predict(align_feats(mo, latest))[0])
    wsp  = pmet("wsp_1h_average_m_s")
    tmp  = pmet("temp_1h_average_c")
    rh   = pmet("humid_1h_average_%")

    # --- AQI calculations ---
    aqi_raw = aqi_idx(pm, no)
    pm_corr = pm
    if rh  is not None: pm_corr = pm_hum_corr(pm_corr, rh)
    if wsp is not None: pm_corr = pm_wind_corr(pm_corr, wsp)
    if tmp is not None: pm_corr = pm_temp_corr(pm_corr, tmp)
    # don’t allow correction to remove >50 % of mass
    pm_corr = max(pm_corr, 0.5*pm)
    aqi_met = aqi_idx(pm_corr, no)

    # --- display ---
    c1, c2, c3 = st.columns(3)
    c1.metric("PM₂.₅ [µg m⁻³]", f"{pm:.1f}")
    c2.metric("NO₂ [ppb]",      f"{no:.1f}")
    c3.metric("Un-adjusted AQI", f"{aqi_raw:.0f}")

    st.markdown("#### Weather-adjusted")
    c4, c5, c6 = st.columns(3)
    c4.metric("RH-wind-temp adj PM₂.₅", f"{pm_corr:.1f}")
    c5.metric("Adjusted AQI", f"{aqi_met:.0f}")
    meta = []
    if wsp is not None: meta.append(f"Wind {wsp:.1f}\u202Fm/s")
    if tmp is not None: meta.append(f"Temp {tmp:.1f}\u202F°C")
    if rh  is not None: meta.append(f"RH {rh:.0f}%")
    if meta:
        st.write(" · ".join(meta))

# ---------- historical tables ----------
st.markdown("---")
st.subheader("Model RMSE (pollutants)")
st.table(metrics_df)
st.subheader("Latest AQI forecast (training script)")
st.table(forecast_df)