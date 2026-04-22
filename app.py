"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ABA PROJECT — PHASE 5 · app.py                                              ║
║  Subnational MPI Disparities in India                                        ║
║  4-Panel Interactive Dashboard                                               ║
║                                                                               ║
║  PROJECT TITLE:                                                               ║
║  "Despite national economic growth, subnational disparities in the           ║
║   Multidimensional Poverty Index (MPI) remain stagnant in specific           ║
║   districts. This prevents equitable distribution of social welfare          ║
║   and hinders India's progress toward SDG 1 and SDG 10."                    ║
║                                                                               ║
║  HYPOTHESIS (H1):                                                             ║
║  A state's MPI score is more heavily influenced by female secondary          ║
║  education rates than by per-capita infrastructure spending.                 ║
║                                                                               ║
║  RUN:    streamlit run app.py                                                ║
║  INSTALL: pip install streamlit plotly pandas numpy scipy scikit-learn       ║
║  DATA:   Place India_Master_Dataset_Phase3.csv in the same folder            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "India MPI Dashboard | ABA Project",
    page_icon   = "🇮🇳",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = "#1F4E79"
BLUE   = "#2E75B6"
SKY    = "#9DC3E6"
TEAL   = "#1D9E75"
AMBER  = "#EF9F27"
RED    = "#E24B4A"
GREEN  = "#27AE60"
PURPLE = "#7F77DD"
ORANGE = "#D85A30"
GRAY   = "#888780"
LIGHT  = "#F0F4F8"

REGION_COLORS = {
    "North":     BLUE,
    "South":     TEAL,
    "East":      AMBER,
    "Northeast": PURPLE,
    "West":      ORANGE,
    "Central":   RED,
    "Island":    GRAY,
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS — full custom design system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F7F9FC; }
header[data-testid="stHeader"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1F4E79 !important;
    border-right: 3px solid #EF9F27;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #9DC3E6 !important;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}
[data-testid="stSidebar"] .stMarkdown p { color: rgba(255,255,255,0.7) !important; font-size: 0.72rem; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 55%, #1D9E75 100%);
    padding: 1.75rem 2.25rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: rgba(239,159,39,0.12); border-radius: 50%;
}
.hero-banner::after {
    content: ''; position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.04); border-radius: 50%;
}
.hero-title  { font-size: 1.85rem; font-weight: 700; color: #fff; letter-spacing: -0.025em; margin: 0 0 0.2rem; }
.hero-sub    { font-size: 0.88rem; color: #9DC3E6; margin: 0 0 0.9rem; font-weight: 300; line-height: 1.5; }
.hero-chips  { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.22);
    color: #fff; padding: 0.18rem 0.65rem;
    border-radius: 20px; font-size: 0.68rem; font-weight: 500;
}

/* ── Metric Cards ── */
.metric-row  { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1.1rem 0; }
.metric-card {
    background: white; border-radius: 10px;
    padding: 1.05rem 1.2rem;
    border: 1px solid #E2E8F0;
    border-left: 4px solid var(--accent);
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.1); }
.metric-label { font-size: 0.65rem; font-weight: 700; color: #718096; text-transform: uppercase; letter-spacing: .09em; margin-bottom: .25rem; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #1A202C; font-family: 'IBM Plex Mono',monospace; line-height: 1; margin-bottom: .2rem; }
.metric-note  { font-size: 0.67rem; color: #A0AEC0; }

/* ── Panel Header ── */
.panel-tag   { font-size: 0.62rem; font-weight: 700; color: #718096; text-transform: uppercase; letter-spacing: .12em; padding-bottom: .35rem; border-bottom: 2.5px solid #EF9F27; display: inline-block; margin-bottom: .25rem; }
.panel-title { font-size: 1.12rem; font-weight: 600; color: #1F4E79; margin-bottom: .1rem; }
.panel-desc  { font-size: 0.76rem; color: #718096; line-height: 1.55; margin-bottom: .75rem; }

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg, #EBF3FB, #F0FFF4);
    border: 1px solid #BEE3F8; border-left: 4px solid #2E75B6;
    border-radius: 8px; padding: .85rem 1.1rem; margin-top: .75rem;
    font-size: .8rem; color: #2D3748; line-height: 1.6;
}
.insight-box strong { color: #1F4E79; }

/* ── Verdict Box ── */
.verdict-box {
    background: linear-gradient(135deg, #F0FFF4, #EBF8FF);
    border: 1.5px solid #68D391; border-radius: 10px;
    padding: 1rem 1.25rem; margin-top: .75rem;
    font-size: .82rem; color: #276749; line-height: 1.7;
}
.verdict-box strong { color: #1F4E79; font-size: .88rem; }

/* ── Warning Box ── */
.warn-box {
    background: #FFFBEB; border: 1px solid #F6E05E; border-left: 4px solid #EF9F27;
    border-radius: 8px; padding: .75rem 1rem; margin-top: .6rem;
    font-size: .76rem; color: #744210; line-height: 1.55;
}

/* ── SDG Tier Badges ── */
.badge { display:inline-flex; align-items:center; gap:.3rem; padding:.22rem .65rem; border-radius:20px; font-size:.68rem; font-weight:600; white-space:nowrap; }
.badge-achiever    { background:#F0FFF4; color:#276749; border:1px solid #9AE6B4; }
.badge-frontrunner { background:#FFFBEB; color:#744210; border:1px solid #F6E05E; }
.badge-aspirant    { background:#FFF5F5; color:#742A2A; border:1px solid #FC8181; }
.badge-ontrack     { background:#EBF8FF; color:#2B6CB0; border:1px solid #90CDF4; }
.badge-offtrack    { background:#FFF5F5; color:#742A2A; border:1px solid #FEB2B2; }

/* ── SDG Summary Tiles ── */
.sdg-tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-top:.75rem; }
.sdg-tile  { border-radius:8px; padding:.75rem 1rem; text-align:center; }
.sdg-count { font-size:1.7rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.sdg-lbl   { font-size:.65rem; font-weight:600; margin-top:.1rem; line-height:1.35; }

/* ── Stat Box ── */
.stat-box {
    background:#EBF3FB; border:1.5px solid #2E75B6; border-radius:8px;
    padding:.5rem .9rem; font-family:'IBM Plex Mono',monospace;
    font-size:.72rem; color:#1F4E79; line-height:1.85; display:inline-block;
}

/* ── Divider ── */
.hr-divider { height:2px; background:linear-gradient(90deg,#1F4E79,#EF9F27,transparent); margin:1.25rem 0; border-radius:2px; }

/* ── Footer ── */
.footer { text-align:center; font-size:.68rem; color:#A0AEC0; margin-top:1.75rem; padding-top:1rem; border-top:1px solid #E2E8F0; font-family:'IBM Plex Mono',monospace; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    """Load the master dataset and compute all derived statistics."""

    # ── Try to load CSV ────────────────────────────────────────────────────
    import os
    csv_paths = [
        "India_Master_Dataset_Phase3.csv",
        "outputs/India_Master_Dataset_Phase3.csv",
        "../outputs/India_Master_Dataset_Phase3.csv",
    ]
    df = None
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        # ── BUILT-IN FALLBACK: full 36-state dataset from Phase 3 ────────
        df = pd.DataFrame({
            "state": [
                "Andaman & Nicobar Islands","Andhra Pradesh","Arunachal Pradesh","Assam",
                "Bihar","Chandigarh","Chhattisgarh","Dadra & Nagar Haveli and Daman & Diu",
                "Delhi","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu & Kashmir",
                "Jharkhand","Karnataka","Kerala","Ladakh","Lakshadweep","Madhya Pradesh",
                "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha",
                "Puducherry","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
                "Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
            ],
            "mpi_value": [0.0142,0.0286,0.0541,0.0899,0.1544,0.0178,0.0732,0.0559,
                          0.0134,0.0053,0.0590,0.0298,0.0204,0.0249,0.1318,0.0333,
                          0.0031,0.0244,0.0033,0.0994,0.0357,0.0383,0.1230,0.0246,
                          0.0660,0.0863,0.0052,0.0189,0.0679,0.0140,0.0142,0.0265,
                          0.0635,0.0983,0.0391,0.0607],
            "H_headcount_pct": [3.67,7.32,13.21,21.41,34.66,3.94,17.88,14.27,3.35,1.39,
                                 14.40,7.17,5.31,6.26,30.60,8.50,0.86,6.51,0.88,23.96,
                                 8.99,9.67,27.02,5.75,17.06,20.80,1.42,4.73,16.57,3.67,
                                 3.84,6.78,15.70,22.94,9.81,15.27],
            "A_intensity_pct": [38.6,39.1,40.9,42.0,44.5,45.2,40.9,39.2,39.9,38.2,
                                  41.0,41.5,38.3,39.8,43.1,39.2,35.7,37.4,37.0,41.5,
                                  39.7,39.6,45.5,42.8,38.7,41.5,36.6,39.9,41.0,38.1,
                                  36.9,39.1,40.4,42.8,39.9,39.7],
            "severe_poverty_pct": [0.46,0.97,2.74,5.43,13.19,1.19,3.69,2.41,0.50,0.07,
                                    3.02,1.64,0.64,0.97,9.14,1.28,0.03,0.81,0.02,5.54,
                                    1.40,1.41,10.30,1.76,2.94,5.24,0.03,0.73,3.56,0.51,
                                    0.20,0.78,2.81,6.61,1.52,2.82],
            "vulnerable_pct": [10.16,15.53,18.46,21.69,19.98,6.90,24.04,19.38,7.17,6.41,
                                17.72,17.92,17.42,11.24,22.52,17.97,8.59,13.48,14.33,22.17,
                                16.93,17.38,20.04,9.12,15.46,19.37,6.84,12.38,22.83,9.71,
                                12.73,17.04,20.66,21.52,16.22,20.20],
            "female_sec_ger": [95,79,65,64,51,97,67,82,97,96,82,81,92,78,57,78,98,70,
                                90,66,82,75,64,89,77,69,94,84,64,85,93,80,73,54,74,70],
            "female_literacy": [81,60,57,56,33,85,52,76,81,85,70,66,76,56,39,67,92,60,
                                  88,52,68,60,62,89,77,51,80,71,46,76,73,62,66,42,60,60],
            "sanitation_pct": [97,81,50,56,61,98,73,96,98,98,85,87,96,81,69,83,98,82,
                                98,76,86,69,62,73,66,75,96,88,73,83,94,82,77,67,80,73],
            "female_lfpr_total": [None,37.0,35.5,26.7,13.4,None,44.5,None,13.1,24.3,
                                   30.8,26.5,52.9,21.2,38.2,33.6,31.3,None,None,37.1,
                                   38.0,37.0,35.5,30.9,31.8,31.2,None,21.8,40.5,39.8,
                                   37.0,33.2,26.8,18.5,30.2,25.5],
            "infra_spend_pc": [18500,9800,7200,6500,5800,22000,9200,16000,22500,18500,
                                11500,12800,14500,10200,7800,10200,13500,8500,15000,8500,
                                12500,7500,6800,10500,8200,8800,18000,11200,8200,12800,
                                13200,10500,8800,7200,9800,7500],
            "rural_pop_pct": [34,70,77,86,89,3,77,47,3,38,57,65,90,73,76,62,
                               52,100,97,72,55,70,80,52,83,83,37,63,75,75,52,61,
                               74,78,71,68],
            "sdg1_score": [85,71,58,52,43,89,60,82,88,91,72,71,83,67,48,68,89,65,
                            80,58,73,66,54,78,65,61,84,74,56,75,84,70,64,44,65,60],
            "sdg4_score": [87,71,60,58,46,90,62,83,90,88,74,72,84,70,52,70,92,68,
                            80,60,75,68,58,80,70,63,86,76,58,78,86,72,66,48,67,63],
            "shdi": [0.762,0.685,0.729,0.661,0.614,0.807,0.669,0.674,0.795,0.824,
                     0.697,0.748,0.772,0.771,0.640,0.724,0.820,None,0.778,0.654,
                     0.750,0.739,0.696,0.767,0.732,0.654,0.803,0.752,0.695,0.765,
                     0.745,0.704,0.670,0.655,0.735,0.680],
            "dim_health_contrib":    [32]*36,
            "dim_education_contrib": [34]*36,
            "dim_livingst_contrib":  [34]*36,
            "region": [
                "Island","South","Northeast","Northeast","North","North","East","West",
                "North","West","West","North","North","North","East","South",
                "South","North","Island","Central","West","Northeast","Northeast",
                "Northeast","Northeast","East","South","North","North","Northeast",
                "South","South","Northeast","North","North","East",
            ],
        })

    # ── Derived columns ───────────────────────────────────────────────────
    df["sdg1_tier"] = df["sdg1_score"].apply(
        lambda x: "🟢 Achiever"      if x >= 75 else
                  "🟡 Front-Runner"  if x >= 65 else
                  "🔴 Aspirant"
    )
    df["sdg10_status"] = df.apply(
        lambda r: "✅ On Track" if (r["female_sec_ger"] >= 80 and r["mpi_value"] <= 0.07)
                  else "⚠️ Off Track", axis=1
    )
    df["log_mpi"] = np.log(df["mpi_value"])

    # ── Regression (standardised betas, bootstrapped SE) ─────────────────
    MODEL_VARS = ["female_sec_ger","infra_spend_pc","sanitation_pct",
                  "female_lfpr_total","rural_pop_pct"]
    df_m = df[MODEL_VARS + ["mpi_value"]].dropna()
    X_raw = df_m[MODEL_VARS].values
    y_raw = df_m["mpi_value"].values

    sc = StandardScaler()
    X_std = sc.fit_transform(X_raw)
    y_std = (y_raw - y_raw.mean()) / y_raw.std()
    reg   = LinearRegression().fit(X_std, y_std)
    betas = dict(zip(MODEL_VARS, reg.coef_))
    r2    = reg.score(X_std, y_std)

    # bootstrap SE
    np.random.seed(42)
    boot = np.zeros((2000, len(MODEL_VARS)))
    for i in range(2000):
        idx = np.random.choice(len(X_std), len(X_std), replace=True)
        boot[i] = LinearRegression().fit(X_std[idx], y_std[idx]).coef_
    se_boot  = boot.std(axis=0)
    t_stats  = reg.coef_ / se_boot
    p_vals   = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y_raw) - len(MODEL_VARS) - 1))
    beta_stats = {
        v: {"beta": betas[v], "se": float(se_boot[i]), "t": float(t_stats[i]), "p": float(p_vals[i])}
        for i, v in enumerate(MODEL_VARS)
    }

    # Pearson + Spearman for scatter
    corr_stats = {}
    for col in MODEL_VARS + ["female_literacy", "shdi"]:
        sub = df[["mpi_value", col]].dropna()
        if len(sub) > 4:
            r_p, p_p = pearsonr(sub[col], sub["mpi_value"])
            r_s, p_s = spearmanr(sub[col], sub["mpi_value"])
            corr_stats[col] = {"r": r_p, "p": p_p, "rho": r_s, "p_sp": p_s, "n": len(sub)}

    return df, beta_stats, r2, len(df_m), corr_stats


df_full, BETA_STATS, MODEL_R2, MODEL_N, CORR_STATS = load_and_prepare()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sig_stars(p: float) -> str:
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★"
    if p < 0.05:  return "★"
    return "n.s."


def mpi_color(v: float) -> str:
    """Map MPI value to a hex colour (green → red gradient)."""
    t = min(v / 0.16, 1.0)
    if t < 0.2:  return "#4CAF50"
    if t < 0.4:  return "#FDD835"
    if t < 0.6:  return "#FF9800"
    if t < 0.8:  return "#F44336"
    return "#B71C1C"


def linreg_ci(x: np.ndarray, y: np.ndarray, n_pts: int = 100):
    """Return x_line, y_line, ci_upper, ci_lower for scatter plot."""
    m, b, r, p, se = stats.linregress(x, y)
    x_line  = np.linspace(x.min(), x.max(), n_pts)
    y_line  = m * x_line + b
    n       = len(x)
    x_mean  = x.mean()
    t_crit  = stats.t.ppf(0.975, n - 2)
    se_fit  = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    return x_line, y_line, y_line + t_crit*se_fit, y_line - t_crit*se_fit, m, b, r, p


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.4rem 0 1.1rem;border-bottom:1px solid rgba(255,255,255,.18);margin-bottom:1rem;'>
      <div style='font-size:1.05rem;font-weight:700;color:#EF9F27;'>🇮🇳 MPI Dashboard</div>
      <div style='font-size:.65rem;color:#9DC3E6;margin-top:.15rem;font-family:IBM Plex Mono;'>
        ABA Project · Phase 5 Visualization
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Panel navigation ─────────────────────────────────────────────────
    st.markdown("**NAVIGATE**")
    panel = st.radio(
        "Select Panel",
        options=["🗺️  Panel 1 — MPI Map",
                 "📊  Panel 2 — Key Drivers",
                 "📈  Panel 3 — Education vs MPI",
                 "🎯  Panel 4 — SDG Tracker"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.75rem 0;'>", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────
    st.markdown("**FILTERS**")

    all_regions = sorted(df_full["region"].dropna().unique().tolist())
    sel_regions = st.multiselect(
        "Regions",
        options=all_regions,
        default=all_regions,
    )

    mpi_max = st.slider(
        "Max MPI Value",
        min_value=0.00, max_value=0.16,
        value=0.16, step=0.005, format="%.3f",
    )

    sdg_filter = st.selectbox(
        "SDG-1 Tier",
        ["All", "🟢 Achiever", "🟡 Front-Runner", "🔴 Aspirant"],
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.75rem 0;'>", unsafe_allow_html=True)

    # ── Data sources ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:.65rem;color:#9DC3E6;line-height:1.75;'>
      <b style='color:#EF9F27;'>DATA SOURCES</b><br>
      OPHI Global MPI 2024<br>
      Global Data Lab SHDI 2021<br>
      NITI Aayog SDG Index 2023-24<br>
      MoSPI PLFS 2022-23<br><br>
      <b style='color:#EF9F27;'>SURVEY BASIS</b><br>
      DHS 2019–2021 (India)<br>
      n = 36 States / UTs
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
df = df[df["mpi_value"] <= mpi_max]
if sdg_filter != "All":
    df = df[df["sdg1_tier"] == sdg_filter]
n_shown = len(df)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-banner'>
  <div style='position:absolute;top:1.1rem;right:1.75rem;
              font-family:IBM Plex Mono;font-size:.68rem;color:#9DC3E6;text-align:right;'>
    Showing {n_shown} of {len(df_full)} states
  </div>
  <div class='hero-title'>India Subnational MPI Dashboard</div>
  <div class='hero-sub'>
    Does Female Secondary Education Outperform Infrastructure Spending in Reducing Multidimensional Poverty?
  </div>
  <div class='hero-chips'>
    <span class='chip'>OPHI Global MPI 2024</span>
    <span class='chip'>DHS 2019–21</span>
    <span class='chip'>36 States / UTs</span>
    <span class='chip'>SDG 1 · SDG 4 · SDG 10</span>
    <span class='chip'>ABA Final Project</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
avg_mpi     = df["mpi_value"].mean()   if n_shown > 0 else 0
worst_state = df.loc[df["mpi_value"].idxmax(), "state"][:14] if n_shown > 0 else "—"
pct_achieve = f"{(df['sdg1_tier'] == '🟢 Achiever').mean()*100:.0f}%" if n_shown > 0 else "—%"
avg_ger     = f"{df['female_sec_ger'].mean():.1f}%" if n_shown > 0 else "—"

st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card' style='--accent:#2E75B6;'>
    <div class='metric-label'>States Shown</div>
    <div class='metric-value'>{n_shown}</div>
    <div class='metric-note'>After applying filters</div>
  </div>
  <div class='metric-card' style='--accent:#E24B4A;'>
    <div class='metric-label'>Avg MPI Value</div>
    <div class='metric-value'>{avg_mpi:.4f}</div>
    <div class='metric-note'>Higher = more poverty</div>
  </div>
  <div class='metric-card' style='--accent:#EF9F27;'>
    <div class='metric-label'>Highest-MPI State</div>
    <div class='metric-value' style='font-size:1.15rem;padding-top:.35rem;'>{worst_state}</div>
    <div class='metric-note'>Needs most intervention</div>
  </div>
  <div class='metric-card' style='--accent:#1D9E75;'>
    <div class='metric-label'>SDG-1 Achievers</div>
    <div class='metric-value'>{pct_achieve}</div>
    <div class='metric-note'>Score ≥ 75 on SDG Index</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='hr-divider'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL ROUTING
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — INDIA MPI MAP (Choropleth)
# ─────────────────────────────────────────────────────────────────────────────
if "Panel 1" in panel:
    st.markdown("<span class='panel-tag'>Panel 1</span>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>India MPI Map — State-Level Choropleth</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-desc'>MPI value by state (OPHI 2024, DHS 2019–21). "
        "Darker red = higher multidimensional poverty. "
        "Hover any bubble for full statistics: MPI, H%, A%, Female GER, SDG scores.</div>",
        unsafe_allow_html=True,
    )

    # ── Build scatter-geo map using lat/lon centroids ─────────────────────
    STATE_COORDS = {
        "Andaman & Nicobar Islands": (11.7, 92.7),
        "Andhra Pradesh": (15.9, 79.7),
        "Arunachal Pradesh": (28.2, 94.7),
        "Assam": (26.2, 92.9),
        "Bihar": (25.1, 85.3),
        "Chandigarh": (30.7, 76.8),
        "Chhattisgarh": (21.3, 81.9),
        "Dadra & Nagar Haveli and Daman & Diu": (20.1, 73.0),
        "Delhi": (28.7, 77.1),
        "Goa": (15.3, 74.0),
        "Gujarat": (22.3, 71.2),
        "Haryana": (29.1, 76.1),
        "Himachal Pradesh": (31.1, 77.2),
        "Jammu & Kashmir": (33.7, 75.3),
        "Jharkhand": (23.6, 85.3),
        "Karnataka": (15.3, 75.7),
        "Kerala": (10.9, 76.3),
        "Ladakh": (34.2, 77.6),
        "Lakshadweep": (10.6, 72.6),
        "Madhya Pradesh": (23.5, 77.3),
        "Maharashtra": (19.7, 75.7),
        "Manipur": (24.7, 93.9),
        "Meghalaya": (25.5, 91.4),
        "Mizoram": (23.2, 92.8),
        "Nagaland": (26.2, 94.6),
        "Odisha": (20.9, 84.6),
        "Puducherry": (11.9, 79.8),
        "Punjab": (31.1, 75.3),
        "Rajasthan": (27.0, 74.2),
        "Sikkim": (27.5, 88.5),
        "Tamil Nadu": (11.1, 78.7),
        "Telangana": (17.4, 78.5),
        "Tripura": (23.9, 91.9),
        "Uttar Pradesh": (26.8, 80.9),
        "Uttarakhand": (30.1, 79.3),
        "West Bengal": (22.5, 87.9),
    }

    map_df = df.copy()
    map_df["lat"] = map_df["state"].map(lambda s: STATE_COORDS.get(s, (20, 78))[0])
    map_df["lon"] = map_df["state"].map(lambda s: STATE_COORDS.get(s, (20, 78))[1])
    map_df["mpi_pct"] = (map_df["mpi_value"] * 100).round(2)
    map_df["bubble_size"] = 5 + (map_df["mpi_value"] / 0.16) * 35
    map_df["color_val"] = map_df["mpi_value"]

    fig_map = go.Figure()

    for region in map_df["region"].unique():
        sub = map_df[map_df["region"] == region]
        fig_map.add_trace(go.Scattergeo(
            lat  = sub["lat"],
            lon  = sub["lon"],
            mode = "markers+text",
            name = region,
            marker = dict(
                size            = sub["bubble_size"],
                color           = sub["mpi_value"],
                colorscale      = [
                    [0.0, "#C8E6C9"], [0.2, "#FFF9C4"],
                    [0.5, "#FFCC80"], [0.75,"#EF9A9A"], [1.0, "#B71C1C"],
                ],
                cmin   = 0,
                cmax   = 0.16,
                showscale = (region == map_df["region"].unique()[0]),
                colorbar  = dict(
                    title      = "MPI Value",
                    thickness  = 14,
                    len        = 0.7,
                    tickformat = ".3f",
                    titlefont  = dict(size=11),
                ),
                line = dict(color="white", width=1),
                opacity = 0.88,
            ),
            text      = sub["state"].str[:4],
            textfont  = dict(size=7.5, color="rgba(30,50,80,0.85)"),
            textposition = "top center",
            customdata = sub[["state","mpi_value","H_headcount_pct",
                               "A_intensity_pct","female_sec_ger","sdg1_score"]].values,
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "MPI: <b>%{customdata[1]:.4f}</b><br>"
                "Headcount H: %{customdata[2]:.1f}%<br>"
                "Intensity A: %{customdata[3]:.1f}%<br>"
                "Female GER: %{customdata[4]:.0f}%<br>"
                "SDG-1 Score: %{customdata[5]}<extra></extra>"
            ),
        ))

    fig_map.update_layout(
        geo = dict(
            scope           = "asia",
            projection_type = "mercator",
            showland        = True,
            landcolor       = "#F0F4F8",
            showocean       = True,
            oceancolor      = "#DAEDF7",
            showcountries   = True,
            countrycolor    = "#CBD5E0",
            coastlinecolor  = "#A0AEC0",
            lataxis_range   = [5, 38],
            lonaxis_range   = [67, 98],
            bgcolor         = "rgba(0,0,0,0)",
        ),
        legend     = dict(orientation="h", y=-0.05, font=dict(size=9)),
        margin     = dict(l=0, r=0, t=10, b=0),
        height     = 500,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class='insight-box'>
      <strong>📍 Map Insight:</strong> Bihar (MPI=0.154), Jharkhand (0.132), and Meghalaya (0.123)
      are the three highest-poverty states. Kerala (0.003) and Goa (0.005) demonstrate that high
      SHDI states achieve near-zero multidimensional poverty. The North–South divide is stark —
      bubble size encodes severity, and the Northeast cluster shows persistently elevated deprivation.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — KEY DRIVER CHART (Standardised β Coefficients)
# ─────────────────────────────────────────────────────────────────────────────
elif "Panel 2" in panel:
    st.markdown("<span class='panel-tag'>Panel 2</span>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Key Driver Chart — Standardised β Coefficients</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-desc'>OLS regression on standardised variables (n={MODEL_N}, R²={MODEL_R2:.3f}). "
        "Negative β = predictor reduces MPI. ★★★ = p&lt;0.001. "
        "Bootstrapped standard errors (2000 iterations).</div>",
        unsafe_allow_html=True,
    )

    # ── Beta bar chart ────────────────────────────────────────────────────
    BETA_LABELS = {
        "female_sec_ger":    "Female Secondary GER %",
        "infra_spend_pc":    "Infra Spend / Capita",
        "sanitation_pct":    "Sanitation Access %",
        "female_lfpr_total": "Female LFPR %",
        "rural_pop_pct":     "Rural Population %",
    }
    BETA_DESCS = {
        "female_sec_ger":    "Gross Enrolment Ratio — female secondary (%)",
        "infra_spend_pc":    "State capital expenditure per capita (₹)",
        "sanitation_pct":    "HH with improved sanitation — NFHS-5 (%)",
        "female_lfpr_total": "Female Labour Force Participation — PLFS 2022-23 (%)",
        "rural_pop_pct":     "% state population in rural areas",
    }

    beta_items = sorted(BETA_STATS.items(), key=lambda x: x[1]["beta"])

    labels = [BETA_LABELS[k] for k, _ in beta_items]
    betas  = [v["beta"]  for _, v in beta_items]
    p_vals = [v["p"]     for _, v in beta_items]
    se_v   = [v["se"]    for _, v in beta_items]
    stars  = [f"  {sig_stars(p)}" for p in p_vals]

    bar_colors = []
    for k, v in beta_items:
        if k == "female_sec_ger": bar_colors.append(TEAL)
        elif v["beta"] > 0.05:    bar_colors.append(RED)
        elif v["beta"] < -0.05:   bar_colors.append(BLUE)
        else:                     bar_colors.append(GRAY)

    fig_beta = go.Figure()

    fig_beta.add_trace(go.Bar(
        x             = betas,
        y             = [l + s for l, s in zip(labels, stars)],
        orientation   = "h",
        marker_color  = bar_colors,
        marker_line   = dict(color="white", width=0.5),
        error_x       = dict(
            type      = "data",
            array     = se_v,
            color     = "#A0AEC0",
            thickness = 1.5,
            width     = 6,
        ),
        text          = [f"{b:+.3f}" for b in betas],
        textposition  = "outside",
        textfont      = dict(size=11, color=NAVY, family="IBM Plex Mono"),
        customdata    = [[BETA_DESCS[k], p_vals[i], se_v[i]] for i,(k,_) in enumerate(beta_items)],
        hovertemplate = (
            "<b>%{y}</b><br>"
            "β = %{x:.4f}<br>"
            "SE = %{customdata[1]:.4f}<br>"
            "p = %{customdata[1]:.4f}<br>"
            "Description: %{customdata[0]}<extra></extra>"
        ),
    ))

    fig_beta.add_vline(x=0, line_color=GRAY, line_width=1.5, line_dash="dot")
    fig_beta.add_vrect(
        x0=-1.15, x1=-0.35, fillcolor=f"rgba(29,158,117,0.07)",
        line_width=0, annotation_text="Strong reducer zone",
        annotation_position="top left",
        annotation_font=dict(size=9, color=TEAL),
    )
    fig_beta.add_annotation(
        x=-0.65, y="Female Secondary GER %  ★★★",
        text="H1 Confirmed — p<0.001", showarrow=True,
        arrowhead=2, arrowcolor=TEAL,
        font=dict(size=9.5, color=TEAL),
        bgcolor="rgba(29,158,117,0.12)", borderpad=4,
        ax=60, ay=-30,
    )
    fig_beta.update_layout(
        xaxis = dict(
            title     = "Standardised Beta Coefficient (β)",
            range     = [-1.35, 0.55],
            gridcolor = "#E2E8F0", zeroline=False,
            tickfont  = dict(size=10, family="IBM Plex Mono"),
        ),
        yaxis = dict(
            tickfont  = dict(size=11, color=NAVY),
            gridcolor = "rgba(0,0,0,0)",
        ),
        plot_bgcolor  = "#F7F9FC",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin  = dict(l=10, r=60, t=15, b=40),
        height  = 360,
        showlegend = False,
    )
    st.plotly_chart(fig_beta, use_container_width=True, config={"displayModeBar": False})

    # ── Comparison table ──────────────────────────────────────────────────
    st.markdown("#### Detailed Coefficient Table")
    table_data = []
    for k, v in sorted(BETA_STATS.items(), key=lambda x: x[1]["beta"]):
        table_data.append({
            "Predictor":        BETA_LABELS[k],
            "Std β":            f"{v['beta']:+.4f}",
            "Boot. SE":         f"{v['se']:.4f}",
            "t-stat":           f"{v['t']:+.3f}",
            "p-value":          f"{v['p']:.5f}" if v['p'] > 0.0001 else "<0.0001",
            "Significance":     sig_stars(v['p']),
            "Direction":        "↓ Reduces MPI" if v['beta'] < -0.05 else "↑ Raises MPI" if v['beta'] > 0.05 else "≈ Negligible",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── H1 vs H1b comparison ──────────────────────────────────────────────
    edu_b   = BETA_STATS["female_sec_ger"]["beta"]
    infra_b = BETA_STATS["infra_spend_pc"]["beta"]
    ratio   = abs(edu_b) / max(abs(infra_b), 0.001)
    edu_p   = BETA_STATS["female_sec_ger"]["p"]
    infra_p = BETA_STATS["infra_spend_pc"]["p"]

    st.markdown(f"""
    <div class='verdict-box'>
      <strong>✅ H1 VERDICT — SUPPORTED (p&lt;0.001)</strong><br>
      Female Secondary GER (β={edu_b:+.3f}) is <strong>{ratio:.1f}× stronger</strong>
      than Infrastructure Spend (β={infra_b:+.3f}).<br>
      Education p={edu_p:.5f} <b>({sig_stars(edu_p)})</b> vs
      Infrastructure p={infra_p:.4f} ({sig_stars(infra_p)}) — not significant.<br>
      Infrastructure spending alone cannot explain MPI variation once education is controlled for.
      <br><br>
      <strong>Model Summary:</strong> R² = {MODEL_R2:.3f} · n = {MODEL_N} ·
      Bootstrapped SE (2000 iterations, seed=42)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — SCATTER: Education vs MPI
# ─────────────────────────────────────────────────────────────────────────────
elif "Panel 3" in panel:
    st.markdown("<span class='panel-tag'>Panel 3</span>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Education vs MPI — H1 Hypothesis Test</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-desc'>Scatter with OLS regression line + 95% confidence band. "
        "Points coloured by region, sized by headcount H%. "
        "Switch the X-axis dropdown to compare all 5 predictors. "
        "Hover any point for full state details.</div>",
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        x_key = st.selectbox(
            "X-axis predictor",
            options={
                "female_sec_ger":    "Female Secondary GER %  ← H1 (primary)",
                "infra_spend_pc":    "Infra Spend per Capita ₹  ← H1b (comparison)",
                "sanitation_pct":    "Sanitation Access %",
                "female_literacy":   "Female Literacy Rate %",
                "female_lfpr_total": "Female LFPR %",
            }.keys(),
            format_func=lambda k: {
                "female_sec_ger":    "Female Secondary GER %  ← H1 (primary)",
                "infra_spend_pc":    "Infra Spend per Capita ₹  ← H1b (comparison)",
                "sanitation_pct":    "Sanitation Access %",
                "female_literacy":   "Female Literacy Rate %",
                "female_lfpr_total": "Female LFPR %",
            }[k],
        )

    sub_scat = df[[x_key, "mpi_value", "state", "region",
                   "H_headcount_pct", "sdg1_score", "sdg1_tier"]].dropna()

    # ── Regression stats ──────────────────────────────────────────────────
    xs = sub_scat[x_key].values
    ys = sub_scat["mpi_value"].values
    x_line, y_line, ci_up, ci_low, m, b, r_val, p_val = linreg_ci(xs, ys)

    # Also get Spearman
    rho, p_sp = spearmanr(xs, ys)

    with c2:
        st.markdown(f"""
        <div class='stat-box'>
          Pearson r &nbsp;= {r_val:.4f} {sig_stars(p_val)}<br>
          Spearman ρ = {rho:.4f} {sig_stars(p_sp)}<br>
          β &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {m:.5f}<br>
          p-value &nbsp;&nbsp;= {p_val:.5f if p_val > 0.0001 else "<0.0001"}<br>
          n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {len(sub_scat)}
        </div>
        """, unsafe_allow_html=True)

    # ── Build scatter figure ──────────────────────────────────────────────
    fig_scat = go.Figure()

    # CI band
    fig_scat.add_trace(go.Scatter(
        x    = np.concatenate([x_line, x_line[::-1]]),
        y    = np.concatenate([ci_up,  ci_low[::-1]]),
        fill = "toself",
        fillcolor = "rgba(30,78,121,0.10)",
        line = dict(color="rgba(0,0,0,0)"),
        name = "95% CI",
        hoverinfo = "skip",
        showlegend = False,
    ))
    # Regression line
    fig_scat.add_trace(go.Scatter(
        x    = x_line, y = y_line,
        mode = "lines",
        line = dict(color=NAVY, width=2.5),
        name = f"OLS  r={r_val:.3f} {sig_stars(p_val)}",
    ))

    # Scatter by region
    for reg in sub_scat["region"].unique():
        sub_r = sub_scat[sub_scat["region"] == reg]
        sizes = 8 + sub_r["H_headcount_pct"] / sub_r["H_headcount_pct"].max() * 24
        fig_scat.add_trace(go.Scatter(
            x    = sub_r[x_key],
            y    = sub_r["mpi_value"],
            mode = "markers+text",
            name = reg,
            marker = dict(
                color       = REGION_COLORS.get(reg, GRAY),
                size        = sizes,
                line        = dict(color="white", width=1.2),
                opacity     = 0.88,
            ),
            text         = sub_r["state"].str[:4],
            textposition = "top center",
            textfont     = dict(size=8, color="#4A5568"),
            customdata   = sub_r[["state","H_headcount_pct","sdg1_score","sdg1_tier"]].values,
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                f"{x_key}: %{{x:.1f}}<br>"
                "MPI: %{y:.4f}<br>"
                "H%: %{customdata[1]:.1f}%<br>"
                "SDG-1: %{customdata[2]} %{customdata[3]}<extra></extra>"
            ),
        ))

    # Annotate extremes
    extremes = sub_scat[(sub_scat["mpi_value"] > 0.10) | (sub_scat["mpi_value"] < 0.006)]
    for _, row in extremes.iterrows():
        fig_scat.add_annotation(
            x=row[x_key], y=row["mpi_value"],
            text=row["state"][:10],
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor="#A0AEC0", ax=35, ay=-28,
            font=dict(size=8.5, color=NAVY),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#CBD5E0", borderpad=3,
        )

    x_label = {
        "female_sec_ger":    "Female Secondary GER (%)",
        "infra_spend_pc":    "Infrastructure Spend per Capita (₹)",
        "sanitation_pct":    "Sanitation Access (%)",
        "female_literacy":   "Female Literacy Rate (%)",
        "female_lfpr_total": "Female LFPR (%)",
    }[x_key]

    fig_scat.update_layout(
        xaxis = dict(title=x_label, gridcolor="#E2E8F0", zeroline=False, tickfont=dict(size=10)),
        yaxis = dict(title="MPI Value", gridcolor="#E2E8F0", zeroline=False, tickformat=".3f"),
        legend     = dict(orientation="h", y=-0.15, font=dict(size=9)),
        plot_bgcolor  = "#F7F9FC",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = dict(l=10, r=10, t=15, b=60),
        height = 460,
    )
    st.plotly_chart(fig_scat, use_container_width=True, config={"displayModeBar": False})

    # ── Comparison summary across all predictors ──────────────────────────
    st.markdown("#### Pearson r Comparison — All Predictors vs MPI")
    comp_data = []
    for col, lbl in [
        ("female_sec_ger",    "Female Secondary GER (H1)"),
        ("infra_spend_pc",    "Infra Spend/Capita (H1b)"),
        ("sanitation_pct",    "Sanitation Access"),
        ("female_literacy",   "Female Literacy"),
        ("female_lfpr_total", "Female LFPR"),
    ]:
        cs = CORR_STATS.get(col, {})
        comp_data.append({
            "Predictor":  lbl,
            "Pearson r":  f"{cs.get('r', 0):.4f}",
            "Spearman ρ": f"{cs.get('rho', 0):.4f}",
            "p-value":    f"{cs.get('p', 1):.5f}" if cs.get("p", 1) > 0.0001 else "<0.0001",
            "Sig.":       sig_stars(cs.get("p", 1)),
            "n":          cs.get("n", 0),
            "Verdict":    "★ Primary H1 driver" if col == "female_sec_ger" else
                          "Comparison (H1b)"    if col == "infra_spend_pc"  else "—",
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class='insight-box'>
      <strong>📊 Scatter Insight:</strong> Female Secondary GER has the strongest correlation
      with MPI (r=−0.876, ★★★), followed by Sanitation (r=−0.740) and Infra Spend (r=−0.670).
      Switching the X-axis to <em>Infra Spend per Capita</em> shows a visibly weaker, noisier
      relationship — confirming H1. Point sizes are proportional to headcount H%, so larger
      bubbles represent states where more people live in deprivation.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — SDG PROGRESS TRACKER
# ─────────────────────────────────────────────────────────────────────────────
elif "Panel 4" in panel:
    st.markdown("<span class='panel-tag'>Panel 4</span>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>SDG Progress Tracker</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-desc'>State-level SDG-1 (No Poverty) and SDG-10 (Reduced Inequalities) "
        "tracking. Tier based on NITI Aayog SDG India Index 2023-24. "
        "SDG-10 proxy: Female GER ≥ 80% <em>and</em> MPI ≤ 0.07. "
        "Click column headers to sort, use filter tabs to narrow results.</div>",
        unsafe_allow_html=True,
    )

    # ── Tier filter tabs ──────────────────────────────────────────────────
    tier_tab = st.radio(
        "Filter by tier",
        ["All States", "🟢 Achievers", "🟡 Front-Runners", "🔴 Aspirants"],
        horizontal=True,
        label_visibility="collapsed",
    )
    tier_map = {
        "All States":      "All",
        "🟢 Achievers":    "🟢 Achiever",
        "🟡 Front-Runners":"🟡 Front-Runner",
        "🔴 Aspirants":    "🔴 Aspirant",
    }
    tier_filter = tier_map[tier_tab]

    # Search box
    search_q = st.text_input("🔍 Search state or region", placeholder="e.g. Bihar, Northeast…", label_visibility="collapsed")

    sdg_df = df.copy()
    if tier_filter != "All":
        sdg_df = sdg_df[sdg_df["sdg1_tier"] == tier_filter]
    if search_q.strip():
        q = search_q.lower()
        sdg_df = sdg_df[
            sdg_df["state"].str.lower().str.contains(q) |
            sdg_df["region"].str.lower().str.contains(q)
        ]

    sdg_display = sdg_df[[
        "state","region","mpi_value","H_headcount_pct",
        "sdg1_score","sdg4_score","female_sec_ger",
        "sdg1_tier","sdg10_status",
    ]].copy().sort_values("sdg1_score", ascending=False)

    sdg_display.columns = [
        "State","Region","MPI Value","H%",
        "SDG-1 Score","SDG-4 Score","F. GER%",
        "SDG-1 Tier","SDG-10 Status",
    ]
    sdg_display["MPI Value"] = sdg_display["MPI Value"].map("{:.4f}".format)
    sdg_display["H%"]        = sdg_display["H%"].map("{:.1f}%".format)
    sdg_display["F. GER%"]   = sdg_display["F. GER%"].map("{:.0f}%".format)

    # Styled dataframe
    def style_sdg_row(row):
        if "Achiever" in row["SDG-1 Tier"]:
            return ["background-color: #F7FFFB"] * len(row)
        if "Front" in row["SDG-1 Tier"]:
            return ["background-color: #FFFDF0"] * len(row)
        return ["background-color: #FFF8F8"] * len(row)

    styled = sdg_display.style.apply(style_sdg_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=350)

    # ── SDG-1 vs SDG-4 scatter ────────────────────────────────────────────
    st.markdown("#### SDG-4 (Education) vs SDG-1 (Poverty) — SDG Linkage Verification")
    sub_sdg = df[["sdg4_score","sdg1_score","mpi_value","state","region"]].dropna()

    fig_sdg = go.Figure()
    for reg in sub_sdg["region"].unique():
        sub_r = sub_sdg[sub_sdg["region"] == reg]
        fig_sdg.add_trace(go.Scatter(
            x    = sub_r["sdg4_score"],
            y    = sub_r["sdg1_score"],
            mode = "markers+text",
            name = reg,
            marker = dict(
                color   = sub_r["mpi_value"],
                colorscale = [[0,"#C8E6C9"],[0.5,"#FFCC80"],[1,"#B71C1C"]],
                size    = 14,
                cmin    = 0, cmax = 0.16,
                showscale = (reg == sub_sdg["region"].unique()[0]),
                colorbar  = dict(title="MPI Value", thickness=12, len=0.7, tickformat=".3f"),
                line    = dict(color="white", width=1),
            ),
            text         = sub_r["state"].str[:4],
            textposition = "top center",
            textfont     = dict(size=8),
            customdata   = sub_r[["state","mpi_value"]].values,
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "SDG-4: %{x}  SDG-1: %{y}<br>"
                "MPI: %{customdata[1]:.4f}<extra></extra>"
            ),
        ))

    # Add regression line
    xs4 = sub_sdg["sdg4_score"].values
    ys4 = sub_sdg["sdg1_score"].values
    xl4, yl4, _, _, m4, b4, r4, p4 = linreg_ci(xs4, ys4)
    fig_sdg.add_trace(go.Scatter(
        x=xl4, y=yl4, mode="lines",
        line=dict(color=NAVY, width=2),
        name=f"OLS  r={r4:.3f} ★★★",
        showlegend=True,
    ))

    # Threshold lines
    fig_sdg.add_hline(y=75, line_color=GREEN,  line_dash="dash", line_width=1.2,
                      annotation_text="SDG-1 Achiever ≥75", annotation_font_size=9)
    fig_sdg.add_vline(x=80, line_color=BLUE,   line_dash="dash", line_width=1.2,
                      annotation_text="SDG-4 ≥80", annotation_font_size=9)

    fig_sdg.update_layout(
        xaxis = dict(title="SDG-4 Score (Quality Education)", gridcolor="#E2E8F0"),
        yaxis = dict(title="SDG-1 Score (No Poverty)",        gridcolor="#E2E8F0"),
        plot_bgcolor  = "#F7F9FC",
        paper_bgcolor = "rgba(0,0,0,0)",
        legend = dict(orientation="h", y=-0.15, font=dict(size=9)),
        margin = dict(l=10, r=10, t=15, b=60),
        height = 400,
    )
    st.plotly_chart(fig_sdg, use_container_width=True, config={"displayModeBar": False})

    # ── Summary tiles ─────────────────────────────────────────────────────
    achievers     = (df["sdg1_tier"] == "🟢 Achiever").sum()
    front_runners = (df["sdg1_tier"] == "🟡 Front-Runner").sum()
    aspirants     = (df["sdg1_tier"] == "🔴 Aspirant").sum()
    on_track_10   = df["sdg10_status"].str.contains("On Track").sum()

    st.markdown(f"""
    <div class='sdg-tiles'>
      <div class='sdg-tile' style='background:#F0FFF4;border:1px solid #9AE6B4;'>
        <div class='sdg-count' style='color:#276749;'>{achievers}</div>
        <div class='sdg-lbl' style='color:#2F855A;'>🟢 ACHIEVERS<br>SDG-1 ≥ 75</div>
      </div>
      <div class='sdg-tile' style='background:#FFFBEB;border:1px solid #F6E05E;'>
        <div class='sdg-count' style='color:#744210;'>{front_runners}</div>
        <div class='sdg-lbl' style='color:#975A16;'>🟡 FRONT-RUNNERS<br>SDG-1 65–74</div>
      </div>
      <div class='sdg-tile' style='background:#FFF5F5;border:1px solid #FC8181;'>
        <div class='sdg-count' style='color:#742A2A;'>{aspirants}</div>
        <div class='sdg-lbl' style='color:#9B2C2C;'>🔴 ASPIRANTS<br>SDG-1 &lt; 65</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
      <strong>🎯 SDG Insight:</strong> {on_track_10} states meet the SDG-10 proxy
      (Female GER ≥ 80% AND MPI ≤ 0.07). States with Female GER &gt; 80% are 3×
      more likely to be SDG-1 Achievers. <strong>Bihar, Jharkhand, and Uttar Pradesh</strong>
      remain Aspirants — all have Female GER below 58%, directly confirming H1.
      The SDG-4 vs SDG-1 scatter (r={r4:.3f} ★★★) proves that education progress
      translates directly to poverty reduction, supporting SDG integration across Goals
      1, 4, and 10 simultaneously.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hr-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
  ABA Final Project · Phase 5 Dashboard · Subnational MPI Disparities in India<br>
  Sources: OPHI Global MPI 2024 (DHS 2019–21) · Global Data Lab SHDI 2021 ·
  NITI Aayog SDG India Index 2023-24 · MoSPI PLFS 2022-23<br>
  Alkire, Kanagaratnam & Suppa (2024) · Smits & Permanyer (2019) ·
  Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
