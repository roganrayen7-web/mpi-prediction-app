"""
╔══════════════════════════════════════════════════════════════════════╗
║   MPI LENS — India Multidimensional Poverty Prediction Dashboard    ║
║   Predict · Diagnose · Simulate · Rank                              ║
╚══════════════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─── Scikit-learn ───────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MPI Lens · India",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark editorial theme with teal accents
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Root & Body ── */
:root {
    --bg-deep:    #0b0f1a;
    --bg-card:    #111827;
    --bg-panel:   #1a2236;
    --border:     #1e2d47;
    --teal:       #00d4aa;
    --teal-dim:   #00a882;
    --amber:      #f59e0b;
    --rose:       #f43f5e;
    --lavender:   #818cf8;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
    --text-dim:   #94a3b8;
}

html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: var(--teal) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Titles ── */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     font-size: 2.4rem !important; letter-spacing: -0.02em !important;
     color: #ffffff !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
     font-size: 1.4rem !important; color: var(--teal) !important; }
h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
     color: var(--text-main) !important; }

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    flex: 1; min-width: 150px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.teal::before  { background: var(--teal); }
.kpi-card.amber::before { background: var(--amber); }
.kpi-card.rose::before  { background: var(--rose); }
.kpi-card.lav::before   { background: var(--lavender); }

.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem; font-weight: 800;
    color: #fff; line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem; color: var(--text-dim);
    margin-top: 4px; font-family: 'IBM Plex Mono', monospace;
}

/* ── Info Box ── */
.info-box {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    font-size: 0.85rem; color: var(--text-dim);
}
.info-box strong { color: var(--teal); }

/* ── Tag badge ── */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em; text-transform: uppercase;
    font-weight: 500;
}
.badge-dv  { background: rgba(0,212,170,0.15); color: var(--teal); border: 1px solid var(--teal-dim); }
.badge-iv  { background: rgba(129,140,248,0.15); color: var(--lavender); border: 1px solid var(--lavender); }
.badge-mod { background: rgba(245,158,11,0.15); color: var(--amber); border: 1px solid var(--amber); }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ── Streamlit elements override ── */
.stDataFrame, .stTable { border-radius: 8px; overflow: hidden; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Syne', sans-serif !important; }
.stSlider > div > div { background: var(--teal) !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Divider ── */
.section-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* ── Prediction Result ── */
.pred-result {
    background: linear-gradient(135deg, #0d2e28, #0f1e38);
    border: 1px solid var(--teal);
    border-radius: 14px; padding: 24px 28px;
    text-align: center; margin: 18px 0;
}
.pred-number {
    font-family: 'Syne', sans-serif; font-size: 3.5rem;
    font-weight: 800; color: var(--teal);
}
.pred-label { color: var(--text-dim); font-size: 0.85rem; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Load master dataset and define IVs/DV."""
    df = pd.read_csv(
        "Final_Master_Dataset__1___1_.csv"
        if __import__("os").path.exists("Final_Master_Dataset__1___1_.csv")
        else "/mnt/user-data/uploads/Final_Master_Dataset__1___1_.csv"
    )

    # ── Dependent Variable (DV) ──────────────────────────────────────────────
    DV = "mpi_value"

    # ── Independent Variables (IVs) grouped by domain ───────────────────────
    IV_GROUPS = {
        "Human Development": ["shdi", "education_index", "health_index", "income_index"],
        "Education": ["female_literacy", "female_sec_ger", "sdg4_score"],
        "Labour Market": ["female_lfpr_total", "male_lfpr_total", "gender_lfpr_gap"],
        "Living Standards": ["sanitation_pct", "rural_pop_pct", "infra_spend_pc"],
        "SDG Scores": ["sdg1_score", "sdg4_score"],
    }

    # Flat list of all IVs (unique, available columns)
    all_ivs_flat = list(dict.fromkeys(
        iv for grp in IV_GROUPS.values() for iv in grp
        if iv in df.columns
    ))

    # Clean: drop rows with NaN in DV or core IVs
    core_ivs = ["shdi", "education_index", "health_index", "income_index",
                "female_literacy", "sanitation_pct", "rural_pop_pct"]
    core_ivs = [c for c in core_ivs if c in df.columns]
    df_clean = df.dropna(subset=[DV] + core_ivs).reset_index(drop=True)

    return df, df_clean, DV, IV_GROUPS, all_ivs_flat


df_raw, df, DV, IV_GROUPS, ALL_IVS = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "🥇 Linear Regression": LinearRegression(),
    "🥈 Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5),
    "🥉 Gradient Boosting (XGBoost-style)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0),
}


@st.cache_data
def train_models(feature_cols):
    """Train all models on selected features and return results."""
    X = df[feature_cols].dropna()
    y = df.loc[X.index, DV]
    states = df.loc[X.index, "state"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_sc, y)
        y_pred = model.predict(X_sc)
        cv_r2  = cross_val_score(model, X_sc, y, cv=min(5, len(y)//3),
                                 scoring="r2").mean()
        cv_mae = (-cross_val_score(model, X_sc, y, cv=min(5, len(y)//3),
                                  scoring="neg_mean_absolute_error")).mean()
        results[name] = {
            "model":  model,
            "scaler": scaler,
            "X":      X,
            "y":      y,
            "y_pred": y_pred,
            "states": states,
            "r2":     r2_score(y, y_pred),
            "cv_r2":  cv_r2,
            "mae":    mean_absolute_error(y, y_pred),
            "cv_mae": cv_mae,
            "rmse":   np.sqrt(mean_squared_error(y, y_pred)),
            "features": feature_cols,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#1e2d47",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2d47",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
TEAL   = "#00d4aa"
AMBER  = "#f59e0b"
ROSE   = "#f43f5e"
LAV    = "#818cf8"
MUTED  = "#1e2d47"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
    color:#00d4aa;letter-spacing:-0.02em'>🔬 MPI Lens</div>
    <div style='font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;
    letter-spacing:0.1em;text-transform:uppercase;margin-top:2px'>India · State-Level Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.68rem;color:#64748b;font-family:monospace;text-transform:uppercase;letter-spacing:.1em'>Select Features (IVs)</div>", unsafe_allow_html=True)

    # Feature selector per domain
    selected_features = []
    for grp_name, grp_ivs in IV_GROUPS.items():
        avail = [c for c in grp_ivs if c in df.columns and df[c].notna().sum() >= 15]
        if not avail:
            continue
        chosen = st.multiselect(
            grp_name, avail,
            default=[avail[0]] if avail else [],
            key=f"feat_{grp_name}"
        )
        selected_features.extend(chosen)

    selected_features = list(dict.fromkeys(selected_features))  # deduplicate

    st.markdown("---")
    active_model_name = st.selectbox(
        "Active Model",
        list(MODELS.keys()),
        index=0
    )

    st.markdown("---")
    show_region = st.multiselect(
        "Filter by Region",
        options=sorted(df["region"].dropna().unique().tolist()),
        default=[]
    )

# ── Guard: need at least 2 features ──────────────────────────────────────────
if len(selected_features) < 2:
    st.markdown("""
    <div style='margin:60px auto;max-width:560px;text-align:center'>
    <div style='font-size:3rem'>🔬</div>
    <h1 style='margin:12px 0'>MPI Lens</h1>
    <p style='color:#64748b;font-size:1rem'>
    Predicting Multidimensional Poverty across Indian States<br>
    using Socio-Economic & Infrastructure Indicators.
    </p>
    <div class='info-box' style='text-align:left;margin-top:24px'>
    <strong>← Get started:</strong> Select at least <strong>2 features</strong>
    from the sidebar to train models and explore predictions.
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Train ─────────────────────────────────────────────────────────────────────
model_results = train_models(tuple(selected_features))
active = model_results[active_model_name]

# ── Region filter ─────────────────────────────────────────────────────────────
df_view = df.copy()
if show_region:
    df_view = df_view[df_view["region"].isin(show_region)]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1>MPI Lens <span style='font-size:1.1rem;color:#64748b;font-weight:400'>— India Poverty Intelligence Dashboard</span></h1>
<div style='display:flex;gap:10px;margin:8px 0 24px;flex-wrap:wrap'>
  <span class='badge badge-dv'>DV: MPI Value</span>
  <span class='badge badge-iv'>IVs: Socio-Economic Indicators</span>
  <span class='badge badge-mod'>Models: LR · RF · GBM · Ridge</span>
  <span style='color:#64748b;font-size:0.72rem;align-self:center;font-family:monospace'>DHS 2019–2021 · OPHI · GDL</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
def kpi_card(col, label, value, sub, color_cls):
    col.markdown(f"""
    <div class='kpi-card {color_cls}'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(k1, "National MPI", "0.069", "2019–21 Baseline", "teal")
kpi_card(k2, f"R² ({active_model_name.split()[0]})", f"{active['cv_r2']:.3f}",
         f"CV MAE: {active['cv_mae']:.4f}", "amber")
kpi_card(k3, "States Analysed", f"{len(active['y'])}", f"{len(selected_features)} IVs selected", "lav")
kpi_card(k4, "Poverty Reduction", "75%", "2006 → 2021 (MPI 0.283→0.069)", "rose")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Model Arena",
    "🔍 Feature Impact",
    "🗺️ State Rankings",
    "🔮 Predict a State",
    "📐 Diagnostics",
    "📖 Variable Guide",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 ─ MODEL ARENA
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Model Performance Comparison")
    st.markdown("<div class='info-box'><strong>Objective:</strong> Predict state-level MPI using selected IVs. Compare all models side-by-side across R², Cross-Validated R², MAE, and RMSE.</div>", unsafe_allow_html=True)

    # ── Metric table ─────────────────────────────────────────────────────────
    rows = []
    for mname, res in model_results.items():
        rows.append({
            "Model": mname,
            "R² (Train)": round(res["r2"], 4),
            "R² (CV)": round(res["cv_r2"], 4),
            "MAE": round(res["mae"], 5),
            "CV MAE": round(res["cv_mae"], 5),
            "RMSE": round(res["rmse"], 5),
        })
    metric_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        metric_df.style
          .highlight_max(subset=["R² (Train)", "R² (CV)"], color="#0d3a30")
          .highlight_min(subset=["MAE", "CV MAE", "RMSE"], color="#0d3a30")
          .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Actual vs Predicted (4 panels) ───────────────────────────────────────
    st.markdown("### Actual vs Predicted MPI")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    colors_map = [TEAL, AMBER, LAV, ROSE]
    for ax, (mname, res), col in zip(axes, model_results.items(), colors_map):
        y, yp = res["y"].values, res["y_pred"]
        ax.scatter(y, yp, color=col, alpha=0.75, s=55, edgecolors="none")
        mn, mx = min(y.min(), yp.min()), max(y.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "--", color="#ffffff", lw=1, alpha=0.4)
        ax.set_title(mname.split()[-1], color=col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual MPI", fontsize=7)
        ax.set_ylabel("Predicted MPI", fontsize=7)
        ax.text(0.05, 0.92, f"R²={res['cv_r2']:.3f}", transform=ax.transAxes,
                fontsize=8, color=col, fontfamily="monospace")
        ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.markdown("### Residual Distribution")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3), constrained_layout=True)
    for ax, (mname, res), col in zip(axes, model_results.items(), colors_map):
        resid = res["y"].values - res["y_pred"]
        ax.hist(resid, bins=12, color=col, alpha=0.7, edgecolor="none")
        ax.axvline(0, color="#fff", lw=1.2, ls="--", alpha=0.5)
        ax.set_title(mname.split()[-1], color=col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Residual", fontsize=7)
        ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 ─ FEATURE IMPACT
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Feature Impact Analysis")
    st.markdown("<div class='info-box'><strong>What drives MPI?</strong> Coefficient magnitudes (Linear/Ridge) and Feature Importance (RF/GBM) show which IVs most influence poverty predictions.</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # ── Linear coefficients ──────────────────────────────────────────────────
    with col_l:
        st.markdown("### Linear Regression Coefficients")
        lr = model_results["🥇 Linear Regression"]
        coef_df = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": lr["model"].coef_
        }).sort_values("Coefficient")

        fig, ax = plt.subplots(figsize=(7, max(3, len(selected_features) * 0.55)))
        colors = [ROSE if c < 0 else TEAL for c in coef_df["Coefficient"]]
        bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"],
                       color=colors, height=0.6, edgecolor="none")
        ax.axvline(0, color="#fff", lw=0.8, alpha=0.4)
        ax.set_xlabel("Standardised Coefficient", fontsize=8)
        ax.set_title("Impact on MPI (negative = poverty-reducing)", fontsize=9,
                     color=TEAL, pad=10)
        for bar, val in zip(bars, coef_df["Coefficient"]):
            ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                    fontsize=7.5, color="#e2e8f0", fontfamily="monospace")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # ── RF importance ────────────────────────────────────────────────────────
    with col_r:
        st.markdown("### Random Forest Feature Importance")
        rf = model_results["🥈 Random Forest"]
        imp = rf["model"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": imp
        }).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(7, max(3, len(selected_features) * 0.55)))
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                       color=AMBER, height=0.6, edgecolor="none", alpha=0.85)
        ax.set_xlabel("Importance Score", fontsize=8)
        ax.set_title("RF Feature Importance (higher = more influential)", fontsize=9,
                     color=AMBER, pad=10)
        for bar, val in zip(bars, imp_df["Importance"]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7.5,
                    color="#e2e8f0", fontfamily="monospace")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown("### Correlation Matrix: IVs ↔ MPI")
    corr_cols = [DV] + selected_features
    corr_data = df[corr_cols].dropna()
    corr_mat = corr_data.corr()

    fig, ax = plt.subplots(figsize=(max(6, len(corr_cols)*0.8), max(5, len(corr_cols)*0.72)))
    im = ax.imshow(corr_mat.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_yticks(range(len(corr_mat.index)))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_mat.index, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat.columns)):
            val = corr_mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="#000" if abs(val) > 0.5 else "#e2e8f0")
    ax.set_title("Pearson Correlation (green = positive, red = negative)", color=TEAL, fontsize=9)
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 ─ STATE RANKINGS
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## State-Level Poverty Rankings")

    # ── Predicted vs actual bar chart ─────────────────────────────────────
    pred_df = pd.DataFrame({
        "State": active["states"].values,
        "Actual MPI":    active["y"].values,
        "Predicted MPI": active["y_pred"],
    }).sort_values("Actual MPI", ascending=False)

    if show_region:
        pred_df = pred_df[pred_df["State"].isin(
            df_view["state"].values
        )]

    fig, ax = plt.subplots(figsize=(14, max(5, len(pred_df) * 0.42)))
    x = np.arange(len(pred_df))
    w = 0.38
    ax.barh(x - w/2, pred_df["Actual MPI"],    w, color=TEAL, alpha=0.85, label="Actual MPI",    edgecolor="none")
    ax.barh(x + w/2, pred_df["Predicted MPI"], w, color=AMBER, alpha=0.85, label="Predicted MPI", edgecolor="none")
    ax.set_yticks(x)
    ax.set_yticklabels(pred_df["State"], fontsize=8.5)
    ax.set_xlabel("MPI Value", fontsize=9)
    ax.set_title(f"Actual vs Predicted MPI by State — {active_model_name}", color=TEAL, fontsize=10, pad=10)
    ax.legend(framealpha=0, fontsize=9)
    ax.axvline(0.069, color="#f43f5e", lw=1.2, ls="--", alpha=0.6)
    ax.text(0.070, len(pred_df)-0.5, "National Avg 0.069", color="#f43f5e", fontsize=7.5)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Scatter: 2 selected IVs vs MPI ───────────────────────────────────
    st.markdown("### Scatter: IV vs MPI")
    iv_choice = st.selectbox("Select an IV to plot against MPI", selected_features)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_vals = df[iv_choice]
    y_vals = df[DV]
    valid  = x_vals.notna() & y_vals.notna()
    ax.scatter(x_vals[valid], y_vals[valid], color=LAV, s=70, alpha=0.8, edgecolors="none")
    # regression line
    from numpy.polynomial import polynomial as P
    if valid.sum() > 3:
        c = np.polyfit(x_vals[valid], y_vals[valid], 1)
        xl = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
        ax.plot(xl, np.polyval(c, xl), color=TEAL, lw=2, alpha=0.7)
    for _, row in df[valid].iterrows():
        ax.annotate(row["state"][:4], (row[iv_choice], row[DV]),
                    fontsize=6, color="#64748b", alpha=0.7,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(iv_choice, fontsize=9)
    ax.set_ylabel("MPI Value", fontsize=9)
    ax.set_title(f"{iv_choice}  ↔  MPI", color=LAV, fontsize=10)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # ── Ranking table ─────────────────────────────────────────────────────
    st.markdown("### Full Ranking Table")
    rank_df = pred_df.copy()
    rank_df["Error"] = (rank_df["Predicted MPI"] - rank_df["Actual MPI"]).round(5)
    rank_df["Rank (Actual)"] = rank_df["Actual MPI"].rank(ascending=False).astype(int)
    rank_df = rank_df.reset_index(drop=True)
    st.dataframe(
        rank_df.style.background_gradient(subset=["Actual MPI", "Predicted MPI"],
                                           cmap="YlOrRd"),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 ─ PREDICT A STATE
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔮 Predict MPI for a Custom Profile")
    st.markdown("<div class='info-box'><strong>Policy Simulator:</strong> Adjust socio-economic indicators below to see how a state's predicted MPI would change. Perfect for <em>what-if</em> scenario planning.</div>", unsafe_allow_html=True)

    # ── Pre-fill from existing state ────────────────────────────────────
    col_pick, _ = st.columns([1, 2])
    with col_pick:
        prefill_state = st.selectbox(
            "Pre-fill from existing state",
            ["(custom)"] + sorted(df["state"].tolist())
        )

    if prefill_state != "(custom)":
        row_ref = df[df["state"] == prefill_state].iloc[0]
    else:
        row_ref = df[selected_features].median()

    # ── Sliders ──────────────────────────────────────────────────────────
    user_vals = {}
    n_cols = 3
    slider_cols = st.columns(n_cols)
    for i, feat in enumerate(selected_features):
        col = slider_cols[i % n_cols]
        feat_data = df[feat].dropna()
        fmin, fmax, fmean = float(feat_data.min()), float(feat_data.max()), float(feat_data.mean())
        default_val = float(row_ref[feat]) if feat in row_ref.index and pd.notna(row_ref[feat]) else fmean
        user_vals[feat] = col.slider(
            feat,
            min_value=round(fmin, 3),
            max_value=round(fmax, 3),
            value=round(np.clip(default_val, fmin, fmax), 3),
            step=round((fmax - fmin) / 200, 4),
            key=f"pred_{feat}"
        )

    # ── Run prediction ────────────────────────────────────────────────────
    input_arr = np.array([[user_vals[f] for f in selected_features]])
    predictions_out = {}
    for mname, res in model_results.items():
        X_in_sc = res["scaler"].transform(input_arr)
        predictions_out[mname] = float(res["model"].predict(X_in_sc)[0])

    # ── Result display ────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    pred_cols = st.columns(4)
    mcols = list(model_results.keys())
    pcols_colors = [("teal", TEAL), ("amber", AMBER), ("lav", LAV), ("rose", ROSE)]
    for col, mname, (cls, hex_) in zip(pred_cols, mcols, pcols_colors):
        pred_val = predictions_out[mname]
        label = "VERY LOW" if pred_val < 0.03 else ("LOW" if pred_val < 0.06 else ("MEDIUM" if pred_val < 0.1 else "HIGH"))
        col.markdown(f"""
        <div class='pred-result' style='border-color:{hex_}'>
          <div class='kpi-label'>{mname.split()[-1]}</div>
          <div class='pred-number' style='color:{hex_}'>{pred_val:.4f}</div>
          <div class='pred-label'>Poverty Level: <strong style='color:{hex_}'>{label}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Ensemble average
    avg_pred = np.mean(list(predictions_out.values()))
    st.markdown(f"""
    <div style='text-align:center;margin:16px 0'>
      <span style='font-family:Syne,sans-serif;font-size:1.1rem;color:#94a3b8'>
      Ensemble Average MPI: </span>
      <span style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#00d4aa'>
      {avg_pred:.4f}</span>
      <span style='font-size:0.78rem;color:#64748b;margin-left:8px'>
      (National avg: 0.069)</span>
    </div>
    """, unsafe_allow_html=True)

    # Context bar
    fig, ax = plt.subplots(figsize=(10, 1.2))
    ax.barh(0, 0.154, color=MUTED, height=0.5, edgecolor="none")  # max scale
    ax.barh(0, avg_pred, color=TEAL, height=0.5, edgecolor="none", alpha=0.9)
    ax.axvline(0.069, color=ROSE, lw=2, ls="--", alpha=0.8)
    ax.text(0.070, 0.3, "National avg", color=ROSE, fontsize=7.5, va="center")
    ax.set_xlim(0, 0.16)
    ax.set_yticks([])
    ax.set_xlabel("MPI Scale (0 = no poverty, 0.154 = Bihar, highest)", fontsize=8)
    ax.set_title("Your profile on the MPI scale", color=TEAL, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 ─ DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Model Diagnostics")

    diag_model = st.selectbox("Select model for diagnostics", list(model_results.keys()))
    res = model_results[diag_model]
    y_true = res["y"].values
    y_pred = res["y_pred"]
    resid  = y_true - y_pred

    dc1, dc2, dc3 = st.columns(3)

    # ── Q-Q plot ──────────────────────────────────────────────────────────
    with dc1:
        st.markdown("**Q-Q Plot (Residuals)**")
        from scipy import stats as scipy_stats
        fig, ax = plt.subplots(figsize=(5, 4))
        osm, osr = scipy_stats.probplot(resid, dist="norm")
        ax.scatter(osm[0], osm[1], color=TEAL, s=40, alpha=0.8)
        ax.plot(osm[0], osr[0]*np.array(osm[0]) + osr[1], color=AMBER, lw=1.5)
        ax.set_xlabel("Theoretical Quantiles", fontsize=8)
        ax.set_ylabel("Sample Quantiles", fontsize=8)
        ax.set_title("Normal Q-Q", color=TEAL, fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    # ── Scale-Location ────────────────────────────────────────────────────
    with dc2:
        st.markdown("**Scale-Location (Homoscedasticity)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred, np.sqrt(np.abs(resid)), color=LAV, s=40, alpha=0.8)
        ax.set_xlabel("Fitted Values", fontsize=8)
        ax.set_ylabel("√|Residuals|", fontsize=8)
        ax.set_title("Scale-Location", color=LAV, fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    # ── Residuals vs Fitted ───────────────────────────────────────────────
    with dc3:
        st.markdown("**Residuals vs Fitted**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred, resid, color=AMBER, s=40, alpha=0.8)
        ax.axhline(0, color="#fff", lw=1.2, ls="--", alpha=0.5)
        ax.set_xlabel("Fitted Values", fontsize=8)
        ax.set_ylabel("Residuals", fontsize=8)
        ax.set_title("Residuals vs Fitted", color=AMBER, fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("### Regression Summary")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | R² (Training) | `{res['r2']:.4f}` |
        | R² (Cross-Val) | `{res['cv_r2']:.4f}` |
        | MAE | `{res['mae']:.6f}` |
        | RMSE | `{res['rmse']:.6f}` |
        | CV MAE | `{res['cv_mae']:.6f}` |
        | N (states) | `{len(y_true)}` |
        | Features | `{len(selected_features)}` |
        """)
    with s_col2:
        if diag_model == "🥇 Linear Regression":
            st.markdown("**Coefficient Table**")
            coef_tab = pd.DataFrame({
                "Variable": selected_features,
                "Coefficient": res["model"].coef_.round(6),
                "Abs. Impact": np.abs(res["model"].coef_).round(6)
            }).sort_values("Abs. Impact", ascending=False)
            st.dataframe(coef_tab, use_container_width=True)

    # ── Outlier detection ─────────────────────────────────────────────────
    st.markdown("### Outlier Detection (High-Leverage States)")
    outlier_threshold = 2 * np.std(resid)
    outlier_df = pd.DataFrame({
        "State": res["states"].values,
        "Actual": y_true.round(4),
        "Predicted": y_pred.round(4),
        "Residual": resid.round(5),
        "Flag": ["⚠️ Outlier" if abs(r) > outlier_threshold else "✓ Normal" for r in resid]
    }).sort_values("Residual", key=abs, ascending=False)
    st.dataframe(outlier_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 ─ VARIABLE GUIDE
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Variable Guide & Conceptual Framework")

    st.markdown("""
    <div class='info-box'>
    <strong>Project Statement:</strong> Predict the Multidimensional Poverty Index (MPI) of Indian states
    using socio-economic and infrastructure indicators. The MPI = H × A, where H = headcount ratio
    and A = average intensity of deprivation across 10 indicators in 3 dimensions.
    </div>
    """, unsafe_allow_html=True)

    guide_data = {
        "Variable": [
            "mpi_value", "H_headcount_pct", "A_intensity_pct",
            "shdi", "education_index", "health_index", "income_index",
            "female_literacy", "female_sec_ger", "sdg4_score",
            "female_lfpr_total", "male_lfpr_total", "gender_lfpr_gap",
            "sanitation_pct", "rural_pop_pct", "infra_spend_pc", "sdg1_score"
        ],
        "Type": [
            "DV", "DV-component", "DV-component",
            "IV (HDI)", "IV (HDI)", "IV (HDI)", "IV (HDI)",
            "IV (Edu)", "IV (Edu)", "IV (Edu)",
            "IV (Labour)", "IV (Labour)", "IV (Labour)",
            "IV (Infra)", "IV (Infra)", "IV (Infra)", "IV (SDG)"
        ],
        "Description": [
            "Global MPI value (OPHI, 2019–21) — PRIMARY TARGET",
            "% of population classified as multidimensionally poor",
            "Average intensity of deprivation among the poor (%)",
            "Sub-national Human Development Index (GDL)",
            "Education sub-index from GDL (literacy + enrolment)",
            "Health sub-index from GDL (life expectancy)",
            "Income sub-index from GDL (per-capita GNI proxy)",
            "Female adult literacy rate (%)",
            "Female secondary gross enrolment ratio (%)",
            "SDG-4 (Quality Education) state score",
            "Female Labour Force Participation Rate — total (%)",
            "Male Labour Force Participation Rate — total (%)",
            "Gap between male and female LFPR (percentage points)",
            "Households with access to improved sanitation (%)",
            "Rural population as % of total state population",
            "Per-capita infrastructure expenditure (INR)",
            "SDG-1 (No Poverty) composite state score"
        ],
        "Expected Effect on MPI": [
            "—", "Positive (component)", "Positive (component)",
            "Negative ↓", "Negative ↓", "Negative ↓", "Negative ↓",
            "Negative ↓", "Negative ↓", "Negative ↓",
            "Ambiguous", "Negative ↓", "Positive ↑",
            "Negative ↓", "Positive ↑", "Negative ↓", "Negative ↓"
        ]
    }
    guide_df = pd.DataFrame(guide_data)

    def style_type(v):
        if "DV" in v: return "background-color:#0d3a30;color:#00d4aa"
        if "HDI" in v: return "background-color:#1e1a3a;color:#818cf8"
        if "Edu" in v: return "background-color:#1a2a1a;color:#4ade80"
        if "Labour" in v: return "background-color:#2a1a1a;color:#fb923c"
        return "background-color:#1a1a2a;color:#94a3b8"

    st.dataframe(
        guide_df.style.applymap(style_type, subset=["Type"]),
        use_container_width=True, height=520
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    ### Model Hierarchy

    | Rank | Model | Why Use It |
    |---|---|---|
    | 🥇 | **Multiple Linear Regression** | Baseline; interpretable coefficients; shows direction & magnitude of each IV |
    | 🥈 | **Random Forest Regressor** | Handles non-linearity; feature importance; robust to outliers |
    | 🥉 | **Gradient Boosting** | Highest accuracy; captures complex interactions; industry-grade |
    | 🔧 | **Ridge Regression** | Handles multicollinearity among correlated HDI sub-indices |

    ### Key Research Questions
    1. **Which IV most reduces MPI?** → Check Feature Impact tab
    2. **Which states are outliers vs predictions?** → Diagnostics tab
    3. **What-if policy scenario?** → Predict a State tab
    4. **Is the DV normally distributed?** → Q-Q plot in Diagnostics
    """)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-top:40px;padding:20px;border-top:1px solid #1e2d47;
text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#334155'>
MPI Lens · Data: OPHI Global MPI 2024 (DHS 2019–21) · GDL Subnational HDI · PLFS / ILO-STAT LFPR ·
SDG India Index · Built with Streamlit · For academic use
</div>
""", unsafe_allow_html=True)
