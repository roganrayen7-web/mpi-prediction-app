"""
╔══════════════════════════════════════════════════════════════════════╗
║   MPI LENS — India Multidimensional Poverty Prediction Dashboard    ║
║   FIX: matplotlib.use("Agg") BEFORE pyplot — resolves cloud error  ║
║   Run: streamlit run app.py                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── CRITICAL FIX ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # Must come before pyplot import — fixes Streamlit Cloud error

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MPI Lens · India", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

TEAL="#00d4aa"; AMBER="#f59e0b"; ROSE="#f43f5e"; LAV="#818cf8"; MUTED="#1e2d47"; BG="#111827"

plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":BG,"axes.edgecolor":MUTED,
    "axes.labelcolor":"#94a3b8","xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":MUTED,"grid.linewidth":0.5,
    "font.family":"monospace","axes.spines.top":False,"axes.spines.right":False,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
:root{--bg:#0b0f1a;--card:#111827;--panel:#1a2236;--border:#1e2d47;
  --teal:#00d4aa;--teal2:#00a882;--amber:#f59e0b;--rose:#f43f5e;
  --lav:#818cf8;--tx:#e2e8f0;--tm:#64748b;--td:#94a3b8;}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--tx)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
[data-testid="stSidebar"] label{color:var(--teal)!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:0.72rem!important;letter-spacing:0.08em;text-transform:uppercase;}
h1{font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:2.2rem!important;
  letter-spacing:-0.02em!important;color:#fff!important;}
h2{font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:1.35rem!important;color:var(--teal)!important;}
h3{font-family:'Syne',sans-serif!important;font-weight:600!important;color:var(--tx)!important;}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kpi.t::before{background:var(--teal);}.kpi.a::before{background:var(--amber);}
.kpi.r::before{background:var(--rose);}.kpi.l::before{background:var(--lav);}
.kpi-lb{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--tm);margin-bottom:6px;}
.kpi-v{font-family:'Syne',sans-serif;font-size:1.85rem;font-weight:800;color:#fff;line-height:1;}
.kpi-s{font-size:0.72rem;color:var(--td);margin-top:4px;font-family:'IBM Plex Mono',monospace;}
.ib{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--teal);
  border-radius:8px;padding:14px 18px;margin:12px 0;font-size:0.85rem;color:var(--td);}
.ib strong{color:var(--teal);}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.68rem;
  font-family:'IBM Plex Mono',monospace;letter-spacing:0.06em;text-transform:uppercase;font-weight:500;}
.dv{background:rgba(0,212,170,.15);color:var(--teal);border:1px solid var(--teal2);}
.iv{background:rgba(129,140,248,.15);color:var(--lav);border:1px solid var(--lav);}
.md{background:rgba(245,158,11,.15);color:var(--amber);border:1px solid var(--amber);}
[data-testid="stTabs"] button{font-family:'IBM Plex Mono',monospace!important;
  font-size:0.78rem!important;letter-spacing:0.06em!important;text-transform:uppercase!important;color:var(--tm)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--teal)!important;border-bottom:2px solid var(--teal)!important;}
.stSlider>div>div{background:var(--teal)!important;}
footer{display:none!important;}#MainMenu{display:none!important;}
.sdiv{border:none;border-top:1px solid var(--border);margin:24px 0;}
.pr{background:linear-gradient(135deg,#0d2e28,#0f1e38);border:1px solid var(--teal);
  border-radius:14px;padding:24px 28px;text-align:center;margin:18px 0;}
.pn{font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;color:var(--teal);}
.pl{color:var(--td);font-size:0.85rem;margin-top:6px;}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = ["Final_Master_Dataset.csv",
             "Final_Master_Dataset__1___1_.csv",
             "/mnt/user-data/outputs/Final_Master_Dataset.csv",
             "/mnt/user-data/uploads/1776879438993_Final_Master_Dataset__1___1_.csv"]
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p); break
    if df is None:
        st.error("Dataset not found. Place Final_Master_Dataset.csv in the app folder.")
        st.stop()

    DV = "mpi_value"
    IVG = {
        "Human Development": ["shdi","education_index","health_index","income_index"],
        "Education":         ["female_literacy","female_sec_ger","sdg4_score"],
        "Labour Market":     ["female_lfpr_total","male_lfpr_total","gender_lfpr_gap"],
        "Living Standards":  ["sanitation_pct","rural_pop_pct","infra_spend_pc"],
        "SDG Scores":        ["sdg1_score"],
    }
    IVG = {g:[c for c in cols if c in df.columns] for g,cols in IVG.items()}
    all_ivs = list(dict.fromkeys(c for cols in IVG.values() for c in cols))
    core = [c for c in ["shdi","education_index","health_index","female_literacy","sanitation_pct"] if c in df.columns]
    df_clean = df.dropna(subset=[DV]+core).reset_index(drop=True)
    return df, df_clean, DV, IVG, all_ivs

df_raw, df, DV, IVG, ALL_IVS = load_data()

MODELS_DEF = {
    "🥇 Linear Regression": LinearRegression(),
    "🥈 Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5),
    "🥉 Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "🔧 Ridge Regression":  Ridge(alpha=1.0),
}

@st.cache_data
def train_models(feat_tuple):
    feats = list(feat_tuple)
    sub = df[feats+[DV,"state","region"]].dropna()
    X,y,states,regions = sub[feats], sub[DV], sub["state"], sub["region"]
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    n_cv = min(5, max(2, len(y)//4))
    res = {}
    for name, m in MODELS_DEF.items():
        m = clone(m); m.fit(Xs,y); yp = m.predict(Xs)
        cv_r2  = cross_val_score(m,Xs,y,cv=n_cv,scoring="r2").mean()
        cv_mae = -cross_val_score(m,Xs,y,cv=n_cv,scoring="neg_mean_absolute_error").mean()
        res[name]={"model":m,"scaler":sc,"X":X,"y":y,"y_pred":yp,"states":states,
                   "regions":regions,"r2":r2_score(y,yp),"cv_r2":cv_r2,
                   "mae":mean_absolute_error(y,yp),"cv_mae":cv_mae,
                   "rmse":np.sqrt(mean_squared_error(y,yp)),"features":feats}
    return res


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='padding:12px 0 8px'><div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:{TEAL};letter-spacing:-0.02em'>🔬 MPI Lens</div><div style='font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px'>India · State Analysis</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.68rem;color:#64748b;font-family:monospace;text-transform:uppercase;letter-spacing:.1em'>Select Features (IVs)</div>",unsafe_allow_html=True)
    sel_feats=[]
    for gn,gi in IVG.items():
        avail=[c for c in gi if df[c].notna().sum()>=15]
        if not avail: continue
        chosen=st.multiselect(gn,avail,default=[avail[0]],key=f"f_{gn}")
        sel_feats.extend(chosen)
    sel_feats=list(dict.fromkeys(sel_feats))
    st.markdown("---")
    active_name=st.selectbox("Active Model",list(MODELS_DEF.keys()),index=0)
    st.markdown("---")
    show_region=st.multiselect("Filter by Region",sorted(df["region"].dropna().unique().tolist()),default=[])
    st.markdown("---")
    st.markdown(f"<div style='font-size:0.62rem;color:#334155;font-family:IBM Plex Mono,monospace;line-height:1.7;'>Data: OPHI MPI 2024 · DHS 2019–21<br>GDL SHDI · PLFS 2022-23<br>n={len(df)} states · DV: mpi_value<br>{len(ALL_IVS)} IVs available</div>",unsafe_allow_html=True)

if len(sel_feats)<2:
    st.markdown("<div style='margin:60px auto;max-width:560px;text-align:center;'><div style='font-size:3rem'>🔬</div><h1>MPI Lens</h1><p style='color:#64748b;'>Select at least <b>2 features</b> from the sidebar to start.</p></div>",unsafe_allow_html=True)
    st.stop()

with st.spinner("Training 4 models..."):
    model_results=train_models(tuple(sel_feats))
active=model_results[active_name]
df_view=df[df["region"].isin(show_region)] if show_region else df.copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""<h1>MPI Lens <span style='font-size:1rem;color:#64748b;font-weight:400;'>— India Poverty Intelligence Dashboard</span></h1>
<div style='display:flex;gap:10px;margin:8px 0 24px;flex-wrap:wrap;'>
<span class='badge dv'>DV: MPI Value</span><span class='badge iv'>IVs: Socio-Economic</span>
<span class='badge md'>Models: LR · RF · GBM · Ridge</span>
<span style='color:#64748b;font-size:0.72rem;align-self:center;font-family:monospace;'>DHS 2019–2021 · OPHI · GDL · PLFS</span>
</div>""",unsafe_allow_html=True)

def kpi(col,lb,v,s,c):
    col.markdown(f"<div class='kpi {c}'><div class='kpi-lb'>{lb}</div><div class='kpi-v'>{v}</div><div class='kpi-s'>{s}</div></div>",unsafe_allow_html=True)

k1,k2,k3,k4=st.columns(4)
kpi(k1,"National MPI","0.069","2019–21 Baseline","t")
kpi(k2,f"R² ({active_name.split()[0]})",f"{active['cv_r2']:.3f}",f"CV-MAE: {active['cv_mae']:.4f}","a")
kpi(k3,"States Analysed",f"{len(active['y'])}",f"{len(sel_feats)} IVs selected","l")
kpi(k4,"Poverty Reduction","75%","2006→2021  (0.283→0.069)","r")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs=st.tabs(["📊 Model Arena","🔍 Feature Impact","🗺️ State Rankings",
              "🔮 Predict a State","📐 Diagnostics","📖 Variable Guide"])


# ── TAB 0: MODEL ARENA ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## Model Arena — Performance Comparison")
    st.markdown(f"<div class='ib'><strong>Training:</strong> {len(sel_feats)} features · n={len(active['y'])} states · StandardScaler · {min(5,max(2,len(active['y'])//4))}-fold CV</div>",unsafe_allow_html=True)

    rows=[]
    for mn,r in model_results.items():
        rows.append({"Model":mn,"Train R²":round(r["r2"],4),"CV R²":round(r["cv_r2"],4),
                     "Train MAE":round(r["mae"],6),"CV MAE":round(r["cv_mae"],6),"RMSE":round(r["rmse"],6)})
    adf=pd.DataFrame(rows).sort_values("CV R²",ascending=False)
    st.dataframe(adf.style.background_gradient(subset=["CV R²"],cmap="YlGn")
                          .background_gradient(subset=["CV MAE"],cmap="YlOrRd_r"),use_container_width=True)

    st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)
    c1,c2=st.columns(2)

    with c1:
        st.markdown("**CV R² by Model**")
        fig,ax=plt.subplots(figsize=(6,3.5))
        names=[r.split()[0]+" "+r.split()[1] for r in adf["Model"]]
        vals=adf["CV R²"].values
        cols=[TEAL,LAV,AMBER,ROSE][:len(names)]
        bars=ax.barh(names,vals,color=cols,height=0.55,alpha=0.9)
        for b,v in zip(bars,vals):
            ax.text(v+0.005,b.get_y()+b.get_height()/2,f"{v:.3f}",va="center",fontsize=9,color="#e2e8f0")
        ax.set_xlabel("CV R²",fontsize=9); ax.set_xlim(0,1.05); ax.grid(axis="x",alpha=0.3)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown(f"**Actual vs Predicted — {active_name.split()[0]}**")
        yt=active["y"].values; yp=active["y_pred"]
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.scatter(yt,yp,color=TEAL,s=55,alpha=0.85,edgecolors="white",lw=0.5)
        mn2=min(yt.min(),yp.min())-0.002; mx2=max(yt.max(),yp.max())+0.002
        ax.plot([mn2,mx2],[mn2,mx2],color=AMBER,lw=1.5,ls="--",label="Perfect fit")
        resid2=yt-yp; thr=2*resid2.std()
        for i,(a,b,r) in enumerate(zip(yt,yp,resid2)):
            if abs(r)>thr: ax.annotate(active["states"].iloc[i][:8],(a,b),fontsize=7,color=ROSE,xytext=(5,5),textcoords="offset points")
        ax.set_xlabel("Actual MPI",fontsize=9); ax.set_ylabel("Predicted MPI",fontsize=9)
        ax.set_title(f"R²={active['r2']:.3f}  MAE={active['mae']:.4f}",color=TEAL,fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()


# ── TAB 1: FEATURE IMPACT ─────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## Feature Impact — What Drives MPI?")
    imp_name=st.selectbox("Model for analysis",list(model_results.keys()),index=0,key="imp")
    imp=model_results[imp_name]
    Xsc=imp["scaler"].transform(imp["X"])
    c1,c2=st.columns(2)

    with c1:
        st.markdown("**Permutation Importance**")
        perm=permutation_importance(imp["model"],Xsc,imp["y"],n_repeats=20,random_state=42)
        pdf=pd.DataFrame({"Feature":sel_feats,"Importance":perm.importances_mean,"Std":perm.importances_std}).sort_values("Importance",ascending=True)
        fig,ax=plt.subplots(figsize=(6,max(3,len(sel_feats)*0.6)))
        colors_pi=[TEAL if v>=0 else ROSE for v in pdf["Importance"]]
        ax.barh(pdf["Feature"],pdf["Importance"],xerr=pdf["Std"],color=colors_pi,height=0.6,error_kw={"ecolor":"#64748b","linewidth":0.8},alpha=0.9)
        ax.axvline(0,color="#334155",lw=0.8); ax.set_xlabel("Mean ΔR²",fontsize=9)
        ax.set_title("Permutation Importance",color=TEAL,fontsize=10); ax.grid(axis="x",alpha=0.3)
        st.pyplot(fig); plt.close()

    with c2:
        if hasattr(imp["model"],"feature_importances_"):
            st.markdown("**Tree Feature Importances**")
            fi=pd.DataFrame({"Feature":sel_feats,"Importance":imp["model"].feature_importances_}).sort_values("Importance",ascending=True)
            fig,ax=plt.subplots(figsize=(6,max(3,len(sel_feats)*0.6)))
            ax.barh(fi["Feature"],fi["Importance"],color=LAV,height=0.6,alpha=0.9)
            ax.set_xlabel("Gini Importance",fontsize=9); ax.set_title("Built-in Importances",color=LAV,fontsize=10); ax.grid(axis="x",alpha=0.3)
            st.pyplot(fig); plt.close()
        else:
            st.markdown("**Standardised Coefficients**")
            cdf=pd.DataFrame({"Feature":sel_feats,"Coef":imp["model"].coef_}).sort_values("Coef")
            fig,ax=plt.subplots(figsize=(6,max(3,len(sel_feats)*0.6)))
            colors_c=[TEAL if c<0 else ROSE for c in cdf["Coef"]]
            ax.barh(cdf["Feature"],cdf["Coef"],color=colors_c,height=0.6,alpha=0.9)
            ax.axvline(0,color="#334155",lw=0.8); ax.set_xlabel("Standardised β",fontsize=9)
            ax.set_title("Coefficient Plot (neg = reduces MPI)",color=TEAL,fontsize=10); ax.grid(axis="x",alpha=0.3)
            st.pyplot(fig); plt.close()

    st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)
    st.markdown("**Pearson Correlation Matrix**")
    corr_cols=sel_feats+[DV]
    cm=df[corr_cols].dropna().corr()
    fig,ax=plt.subplots(figsize=(max(6,len(corr_cols)*0.9),max(5,len(corr_cols)*0.8)))
    im=ax.imshow(cm,cmap=plt.cm.RdBu_r,vmin=-1,vmax=1,aspect="auto")
    plt.colorbar(im,ax=ax,shrink=0.8)
    ax.set_xticks(range(len(cm.columns))); ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns,rotation=40,ha="right",fontsize=8)
    ax.set_yticklabels(cm.index,fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm.columns)):
            v=cm.iloc[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(v)>0.6 else "#94a3b8")
    ax.set_title("Correlation Matrix",color=TEAL,fontsize=10)
    st.pyplot(fig); plt.close()


# ── TAB 2: STATE RANKINGS ─────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## State Rankings — Actual vs Predicted MPI")
    rn=st.selectbox("Model",list(model_results.keys()),index=0,key="rn")
    rr=model_results[rn]
    rdf=pd.DataFrame({"State":rr["states"].values,"Region":rr["regions"].values,
                       "Actual":rr["y"].values.round(4),"Predicted":rr["y_pred"].round(4),
                       "Residual":(rr["y"].values-rr["y_pred"]).round(5)}
    ).sort_values("Actual",ascending=False).reset_index(drop=True)
    rdf["Rank"]=rdf.index+1
    sdg1_vals=df.set_index("state").reindex(rdf["State"])["sdg1_score"].values
    rdf["SDG-1"]=sdg1_vals
    rdf["SDG-1 Tier"]=rdf["SDG-1"].apply(lambda s:"🟢 Achiever" if s>=75 else "🟡 Front-Runner" if s>=65 else "🔴 Aspirant")
    st.dataframe(rdf[["Rank","State","Region","Actual","Predicted","Residual","SDG-1","SDG-1 Tier"]]
                 .style.applymap(lambda v:"color:#f43f5e" if v>0.01 else "color:#00d4aa" if v<-0.01 else "color:#94a3b8",subset=["Residual"]),
                 use_container_width=True,height=500)

    st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)
    rc1,rc2=st.columns(2)
    top10=rdf.head(10); bot10=rdf.tail(10)
    with rc1:
        st.markdown("**Top 10 Highest Poverty**")
        fig,ax=plt.subplots(figsize=(5.5,3.8))
        ax.barh(top10["State"].str[:15],top10["Actual"],color=ROSE,height=0.6,alpha=0.85,label="Actual")
        ax.barh(top10["State"].str[:15],top10["Predicted"],color=AMBER,height=0.3,alpha=0.7,label="Predicted")
        ax.set_xlabel("MPI",fontsize=9); ax.legend(fontsize=8); ax.grid(axis="x",alpha=0.3); ax.tick_params(axis="y",labelsize=8)
        st.pyplot(fig); plt.close()
    with rc2:
        st.markdown("**Bottom 10 Lowest Poverty**")
        fig,ax=plt.subplots(figsize=(5.5,3.8))
        ax.barh(bot10["State"].str[:15],bot10["Actual"],color=TEAL,height=0.6,alpha=0.85,label="Actual")
        ax.barh(bot10["State"].str[:15],bot10["Predicted"],color=LAV,height=0.3,alpha=0.7,label="Predicted")
        ax.set_xlabel("MPI",fontsize=9); ax.legend(fontsize=8); ax.grid(axis="x",alpha=0.3); ax.tick_params(axis="y",labelsize=8)
        st.pyplot(fig); plt.close()


# ── TAB 3: PREDICT A STATE ────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🔮 Predict a State — What-If Policy Simulator")
    st.markdown("<div class='ib'><strong>How to use:</strong> Adjust sliders to simulate a policy scenario. The model predicts MPI under those conditions.</div>",unsafe_allow_html=True)
    sm_name=st.selectbox("Model for prediction",list(model_results.keys()),index=0,key="sm")
    sm=model_results[sm_name]

    st.markdown("#### Load a state as baseline")
    base=st.selectbox("Copy values from state",["— manual entry —"]+sorted(df["state"].tolist()),key="bs")
    base_row=df[df["state"]==base].iloc[0] if base!="— manual entry —" else None

    META={"shdi":(0.40,1.00,0.01,"Sub-national HDI"),
          "education_index":(0.30,1.00,0.01,"Education Index"),
          "health_index":(0.40,1.00,0.01,"Health Index"),
          "income_index":(0.40,1.00,0.01,"Income Index"),
          "female_literacy":(20.0,100.0,0.5,"Female Literacy (%)"),
          "female_sec_ger":(30.0,100.0,0.5,"Female GER (%)"),
          "sdg4_score":(20.0,100.0,1.0,"SDG-4 Score"),
          "female_lfpr_total":(10.0,60.0,0.5,"Female LFPR (%)"),
          "male_lfpr_total":(50.0,90.0,0.5,"Male LFPR (%)"),
          "gender_lfpr_gap":(10.0,70.0,0.5,"Gender LFPR Gap (pp)"),
          "sanitation_pct":(20.0,100.0,0.5,"Sanitation Access (%)"),
          "rural_pop_pct":(2.0,100.0,1.0,"Rural Population (%)"),
          "infra_spend_pc":(2000,25000,100,"Infra Spend/cap (₹)"),
          "sdg1_score":(20.0,100.0,1.0,"SDG-1 Score")}

    sim_vals={}
    s1,s2=st.columns(2)
    for i,feat in enumerate(sel_feats):
        col=s1 if i%2==0 else s2
        mn,mx,step,label=META.get(feat,(0.0,100.0,0.5,feat))
        defv=float(base_row[feat]) if base_row is not None and pd.notna(base_row.get(feat)) else float((mn+mx)/2)
        defv=max(float(mn),min(float(mx),defv))
        sim_vals[feat]=col.slider(label,float(mn),float(mx),defv,step=float(step))

    if st.button("🔮 Predict MPI",type="primary",use_container_width=True):
        inp=np.array([[sim_vals[f] for f in sel_feats]])
        inp_sc=sm["scaler"].transform(inp)
        pred=float(sm["model"].predict(inp_sc)[0])
        pred=max(0.001,min(0.300,pred))
        if pred<0.02: sev,sc2="Very Low Poverty",TEAL
        elif pred<0.05: sev,sc2="Low Poverty","#4ade80"
        elif pred<0.09: sev,sc2="Moderate Poverty",AMBER
        elif pred<0.13: sev,sc2="High Poverty",ROSE
        else: sev,sc2="Very High Poverty","#dc2626"

        st.markdown(f"<div class='pr'><div class='pn' style='color:{sc2};'>{pred:.4f}</div><div class='pl'>Predicted MPI Value</div><div style='margin-top:10px;font-family:IBM Plex Mono,monospace;font-size:0.9rem;color:{sc2};font-weight:600;'>{sev}</div><div style='margin-top:6px;color:#64748b;font-size:0.78rem;'>National avg: 0.069 · Bihar (worst): 0.154 · Kerala (best): 0.003</div></div>",unsafe_allow_html=True)

        fig,ax=plt.subplots(figsize=(8,1.2))
        ax.barh([0],[0.154],color=MUTED,height=0.5)
        ax.barh([0],[pred],color=sc2,height=0.5,alpha=0.9)
        for val,lbl,c in [(0.003,"Kerala",TEAL),(0.069,"Avg",AMBER),(0.154,"Bihar",ROSE)]:
            ax.axvline(val,color=c,lw=1.2,ls="--"); ax.text(val,0.32,lbl,color=c,fontsize=7.5,ha="center")
        ax.scatter([pred],[0],color=sc2,s=100,zorder=5)
        ax.set_xlim(0,0.16); ax.set_yticks([]); ax.set_xlabel("MPI Scale",fontsize=8)
        ax.set_title("Your profile on the MPI scale",color=TEAL,fontsize=9); ax.grid(axis="x",alpha=0.3)
        st.pyplot(fig); plt.close()

        yt2=active["y"].values; st2=active["states"].values
        closest=st2[np.abs(yt2-pred).argmin()]; closest_mpi=yt2[np.abs(yt2-pred).argmin()]
        st.markdown(f"<div class='ib'><strong>Closest real state:</strong> {closest} (MPI={closest_mpi:.4f})</div>",unsafe_allow_html=True)


# ── TAB 4: DIAGNOSTICS ────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## Model Diagnostics")
    dn=st.selectbox("Model",list(model_results.keys()),key="dn")
    dr2=model_results[dn]
    yt=dr2["y"].values; yp=dr2["y_pred"]; res=yt-yp

    dc1,dc2,dc3=st.columns(3)
    with dc1:
        st.markdown("**Q-Q Plot (Residuals)**")
        fig,ax=plt.subplots(figsize=(5,4))
        (osm,osr),(slope,intercept,_)=scipy_stats.probplot(res,dist="norm")
        ax.scatter(osm,osr,color=TEAL,s=40,alpha=0.85,edgecolors="white",lw=0.4)
        ax.plot(osm,slope*np.array(osm)+intercept,color=AMBER,lw=1.5)
        ax.set_xlabel("Theoretical Quantiles",fontsize=8); ax.set_ylabel("Sample Quantiles",fontsize=8)
        ax.set_title("Normal Q-Q",color=TEAL,fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with dc2:
        st.markdown("**Scale-Location**")
        fig,ax=plt.subplots(figsize=(5,4))
        ax.scatter(yp,np.sqrt(np.abs(res)),color=LAV,s=40,alpha=0.85,edgecolors="white",lw=0.4)
        ax.set_xlabel("Fitted",fontsize=8); ax.set_ylabel("√|Residuals|",fontsize=8)
        ax.set_title("Scale-Location",color=LAV,fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with dc3:
        st.markdown("**Residuals vs Fitted**")
        fig,ax=plt.subplots(figsize=(5,4))
        ax.scatter(yp,res,color=AMBER,s=40,alpha=0.85,edgecolors="white",lw=0.4)
        ax.axhline(0,color="#e2e8f0",lw=1.2,ls="--",alpha=0.6)
        ax.axhline(2*res.std(),color=ROSE,lw=1,ls=":",alpha=0.7,label="±2 SD")
        ax.axhline(-2*res.std(),color=ROSE,lw=1,ls=":",alpha=0.7)
        ax.set_xlabel("Fitted",fontsize=8); ax.set_ylabel("Residuals",fontsize=8)
        ax.set_title("Residuals vs Fitted",color=AMBER,fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    sw_stat,sw_p=scipy_stats.shapiro(res)
    st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)
    sm1,sm2=st.columns(2)
    with sm1:
        st.markdown("### Regression Summary")
        st.markdown(f"""| Metric|Value|\n|---|---|\n|R² (Train)|`{dr2['r2']:.4f}`|\n|R² (CV)|`{dr2['cv_r2']:.4f}`|\n|MAE|`{dr2['mae']:.6f}`|\n|RMSE|`{dr2['rmse']:.6f}`|\n|CV MAE|`{dr2['cv_mae']:.6f}`|\n|Shapiro-Wilk|W=`{sw_stat:.4f}` p=`{sw_p:.4f}`|\n|Normality|`{"✅ Normal" if sw_p>0.05 else "⚠ Non-normal"}`|\n|N|`{len(yt)}`|\n|Features|`{len(sel_feats)}`|""")
    with sm2:
        if dn=="🥇 Linear Regression":
            st.markdown("**Coefficient Table**")
            ct=pd.DataFrame({"Variable":sel_feats,"Coefficient":dr2["model"].coef_.round(6),"Abs Impact":np.abs(dr2["model"].coef_).round(6)}).sort_values("Abs Impact",ascending=False)
            st.dataframe(ct,use_container_width=True)
        else:
            fig,ax=plt.subplots(figsize=(5,3.5))
            ax.hist(res,bins=12,color=TEAL,alpha=0.8,edgecolor="white")
            ax.axvline(0,color=AMBER,lw=1.5,ls="--"); ax.set_xlabel("Residual",fontsize=9)
            ax.set_title("Residual Distribution",color=TEAL,fontsize=9); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close()

    st.markdown("### Outlier States")
    ot=2*res.std()
    od=pd.DataFrame({"State":dr2["states"].values,"Actual":yt.round(4),"Predicted":yp.round(4),
                      "Residual":res.round(5),"Flag":["⚠️ Outlier" if abs(r)>ot else "✓ Normal" for r in res]}
    ).sort_values("Residual",key=abs,ascending=False)
    st.dataframe(od,use_container_width=True)


# ── TAB 5: VARIABLE GUIDE ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## Variable Guide & Conceptual Framework")
    st.markdown("<div class='ib'><strong>Project Statement:</strong> Predict MPI of Indian states using socio-economic indicators. MPI=H×A where H=headcount ratio, A=intensity of deprivation across 10 indicators in 3 dimensions (Health, Education, Living Standards).</div>",unsafe_allow_html=True)

    gdf=pd.DataFrame({
        "Variable":["mpi_value","H_headcount_pct","A_intensity_pct","shdi","education_index","health_index","income_index","female_literacy","female_sec_ger","sdg4_score","female_lfpr_total","male_lfpr_total","gender_lfpr_gap","sanitation_pct","rural_pop_pct","infra_spend_pc","sdg1_score"],
        "Type":["DV","DV-comp","DV-comp","IV (HDI)","IV (HDI)","IV (HDI)","IV (HDI)","IV (Edu)","IV (Edu)","IV (Edu)","IV (Labour)","IV (Labour)","IV (Labour)","IV (Infra)","IV (Infra)","IV (Infra)","IV (SDG)"],
        "Description":["Global MPI value — PRIMARY TARGET","% population multidimensionally poor","Avg intensity of deprivation among poor (%)","Sub-national HDI (GDL 2021)","Education sub-index from GDL","Health sub-index from GDL (life expectancy)","Income sub-index (per-capita GNI proxy)","Female adult literacy rate (%)","Female secondary GER (%)","SDG-4 state score (NITI Aayog 2023-24)","Female LFPR total (%)","Male LFPR total (%)","Male–female LFPR gap (pp)","HH with improved sanitation (%)","Rural pop as % of total","Per-capita infra expenditure (₹)","SDG-1 state score (NITI Aayog 2023-24)"],
        "Expected Effect":["—","Positive","Positive","Negative ↓","Negative ↓","Negative ↓","Negative ↓","Negative ↓","Negative ↓","Negative ↓","Ambiguous","Negative ↓","Positive ↑","Negative ↓","Positive ↑","Negative ↓","Negative ↓"],
    })
    def stype(v):
        if "DV" in v: return "background-color:#0d3a30;color:#00d4aa"
        if "HDI" in v: return "background-color:#1e1a3a;color:#818cf8"
        if "Edu" in v: return "background-color:#1a2a1a;color:#4ade80"
        if "Labour" in v: return "background-color:#2a1a1a;color:#fb923c"
        return "background-color:#1a1a2a;color:#94a3b8"
    st.dataframe(gdf.style.applymap(stype,subset=["Type"]),use_container_width=True,height=520)

    st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)
    st.markdown("""
### Model Hierarchy
|Rank|Model|Why Use It|
|---|---|---|
|🥇|**Linear Regression**|Baseline; interpretable β; direction & magnitude per IV|
|🥈|**Random Forest**|Non-linearity; Gini importance; robust to outliers|
|🥉|**Gradient Boosting**|Highest accuracy; complex interactions; industry-grade|
|🔧|**Ridge Regression**|Handles multicollinearity among correlated HDI sub-indices|

### Key Research Findings
1. **Education** (shdi, female_literacy, female_sec_ger) are strongest negative predictors of MPI
2. **Sanitation access** has strong negative correlation with MPI (r ≈ −0.74)
3. **Rural population %** has positive effect — higher rurality → higher MPI
4. **Infrastructure spending** alone is not significant once education is controlled for

### Data Sources
|Source|Variables|Year|
|---|---|---|
|OPHI Global MPI|mpi_value, H%, A%|2019–21 (DHS)|
|Global Data Lab|shdi, education_index, health_index, income_index|2021|
|NITI Aayog SDG Index|sdg1_score, sdg4_score|2023-24|
|MoSPI PLFS|female/male lfpr|2022-23|
|NFHS-5|sanitation_pct, female_literacy|2019-21|
""")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""<div style='margin-top:40px;padding:20px;border-top:1px solid #1e2d47;
text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#334155;'>
MPI Lens · OPHI Global MPI 2024 (DHS 2019–21) · GDL Subnational HDI · PLFS/ILO · SDG India Index · Built with Streamlit · Academic use only
</div>""",unsafe_allow_html=True)
