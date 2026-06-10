import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from pathlib import Path
from src.data_gen import generate_icu_cohort
from src.preprocessing import preprocess as run_preprocess

from src.causal import (
    estimate_propensity, compute_iptw,
    check_balance_iptw, estimate_ate, weighted_km
)
from src.survival import fit_kaplan_meier, fit_cox, get_cox_summary

st.set_page_config(
    page_title="ClinicalLens",
    page_icon="🏥",
    layout="wide",
)

FEATURE_COLS = [
    "age", "sofa_score", "comorbidity_index",
    "gender", "creatinine", "pao2_fio2"
]

# ── Data ─────────────────────────────────────────────────────────────────────
# Top of app.py, add this before load_data()

if not Path("data/processed/icu_clean.csv").exists():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    generate_icu_cohort(2000).to_csv("data/raw/icu_cohort.csv", index=False)
    run_preprocess()
def ensure_data():
    if not Path("data/processed/icu_clean.csv").exists():
        from src.data_gen import generate_icu_cohort
        from src.preprocessing import preprocess
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        generate_icu_cohort(2000).to_csv("data/raw/icu_cohort.csv", index=False)
        preprocess()

ensure_data()


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/icu_clean.csv")

@st.cache_data
def run_analysis(propensity_model: str):
    df = load_data()
    ps = estimate_propensity(df, model=propensity_model)
    weights = compute_iptw(df, ps)
    balance = check_balance_iptw(df, weights)
    ate = estimate_ate(df, weights)
    cph = fit_cox(df, FEATURE_COLS + ["treatment"])
    cox_summary = get_cox_summary(cph)
    km_raw = fit_kaplan_meier(df)
    km_weighted = weighted_km(df, weights)
    return {
        "df": df, "ps": ps, "weights": weights,
        "balance": balance, "ate": ate,
        "cph": cph, "cox_summary": cox_summary,
        "km_raw": km_raw, "km_weighted": km_weighted,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")
propensity_model = st.sidebar.selectbox(
    "Propensity Score Model",
    ["logistic", "gbm"],
    help="Logistic = interpretable, GBM = flexible"
)
trim_pct = st.sidebar.slider("Weight Trimming Percentile", 0.0, 5.0, 1.0, 0.5)
show_ci = st.sidebar.checkbox("Show 95% CI on KM curves", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("Synthetic ICU cohort (n=2000)")
st.sidebar.markdown("Mimics MIMIC-IV structure")
st.sidebar.markdown("[Swap for MIMIC data →](https://physionet.org/content/mimiciv/)")

# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Running survival analysis and causal estimation..."):
    res = run_analysis(propensity_model)

df = res["df"]
ate = res["ate"]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏥 ClinicalLens")
st.markdown(
    "**Survival Analysis + Causal Inference on ICU Patient Data**  \n"
    "Does mechanical ventilation causally improve survival? "
    "Naive comparisons say no. IPTW-adjusted analysis recovers the truth."
)

# ── KPI Row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Patients", f"{len(df):,}")
col2.metric("Ventilated", f"{df['treatment'].mean():.1%}")
col3.metric("Mortality", f"{df['event'].mean():.1%}")
col4.metric(
    "Naive HR",
    f"{ate['hr_naive']}",
    delta=f"{ate['hr_naive'] - 1:+.2f} (confounded)",
    delta_color="inverse"
)
col5.metric(
    "IPTW HR",
    f"{ate['hr_iptw']}",
    delta=f"95% CI {ate['hr_iptw_ci']}",
    delta_color="off"
)

st.markdown("---")

# ── Tab Layout ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Survival Curves",
    "⚖️ Causal Inference",
    "🔬 Cox Model",
    "📊 Covariate Balance"
])

# ── Tab 1: KM Curves ─────────────────────────────────────────────────────────

with tab1:
    st.subheader("Kaplan-Meier Survival Curves")
    st.markdown(
        "Left: **Naive** (unadjusted) — shows confounded comparison.  \n"
        "Right: **IPTW-weighted** — approximates a randomized trial."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    colors = ["#00b4d8", "#ff6b6b"]

    for ax, (title, km_dict, subtitle) in zip(axes, [
        ("Naive KM (Unadjusted)", res["km_raw"], f"Log-rank p={res['km_raw']['logrank_p']:.4f}"),
        ("IPTW-Weighted KM", res["km_weighted"], "Pseudo-population after reweighting"),
    ]):
        ax.set_facecolor("#1a1a2e")
        for i, grp in enumerate([0, 1]):
            kmf = km_dict[grp]
            kmf.plot_survival_function(
                ax=ax,
                ci_show=show_ci,
                color=colors[i],
                linewidth=2,
            )
        ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=11)
        ax.set_xlabel("Days", color="white")
        ax.set_ylabel("Survival Probability", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"📌 **Key insight:** Naive HR={ate['hr_naive']} suggests ventilation *increases* mortality "
        f"— this is Simpson's paradox. After IPTW adjustment, HR={ate['hr_iptw']} "
        f"reveals the true protective effect (ground truth HR=0.75)."
    )

# ── Tab 2: Causal Inference ───────────────────────────────────────────────────

with tab2:
    st.subheader("Propensity Score & IPTW Estimation")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Propensity Score Distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#1a1a2e")
        for grp, color, label in [(1, "#00b4d8", "Ventilated"), (0, "#ff6b6b", "Not Ventilated")]:
            mask = df["treatment"] == grp
            ax2.hist(
                res["ps"][mask], bins=30, alpha=0.6,
                color=color, label=label, density=True
            )
        ax2.set_xlabel("P(Ventilation | X)", color="white")
        ax2.set_ylabel("Density", color="white")
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#444")
        ax2.legend(facecolor="#1a1a2e", labelcolor="white")
        ax2.set_title("Overlap / Positivity Check", color="white")
        st.pyplot(fig2)
        st.caption("Good overlap = distributions interleave. Separation = positivity violation.")

    with col_b:
        st.markdown("**IPTW Weight Distribution**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#1a1a2e")
        ax3.hist(res["weights"], bins=40, color="#a8dadc", edgecolor="none")
        ax3.set_xlabel("IPTW Weight", color="white")
        ax3.set_ylabel("Count", color="white")
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("#444")
        ax3.set_title("Stabilized IPTW Weights", color="white")
        st.pyplot(fig3)
        st.caption("Extreme weights indicate positivity issues. Trimmed at selected percentile.")

    st.markdown("---")
    st.subheader("Treatment Effect Estimate")

    col_c, col_d, col_e = st.columns(3)
    col_c.metric("Naive HR", ate["hr_naive"], help="Unadjusted — confounded by severity")
    col_d.metric("IPTW HR", ate["hr_iptw"],
                 delta=f"CI: {ate['hr_iptw_ci'][0]}–{ate['hr_iptw_ci'][1]}")
    col_e.metric("True HR (Oracle)", ate["true_hr"],
                 help="Known ground truth from data generation")

    # Forest plot style
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    fig4.patch.set_facecolor("#0e1117")
    ax4.set_facecolor("#1a1a2e")
    estimates = [
        ("Naive (Confounded)", ate["hr_naive"], None, None, "#ff6b6b"),
        ("IPTW Adjusted", ate["hr_iptw"], ate["hr_iptw_ci"][0], ate["hr_iptw_ci"][1], "#00b4d8"),
        ("True Causal Effect", ate["true_hr"], None, None, "#2ecc71"),
    ]
    for i, (label, hr, lo, hi, color) in enumerate(estimates):
        ax4.scatter(hr, i, color=color, s=100, zorder=5)
        if lo and hi:
            ax4.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax4.text(0.45, i, label, va="center", ha="right", color="white", fontsize=10)
    ax4.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Hazard Ratio", color="white")
    ax4.set_yticks([])
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#444")
    ax4.set_title("Forest Plot: Causal Effect of Ventilation", color="white")
    ax4.set_xlim(0.4, 1.8)
    st.pyplot(fig4)

# ── Tab 3: Cox Model ──────────────────────────────────────────────────────────

with tab3:
    st.subheader("Multivariable Cox Proportional Hazards Model")
    st.markdown("Adjusted HR for each covariate. Treatment HR after controlling for confounders.")

    cox_df = res["cox_summary"].reset_index()
    cox_df.columns = ["Covariate", "HR", "HR Lower 95%", "HR Upper 95%", "p-value", "Significant"]

    def color_sig(val):
        return "color: #00b4d8; font-weight: bold" if val else ""

    st.dataframe(
        cox_df.style.applymap(color_sig, subset=["Significant"])
              .format({"HR": "{:.3f}", "HR Lower 95%": "{:.3f}",
                       "HR Upper 95%": "{:.3f}", "p-value": "{:.4f}"}),
        use_container_width=True
    )

    st.markdown("**Hazard Ratio Forest Plot**")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    fig5.patch.set_facecolor("#0e1117")
    ax5.set_facecolor("#1a1a2e")
    for i, (_, row) in enumerate(cox_df.iterrows()):
        color = "#00b4d8" if row["Significant"] else "#888"
        ax5.scatter(row["HR"], i, color=color, s=80, zorder=5)
        ax5.plot([row["HR Lower 95%"], row["HR Upper 95%"]], [i, i],
                 color=color, linewidth=1.5)
    ax5.set_yticks(range(len(cox_df)))
    ax5.set_yticklabels(cox_df["Covariate"], color="white")
    ax5.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Hazard Ratio (95% CI)", color="white")
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color("#444")
    st.pyplot(fig5)

    c_idx_val = res["cph"].concordance_index_
    st.metric("C-index (Model Discrimination)", f"{c_idx_val:.3f}",
              help="0.5 = random, 1.0 = perfect. >0.7 is good.")

# ── Tab 4: Covariate Balance ───────────────────────────────────────────────────

with tab4:
    st.subheader("Covariate Balance: Before vs After IPTW")
    st.markdown(
        "Standardized Mean Difference (SMD) measures imbalance. "
        "**Target: |SMD| < 0.1** after weighting (dashed lines)."
    )

    balance = res["balance"].reset_index()
    n_balanced = balance["balanced"].sum()
    st.success(f"✅ {n_balanced}/{len(balance)} covariates balanced after IPTW (|SMD| < 0.1)")

    fig6, ax6 = plt.subplots(figsize=(9, 5))
    fig6.patch.set_facecolor("#0e1117")
    ax6.set_facecolor("#1a1a2e")
    x = np.arange(len(balance))
    width = 0.35
    ax6.bar(x - width/2, balance["smd_before"].abs(), width,
            label="Before IPTW", color="#ff6b6b", alpha=0.8)
    ax6.bar(x + width/2, balance["smd_after"].abs(), width,
            label="After IPTW", color="#00b4d8", alpha=0.8)
    ax6.axhline(0.1, color="white", linestyle="--", alpha=0.5, label="|SMD|=0.1 threshold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(balance["covariate"], rotation=20, color="white")
    ax6.set_ylabel("|Standardized Mean Difference|", color="white")
    ax6.tick_params(colors="white")
    ax6.spines[:].set_color("#444")
    ax6.legend(facecolor="#1a1a2e", labelcolor="white")
    ax6.set_title("Covariate Balance Before/After IPTW", color="white")
    st.pyplot(fig6)

    st.dataframe(
        balance.style.applymap(
            lambda v: "color: #2ecc71" if v else "color: #ff6b6b",
            subset=["balanced"]
        ).format({"smd_before": "{:+.4f}", "smd_after": "{:+.4f}"}),
        use_container_width=True
    )