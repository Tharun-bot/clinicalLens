import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter


FEATURE_COLS = [
    "age", "sofa_score", "comorbidity_index",
    "gender", "creatinine", "pao2_fio2"
]


def estimate_propensity(
    df: pd.DataFrame,
    features: list = FEATURE_COLS,
    treatment_col: str = "treatment",
    model: str = "logistic",  # "logistic" | "gbm"
) -> pd.Series:
    """
    Estimate P(T=1 | X) — propensity score.
    Uses logistic regression (transparent) or GBM (flexible).
    """
    X = df[features]
    y = df[treatment_col]

    if model == "logistic":
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        base = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf = CalibratedClassifierCV(base, cv=5)

    clf.fit(X, y)
    ps = clf.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, ps)
    print(f"Propensity model AUC: {auc:.3f}")
    return pd.Series(ps, index=df.index, name="propensity_score")


def compute_iptw(
    df: pd.DataFrame,
    ps: pd.Series,
    treatment_col: str = "treatment",
    stabilized: bool = True,
    trim_percentile: float = 1.0,
) -> pd.Series:
    """
    Compute stabilized IPTW weights.
    
    Stabilized: w = P(T) / P(T|X)  — reduces variance vs 1/P(T|X)
    Trimmed at 1st/99th percentile to handle positivity violations.
    """
    t = df[treatment_col]
    p_t1 = t.mean()  # marginal P(T=1)
    p_t0 = 1 - p_t1

    if stabilized:
        weights = np.where(
            t == 1,
            p_t1 / ps,
            p_t0 / (1 - ps)
        )
    else:
        weights = np.where(t == 1, 1 / ps, 1 / (1 - ps))

    # Trim extreme weights
    lower = np.percentile(weights, trim_percentile)
    upper = np.percentile(weights, 100 - trim_percentile)
    weights = np.clip(weights, lower, upper)

    weights = pd.Series(weights, index=df.index, name="iptw")
    print(f"IPTW weights — mean: {weights.mean():.3f}, "
          f"max: {weights.max():.3f}, "
          f"effective N: {(weights.sum()**2 / (weights**2).sum()):.0f}")
    return weights


def check_balance_iptw(
    df: pd.DataFrame,
    weights: pd.Series,
    features: list = FEATURE_COLS,
    treatment_col: str = "treatment",
) -> pd.DataFrame:
    """
    Compute standardized mean differences before and after IPTW.
    Target: |SMD| < 0.1 after weighting.
    """
    rows = []
    for col in features:
        t1 = df[df[treatment_col] == 1][col]
        t0 = df[df[treatment_col] == 0][col]
        w1 = weights[df[treatment_col] == 1]
        w0 = weights[df[treatment_col] == 0]

        pooled_std = np.sqrt((t1.std() ** 2 + t0.std() ** 2) / 2)

        smd_before = (t1.mean() - t0.mean()) / pooled_std

        mean_t1_w = np.average(t1, weights=w1)
        mean_t0_w = np.average(t0, weights=w0)
        smd_after = (mean_t1_w - mean_t0_w) / pooled_std

        rows.append({
            "covariate": col,
            "smd_before": round(smd_before, 4),
            "smd_after": round(smd_after, 4),
            "balanced": abs(smd_after) < 0.1,
        })

    return pd.DataFrame(rows).set_index("covariate")


def estimate_ate(
    df: pd.DataFrame,
    weights: pd.Series,
    time_col: str = "observed_time",
    event_col: str = "event",
    treatment_col: str = "treatment",
) -> dict:
    """
    Estimate Average Treatment Effect using weighted Cox model.
    Only treatment variable in the model — IPTW handles confounding.
    """
    cph = CoxPHFitter()
    tmp = df[[treatment_col, time_col, event_col]].copy()
    cph.fit(
        tmp,
        duration_col=time_col,
        event_col=event_col,
        weights_col=None,
        robust=True,
        # lifelines accepts sample_weight as a separate arg
    )
    # Pass weights properly
    cph_w = CoxPHFitter()
    tmp["_w"] = weights.values
    cph_w.fit(
        tmp,
        duration_col=time_col,
        event_col=event_col,
        weights_col="_w",
        robust=True,
    )

    hr_naive = float(
        CoxPHFitter()
        .fit(df[[treatment_col, time_col, event_col]], time_col, event_col)
        .summary["exp(coef)"][treatment_col]
    )

    hr_iptw = float(cph_w.summary["exp(coef)"][treatment_col])
    hr_iptw_lower = float(cph_w.summary["exp(coef) lower 95%"][treatment_col])
    hr_iptw_upper = float(cph_w.summary["exp(coef) upper 95%"][treatment_col])

    return {
        "hr_naive": round(hr_naive, 3),
        "hr_iptw": round(hr_iptw, 3),
        "hr_iptw_ci": (round(hr_iptw_lower, 3), round(hr_iptw_upper, 3)),
        "true_hr": 0.75,  # ground truth from data_gen.py
    }


def weighted_km(
    df: pd.DataFrame,
    weights: pd.Series,
    time_col: str = "observed_time",
    event_col: str = "event",
    treatment_col: str = "treatment",
) -> dict:
    """Fit KM curves using IPTW weights."""
    results = {}
    labels = {0: "No Ventilation (IPTW)", 1: "Ventilated (IPTW)"}
    for grp in [0, 1]:
        mask = df[treatment_col] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, time_col],
            df.loc[mask, event_col],
            weights=weights[mask].values,
            label=labels[grp],
        )
        results[grp] = kmf
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/processed/icu_clean.csv")

    ps = estimate_propensity(df, model="logistic")
    weights = compute_iptw(df, ps)

    print("\n=== Covariate Balance ===")
    balance = check_balance_iptw(df, weights)
    print(balance.to_string())
    n_balanced = balance["balanced"].sum()
    print(f"\n{n_balanced}/{len(balance)} covariates balanced after IPTW")

    print("\n=== Average Treatment Effect ===")
    ate = estimate_ate(df, weights)
    print(f"Naive HR:     {ate['hr_naive']}  ← biased (confounded)")
    print(f"IPTW HR:      {ate['hr_iptw']} {ate['hr_iptw_ci']}  ← causal estimate")
    print(f"True HR:      {ate['true_hr']}  ← ground truth")