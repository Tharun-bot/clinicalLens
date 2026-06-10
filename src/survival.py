import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from pathlib import Path


def fit_kaplan_meier(
    df: pd.DataFrame,
    group_col: str = "treatment",
    time_col: str = "observed_time",
    event_col: str = "event",
    group_labels: dict = {0: "No Ventilation", 1: "Ventilated"},
) -> dict:
    """Fit KM curves per group, run log-rank test."""
    results = {}
    for grp in df[group_col].unique():
        subset = df[df[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset[time_col],
            subset[event_col],
            label=group_labels.get(grp, str(grp))
        )
        results[grp] = kmf

    # Log-rank test
    g0 = df[df[group_col] == 0]
    g1 = df[df[group_col] == 1]
    lr = logrank_test(
        g0[time_col], g1[time_col],
        g0[event_col], g1[event_col]
    )
    results["logrank_p"] = lr.p_value
    results["logrank_stat"] = lr.test_statistic
    return results


def fit_cox(
    df: pd.DataFrame,
    covariates: list,
    time_col: str = "observed_time",
    event_col: str = "event",
    penalizer: float = 0.1,
) -> CoxPHFitter:
    """Fit multivariable Cox PH model."""
    cph = CoxPHFitter(penalizer=penalizer)
    cols = covariates + [time_col, event_col]
    cph.fit(df[cols], duration_col=time_col, event_col=event_col)
    return cph


def get_cox_summary(cph: CoxPHFitter) -> pd.DataFrame:
    """Return hazard ratios with 95% CI, sorted by significance."""
    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["HR", "HR_lower", "HR_upper", "p_value"]
    summary["significant"] = summary["p_value"] < 0.05
    return summary.sort_values("p_value")


def evaluate_cox(cph: CoxPHFitter, df: pd.DataFrame,
                 covariates: list,
                 time_col: str = "observed_time",
                 event_col: str = "event") -> float:
    cols = covariates + [time_col, event_col]
    c_index = concordance_index(
        df[time_col],
        -cph.predict_partial_hazard(df[cols]),
        df[event_col]
    )
    return c_index


if __name__ == "__main__":
    df = pd.read_csv("data/processed/icu_clean.csv")

    COVARIATES = [
        "age", "sofa_score", "comorbidity_index",
        "gender", "creatinine", "pao2_fio2", "treatment"
    ]

    print("=== Kaplan-Meier ===")
    km_results = fit_kaplan_meier(df)
    print(f"Log-rank p-value: {km_results['logrank_p']:.4f}")
    print(f"(Naive comparison — confounded)")

    print("\n=== Cox PH Model ===")
    cph = fit_cox(df, COVARIATES)
    cph.print_summary()

    summary = get_cox_summary(cph)
    print(summary)

    c_index = evaluate_cox(cph, df, COVARIATES)
    print(f"\nC-index (discrimination): {c_index:.3f}")