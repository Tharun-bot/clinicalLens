import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_icu_cohort(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic ICU cohort mimicking MIMIC-IV structure.
    
    Confounders  → age, severity score (SOFA), comorbidities
    Treatment    → mechanical ventilation (binary)
    Outcome      → survival time (days), event flag (death=1)
    
    Treatment assignment is NOT random — sicker patients get ventilated more.
    This is the confounding we'll correct with IPTW.
    """

    age = np.random.normal(62, 15, n).clip(18, 95)
    sofa_score = np.random.normal(8, 3, n).clip(0, 24)          # ICU severity
    comorbidity_index = np.random.poisson(2, n).clip(0, 8)       # Charlson index proxy
    gender = np.random.binomial(1, 0.55, n)                      # 1 = male
    creatinine = np.random.lognormal(0.3, 0.5, n).clip(0.4, 15) # kidney function
    pao2_fio2 = np.random.normal(250, 80, n).clip(50, 500)       # lung function

    # Treatment propensity — sicker patients more likely to be ventilated (confounding)
    log_odds = (
        -2.5
        + 0.03 * (sofa_score - 8)
        + 0.02 * (age - 62)
        + 0.15 * comorbidity_index
        - 0.005 * (pao2_fio2 - 250)
        + 0.1 * creatinine
    )
    propensity = 1 / (1 + np.exp(-log_odds))
    treatment = np.random.binomial(1, propensity, n)  # 1 = ventilated

    # True treatment effect: ventilation reduces hazard by ~25% for eligible patients
    # But naive comparison will show ventilation looks harmful (confounding)
    baseline_hazard = (
        0.02
        + 0.003 * (age - 62).clip(0)
        + 0.008 * sofa_score
        + 0.01 * comorbidity_index
        + 0.002 * creatinine
    )
    treatment_effect = np.where(treatment == 1, 0.75, 1.0)  # true causal effect
    hazard = baseline_hazard * treatment_effect

    # Survival time from exponential distribution
    survival_time = np.random.exponential(1 / hazard).clip(0.5, 365)

    # Censoring — ~30% of patients are discharged alive (censored)
    censoring_time = np.random.uniform(30, 365, n)
    observed_time = np.minimum(survival_time, censoring_time)
    event = (survival_time <= censoring_time).astype(int)

    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "age": age.round(1),
        "sofa_score": sofa_score.round(1),
        "comorbidity_index": comorbidity_index,
        "gender": gender,
        "creatinine": creatinine.round(2),
        "pao2_fio2": pao2_fio2.round(1),
        "propensity_true": propensity.round(4),   # ground truth — used for validation only
        "treatment": treatment,
        "observed_time": observed_time.round(2),
        "event": event,
    })

    return df


if __name__ == "__main__":
    df = generate_icu_cohort(2000)
    out = Path("data/raw/icu_cohort.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} patients")
    print(f"Ventilated: {df['treatment'].mean():.1%}")
    print(f"Mortality: {df['event'].mean():.1%}")
    print(f"Median survival: {df['observed_time'].median():.1f} days")
    print(df.head())