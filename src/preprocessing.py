import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "age", "sofa_score", "comorbidity_index",
    "gender", "creatinine", "pao2_fio2"
]
TREATMENT_COL = "treatment"
TIME_COL = "observed_time"
EVENT_COL = "event"


def load_raw(path: str = "data/raw/icu_cohort.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def check_overlap(df: pd.DataFrame) -> dict:
    """
    Positivity check — both treated/untreated must exist across covariate space.
    Returns basic overlap diagnostics.
    """
    stats = {}
    for col in FEATURE_COLS:
        t1 = df[df[TREATMENT_COL] == 1][col]
        t0 = df[df[TREATMENT_COL] == 0][col]
        stats[col] = {
            "treated_mean": t1.mean(),
            "control_mean": t0.mean(),
            "std_diff": (t1.mean() - t0.mean()) / np.sqrt(
                (t1.std() ** 2 + t0.std() ** 2) / 2
            ),
        }
    return stats


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 75, 100],
        labels=["<40", "40-60", "60-75", "75+"]
    )
    df["high_severity"] = (df["sofa_score"] >= 10).astype(int)
    df["renal_failure"] = (df["creatinine"] >= 2.0).astype(int)
    df["respiratory_failure"] = (df["pao2_fio2"] < 200).astype(int)
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df, scaler


def preprocess(raw_path: str = "data/raw/icu_cohort.csv",
               out_path: str = "data/processed/icu_clean.csv") -> pd.DataFrame:
    df = load_raw(raw_path)

    # Sanity checks
    assert df[TIME_COL].min() > 0, "Survival times must be positive"
    assert df[EVENT_COL].isin([0, 1]).all(), "Event must be binary"
    assert df[TREATMENT_COL].isin([0, 1]).all(), "Treatment must be binary"

    df = add_derived_features(df)

    # Check overlap before scaling (use raw values)
    overlap = check_overlap(df)
    print("\n--- Covariate Balance (Standardized Differences, pre-IPTW) ---")
    for col, s in overlap.items():
        flag = "⚠" if abs(s["std_diff"]) > 0.1 else "✓"
        print(f"{flag} {col:25s} std_diff={s['std_diff']:+.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    preprocess()