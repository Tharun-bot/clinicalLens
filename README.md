# ClinicalLens — Survival Analysis + Causal Inference on ICU Data

## What this demonstrates
- Survival analysis (Kaplan-Meier, Cox PH) on clinical EHR data
- Causal inference via Inverse Probability of Treatment Weighting (IPTW)
- Correcting for confounding bias (Simpson's Paradox in ICU ventilation data)
- Interactive clinical dashboard built with Streamlit

## Key result
Naive comparison: HR=~1.35 (ventilation appears harmful — confounding)  
IPTW-adjusted: HR=~0.75 (true protective effect recovered)

## Stack
Python · Lifelines · scikit-learn · Streamlit · Pandas

## To run locally
```bash
pip install -r requirements.txt
python src/data_gen.py
python src/preprocessing.py
streamlit run app/app.py
```

## Swapping in real MIMIC-IV data
Replace `data_gen.py` output with MIMIC-IV's `icustays`, `patients`, `chartevents` tables.
Column mapping in `preprocessing.py` covers the schema translation.