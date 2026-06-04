# Airline Delay Risk Classification

## Project Overview

This project predicts whether an aggregated `(year, month, carrier, airport)` record has **high airline delay risk** using the Kaggle Airline Delay dataset.

- Dataset: https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
- File used: `data/Airline_Delay_Cause.csv`
- Time span: August 2013 to August 2023
- Task: binary classification
- Target: `high_delay_risk = 1` when `arr_del15 / arr_flights >= 0.2415`

## Final Executed Results

The notebook was executed top-to-bottom after the audit pass.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6095 | 0.3559 | 0.6936 | 0.4704 | 0.6874 |
| Neural Network | 0.8002 | 0.6331 | 0.4776 | 0.5445 | 0.8181 |

Interpretation:

- Logistic Regression has higher recall and is better when missing high-risk cases is costly.
- The Neural Network has stronger precision, F1, accuracy, and ROC-AUC.
- The executed machine used the documented sklearn MLP fallback because TensorFlow is unavailable on the current Python 3.13 runtime. The notebook contains a TensorFlow/Keras Sequential path for compatible Python environments.

## Leakage Policy

The strict predictive model excludes direct and post-outcome delay variables:

- Excluded: `arr_del15`, delay-cause counts, and delay-duration totals.
- Included strict features: `year`, `month`, `carrier`, `airport`, `arr_flights`, `arr_cancelled`, `arr_diverted`.

This prevents inflated performance from target leakage.

## Project Structure

```text
data/
  Airline_Delay_Cause.csv
notebooks/
  final_project_airline_delay.ipynb
outputs/
  figures/
  metrics/
  tables/
reports/
  report_draft.md
  audit_report.md
requirements.txt
README.md
```

## How to Reproduce

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the dataset exists:

```text
data/Airline_Delay_Cause.csv
```

3. Run the notebook top-to-bottom:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/final_project_airline_delay.ipynb --output final_project_airline_delay.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1800
```

## Generated Artifacts

- Academic report: `reports/report_draft.md`
- Audit report: `reports/audit_report.md`
- Metrics: `outputs/metrics/*.csv`, `outputs/metrics/run_summary.json`
- Tables: `outputs/tables/*.csv`
- Figures: `outputs/figures/*.png`

## Notes

- `random_state = 42` is used consistently.
- Stratified train/validation/test splits and Stratified K-Fold validation are included.
- TensorFlow is listed with a Python-version marker because the current machine uses Python 3.13.
