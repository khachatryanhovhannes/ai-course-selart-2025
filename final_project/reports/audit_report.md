# Project Audit Report

## Audit Scope

Reviewed and improved:

- notebook execution and structure
- report length and academic coverage
- README and requirements
- figures, tables, metrics, and outputs
- preprocessing, feature engineering, model training, tuning, validation, evaluation, and reproducibility

## Issues Found

1. The report was far below the requested 5000-6000 word academic standard.
2. Literature review was too thin and did not discuss flight delay prediction studies in enough detail.
3. Logistic Regression tuning was narrow and did not evaluate penalty or class-weight alternatives.
4. Neural network tuning was missing systematic architecture and learning-rate comparison.
5. TensorFlow/Keras was attempted but not fully documented against the actual runtime limitation.
6. Cross-validation artifacts were incomplete.
7. Train, validation, and test metrics were not exported together for overfitting analysis.
8. Neural network feature importance was missing.
9. Report conclusions were too short and did not discuss operational value deeply enough.
10. README did not fully reflect the improved audit-grade workflow and final executed metrics.
11. Requirements did not include `nbconvert`, `ipykernel`, or TensorFlow compatibility notes.
12. Missing artifact tables for data splits, hyperparameter tuning, CV summary, and neural permutation importance.

## Fixes Applied

1. Rebuilt the notebook in place with a complete end-to-end workflow.
2. Re-ran the notebook top-to-bottom successfully with `python -m jupyter nbconvert`.
3. Added schema validation, defensive cleaning, and explicit leakage-excluded column tracking.
4. Added threshold analysis and class-balance export.
5. Added improved EDA figures with readable titles and axis labels.
6. Expanded Logistic Regression tuning across `C`, `penalty`, and `class_weight`.
7. Added neural network configuration tuning across layers, hidden units, dropout design, and learning rate.
8. Added TensorFlow/Keras Sequential model path with dense layers, dropout, validation data, and early stopping.
9. Added sklearn MLP fallback for the current Python 3.13 runtime and recorded the backend in `run_summary.json`.
10. Added train/validation/test metrics export.
11. Added Stratified K-Fold cross-validation summary.
12. Added Logistic Regression coefficient analysis.
13. Added grouped neural permutation importance.
14. Added new tables:
    - `data_split_summary.csv`
    - `logistic_hyperparameter_tuning.csv`
    - `neural_network_hyperparameter_tuning.csv`
    - `stratified_kfold_cv_summary.csv`
    - `neural_permutation_importance.csv`
15. Expanded the academic report to 5,961 words with all required sections.
16. Added a dedicated literature review with aviation delay prediction and ML references.
17. Updated README and requirements for reproducibility.

## Final Requirement Validation

| Requirement | Status |
|---|---|
| Real-world problem | Complete |
| Target variable | Complete |
| EDA | Complete |
| Preprocessing | Complete |
| Feature engineering | Complete |
| Train/validation/test split | Complete |
| Two ML approaches | Complete |
| Classic ML model | Complete |
| Advanced/deep model | Complete with Keras path and executed MLP fallback |
| Evaluation metrics | Complete |
| Model comparison | Complete |
| Visualizations | Complete |
| Conclusions | Complete |
| References | Complete |
| Reproducibility | Complete |
| Academic writing quality | Improved |
| 5000-6000 word report | Complete, 5,961 words |

## Final Executed Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6095 | 0.3559 | 0.6936 | 0.4704 | 0.6874 |
| Neural Network | 0.8002 | 0.6331 | 0.4776 | 0.5445 | 0.8181 |

## Remaining Limitations

1. The dataset is monthly aggregate data, not flight-level data.
2. The split is stratified random; a production study should add temporal validation.
3. External predictors such as weather, route distance, schedule timing, and aircraft rotations are absent.
4. TensorFlow/Keras did not execute on this machine because TensorFlow is unavailable for the current Python 3.13 runtime. The notebook includes the Keras path for compatible environments.
5. The target threshold is policy-dependent and should be tuned with business cost assumptions in deployment.

## Submission Readiness

The project now includes an executed notebook, expanded academic report, updated README, updated requirements, new figures, new metrics, new tables, and this audit report. It is ready for university-level submission with the remaining limitations disclosed.
