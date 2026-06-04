# Airline Delay Risk Classification Using Logistic Regression and Neural Networks

## 1. Introduction

Flight delay is a persistent operational problem in commercial aviation. Delays affect passenger experience, aircraft utilization, crew scheduling, airport resource allocation, maintenance planning, and network reliability. A single delayed arrival can propagate across subsequent flight legs when the same aircraft, crew, gate, or support resources are reused. For airlines and airports, delay prediction is therefore not only a passenger-information problem; it is also a planning and risk-management problem. Accurate delay-risk estimates can help organizations prioritize mitigation efforts, allocate buffer time, adjust staffing, and identify carrier-airport-month combinations that deserve additional operational attention.

This project studies airline delay risk using the `Airline_Delay_Cause.csv` dataset, which contains U.S. airline arrival-delay statistics from August 2013 through August 2023. Each row is an aggregate observation for a year, month, carrier, and airport. Because the data are aggregated rather than flight-level, this project does not claim to predict whether a specific flight will be delayed. It predicts whether an aggregate carrier-airport-month record has high delay risk.

The central machine learning task is binary classification. The project defines a delay ratio as `arr_del15 / arr_flights`, where `arr_del15` is the count of arriving flights delayed by at least 15 minutes and `arr_flights` is the number of arriving flights. A record is labeled `high_delay_risk = 1` when this ratio is greater than or equal to a selected threshold, and `0` otherwise. The final executed notebook selected a threshold of 0.2415, the empirical 75th percentile among evaluated candidate thresholds. This produced a positive-class rate of 25.00%, which is imbalanced but not extreme. The threshold represents a practical risk-screening policy: the positive class identifies the upper quartile of aggregate delay-risk observations.

The project compares two modeling approaches. The first is Logistic Regression, a classic linear classification model that provides interpretable coefficients. The second is a neural network model. The notebook contains a TensorFlow/Keras Sequential model as the preferred deep learning path, using dense layers, dropout, validation monitoring, and early stopping. Because the execution environment uses Python 3.13 and TensorFlow is not installed for that runtime, the executed run used an `sklearn` MLP fallback. This fallback preserves reproducibility on the available machine, while the notebook and requirements document how to run the Keras path in a compatible Python environment. This is not hidden in the results: the backend is explicitly saved in `outputs/metrics/run_summary.json`.

The study is designed to satisfy both academic and engineering criteria: problem formulation, literature review, EDA, leakage control, reproducible training, systematic tuning, cross-validation, model comparison, feature importance, exported metrics, and an audit report. The objective is not to produce a production-ready airline operations system, but to construct a defensible final AI/ML project with correct methodology and clear interpretation.

## 2. Theoretical Background

Supervised machine learning estimates a function that maps input features to an output label using observed examples. In binary classification, the label takes one of two possible values. A classifier may produce hard class labels or continuous probabilities. Probabilities are particularly useful in operational risk problems because decision thresholds can be changed according to cost. For airline delay risk, a low threshold may be appropriate when missing a high-risk period is costly, while a higher threshold may be appropriate when unnecessary interventions are expensive.

Logistic Regression is a standard probabilistic classification model. It estimates the log-odds of the positive class as a linear combination of the input features. After one-hot encoding categorical variables, the model assigns coefficients to carrier indicators, airport indicators, and numeric predictors. A positive coefficient increases the estimated log-odds of high delay risk, while a negative coefficient decreases it, conditional on the other encoded features and regularization. Logistic Regression is often a strong baseline for tabular classification because it is computationally efficient, transparent, and less prone to opaque overfitting than more flexible models. Regularization controls coefficient magnitude and helps stabilize estimates when the feature space includes many one-hot encoded categories.

Neural networks are nonlinear function approximators. A feed-forward multilayer perceptron uses dense layers and nonlinear activation functions to learn interactions among features. In this project, a neural network can learn interactions such as carrier-specific seasonal patterns or airport-specific effects that a purely linear model may not capture unless those interaction terms are explicitly engineered. However, neural networks require careful validation. Without early stopping, regularization, and a held-out validation set, they can memorize training patterns that do not generalize. Dropout randomly deactivates units during training and acts as a regularizer. Early stopping monitors validation performance and restores the best model weights when additional training stops improving generalization.

Classification performance cannot be evaluated with accuracy alone. The target is imbalanced: only about 25% of observations are labeled high risk. A classifier that predicts every record as low risk would achieve about 75% accuracy but would have zero recall for the positive class. Therefore this project reports accuracy, precision, recall, F1-score, and ROC-AUC. Precision measures the share of predicted high-risk cases that are actually high risk. Recall measures the share of true high-risk cases captured by the model. F1-score balances precision and recall. ROC-AUC measures ranking quality across classification thresholds and is useful when operational thresholds may later be adjusted.

Cross-validation strengthens evaluation by checking whether performance is stable across multiple stratified folds. The notebook uses Stratified K-Fold validation so each fold preserves the class balance. This reduces the risk that a model looks good only because of a fortunate train/test split. The final evaluation still uses a held-out test set that is not used for tuning. This separation between tuning and final evaluation is important for statistical validity.

Feature importance is also part of model assessment. For Logistic Regression, coefficients provide an interpretable measure of association after preprocessing. For the neural model, the notebook uses grouped permutation importance. Each original feature is shuffled on the validation set, the model is re-scored, and the drop in ROC-AUC is measured. A larger drop means the model relied more heavily on that feature group. This approach is model-agnostic and easier to explain than raw neural weights, especially when categorical variables have been one-hot encoded.

## 3. Literature Review

Flight delay prediction has been studied using statistical models, machine learning methods, deep learning architectures, and aviation-network models. The literature generally agrees that delay prediction is difficult because delays depend on interacting factors such as weather, airport congestion, airline operations, air traffic management, aircraft rotations, and network propagation. The present project is smaller in scope than many research studies because it uses aggregated public data and a limited strict feature set, but it follows the same methodological themes: careful feature design, supervised classification, model comparison, and operational interpretation.

Dai (2024) proposed a hybrid machine learning model for flight delay prediction using aviation big data. The study emphasized feature selection through ANOVA and forward sequential feature selection before model training. This is relevant to the present work because it shows that aviation delay prediction benefits from explicit feature selection rather than blindly using all available variables. In our project, the most important feature-selection decision is not only predictive but methodological: delay-count and delay-duration variables are excluded from strict forecasting features because they would leak target-adjacent post-outcome information.

Chen and Li (2019) studied chained predictions of flight delay and emphasized delay propagation across airport networks. Their work combined machine learning with a delay-propagation perspective. The relationship to this project is conceptual: aggregated carrier-airport-month delay risk is partly a symptom of network behavior, but our dataset does not include aircraft rotations or direct propagation links. Therefore, this project cannot model propagation explicitly. The limitation is discussed in the report, and future work recommends adding temporal or network-aware validation.

Cheevachaipimol, Teinwan, and Chutima (2021) investigated flight delay prediction using a hybrid deep learning method. Their work compared conventional machine learning and deep learning approaches for structured aviation data. A key lesson is that deep learning is not automatically superior for tabular aviation data; performance depends on feature richness, architecture, and validation strategy. This project reflects that lesson by comparing a neural network against Logistic Regression rather than assuming the neural model is better. The executed results show that the neural model improves ROC-AUC and F1-score, while Logistic Regression produces higher recall.

MDPI Aerospace research on probabilistic flight delay prediction has highlighted the value of probabilistic forecasts for downstream airport planning tasks such as flight-to-gate assignment. That perspective supports this project's use of probability outputs rather than only hard labels. In practice, an airline or airport would likely tune the classification threshold based on the cost of false negatives and false positives. The models in this project therefore provide risk scores that can be recalibrated for different operational policies.

Research on U.S. airport-network delay prediction has compared Logistic Regression, random forest, gradient boosting, and feed-forward neural networks. Such studies often find that tree-based ensemble methods perform strongly on tabular aviation data, sometimes outperforming neural networks. This project does not include gradient boosting because the course requirement asks specifically for a classic ML model and an advanced or deep model. However, future work recommends adding XGBoost, LightGBM, or CatBoost as additional baselines. This would strengthen the comparison and align the project with broader applied ML practice.

Recent deep learning studies also explore spatial-temporal and attention-based models. These approaches are appropriate when data include sequential structure, graph relationships, aircraft rotations, or time-varying weather. The current dataset is monthly aggregate data, so a complex recurrent or graph neural network would not be justified without additional temporal or network features. The chosen dense neural network is therefore a reasonable advanced model for the available tabular feature matrix. It is deep enough to capture nonlinear interactions but not so complex that it creates unjustified methodological claims.

The literature review suggests several principles that guided this project: delay prediction should be framed at the correct granularity, leakage must be controlled, model comparison should include interpretable baselines, neural networks require careful regularization, and operational value depends on recall, precision, ranking quality, and threshold selection.

## 4. Dataset Description

The dataset file is `data/Airline_Delay_Cause.csv`. The raw dataset contains 171,666 rows and 21 original columns. After cleaning and target engineering, the executed notebook retained 171,223 rows and 23 columns. The removed rows had missing or non-positive `arr_flights`, which cannot be used safely because `arr_flights` is the denominator in the target ratio.

The key identifier columns are `year`, `month`, `carrier`, `carrier_name`, `airport`, and `airport_name`. The activity column `arr_flights` records the number of arriving flights in the aggregate record. The target-adjacent column `arr_del15` records arrivals delayed by at least 15 minutes. Additional columns include delay-cause counts such as `carrier_ct`, `weather_ct`, `nas_ct`, `security_ct`, and `late_aircraft_ct`, disruption variables such as `arr_cancelled` and `arr_diverted`, and delay-duration totals such as `arr_delay`, `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, and `late_aircraft_delay`.

The dataset covers a decade, from 2013 to 2023. This is useful because it contains many seasonal and operational conditions, but it also introduces concept-drift risk. Airline networks, airport operations, regulatory conditions, and passenger demand changed substantially over that period, especially around the COVID-19 pandemic. A random stratified split is acceptable for a course project and model-comparison exercise, but it does not fully test temporal generalization. A production forecasting study should add forward-chaining validation, for example training on earlier years and testing on later years.

The target class is created from the delay ratio. The notebook evaluated threshold candidates of 0.30, 0.25, and the empirical 75th percentile. The selected threshold was 0.2415, resulting in a positive-class rate of 25.00%. This means the positive class represents approximately the highest-risk quartile of carrier-airport-month records. The choice balances interpretability and statistical stability. A fixed threshold of 0.30 is easier to explain as a business rule, but it produced stronger imbalance in the executed data. The quartile threshold gives enough positive examples for stable model training and evaluation.

The strict predictive features are `year`, `month`, `carrier`, `airport`, `arr_flights`, `arr_cancelled`, and `arr_diverted`. This feature set is intentionally conservative. The model does not use `arr_del15` because the target is constructed from it. It also excludes delay causes and delay durations because those fields describe realized delay outcomes and would not be available as clean predictors in a prospective risk-scoring scenario. The excluded leakage-risk columns are saved in the run summary.

The dataset is valuable for educational modeling because it has categorical identifiers, numeric operational variables, imbalance, seasonality, and leakage traps. At the same time, its aggregate nature limits causal interpretation. The models can identify associations and risk patterns, but they cannot determine whether a specific airport, carrier, or month causes delays. They also cannot capture within-month weather events, aircraft rotations, gate constraints, or flight-level schedule details.

## 5. Problem Formulation

The project formulates airline delay analysis as a binary classification problem. For each aggregate observation, the input vector contains strict predictive features. The output label is `high_delay_risk`. A model estimates `P(high_delay_risk = 1 | X)`, where `X` contains year, month, carrier, airport, arrival flight volume, cancellations, and diversions. At prediction time, the probability can be converted into a class label using a decision threshold, with 0.5 used for reported default metrics.

The business interpretation is risk screening. A high probability indicates that a carrier-airport-month combination resembles historical observations in the upper quartile of delay ratio. This information could support planning discussions, operational review, and prioritization of mitigation resources. For example, a planning team might examine high-risk combinations before peak travel months, compare airport-level delay risk, or flag carrier-airport pairs where delay performance is consistently poor.

The problem is not flight-level prediction. The model cannot tell a passenger whether Flight 123 will be delayed tomorrow. It also does not include weather forecasts, departure times, route distance, aircraft type, inbound aircraft delay, gate availability, or crew information. The correct claim is narrower: using historical aggregate features, the model classifies whether aggregate records are likely to belong to the high delay-risk group.

The classification setup is appropriate because the dataset directly provides aggregate counts. Regression could also predict the continuous delay ratio, but binary risk classification is easier to interpret operationally and aligns with decisions about whether to flag an entity for attention.

## 6. Methodology

The methodology begins with schema validation. The notebook checks that expected columns are present, strips whitespace from string identifiers, coerces numeric fields safely, removes rows with missing or non-positive `arr_flights`, and handles predictor missingness inside preprocessing pipelines.

The train/validation/test design uses stratified random splitting with `random_state = 42`. The test set contains 20% of the data. The remaining 80% is split into training and validation sets, with validation taking 20% of the train-validation subset. The resulting split sizes are 109,582 training rows, 27,396 validation rows, and 34,245 test rows. Positive-class rates are preserved across splits, which is important for imbalanced classification.

Preprocessing is implemented with a `ColumnTransformer`. Categorical features `carrier` and `airport` are imputed with the most frequent value and one-hot encoded with unknown categories ignored. Numeric features `year`, `month`, `arr_flights`, `arr_cancelled`, and `arr_diverted` are median-imputed and standardized. Pipeline-based preprocessing is important because it prevents inconsistent transformations across train, validation, and test data. It also avoids fitting imputers, scalers, or encoders on the test set.

The Logistic Regression model is tuned using GridSearchCV with Stratified K-Fold cross-validation. The grid evaluates regularization strength `C` values of 0.05, 0.1, 1.0, and 5.0; penalties `l1` and `l2`; and class weights `None` and `balanced`. The best configuration in the executed run was `C = 1.0`, `penalty = l1`, and `class_weight = balanced`, selected by cross-validated F1-score. This configuration reflects the imbalance-sensitive objective: balanced class weighting increases attention to the positive class and improves recall.

The neural model is implemented with two paths. The preferred path uses TensorFlow/Keras Sequential: dense hidden layers, ReLU activations, dropout, a sigmoid output unit, Adam optimization, binary cross-entropy loss, validation data, and early stopping. The executed environment did not contain TensorFlow, so the notebook used an `sklearn` MLP fallback with equivalent dense feed-forward structure, early stopping, and validation monitoring. The neural hyperparameter screen evaluated layer depth, hidden units, dropout level, and learning rate. The selected executed configuration was `(128, 64, 32)` hidden units, dropout 0.30 in the Keras design, and learning rate 0.0005. In fallback execution, dropout is documented as part of the Keras target architecture, while sklearn MLP approximates the dense neural comparison without dropout.

The evaluation protocol reports metrics for training, validation, and test sets. It also exports a cross-validation summary. This allows overfitting analysis rather than relying on a single final table. Confusion matrices and ROC curves are saved for both models. Feature importance is evaluated with Logistic Regression coefficients and neural grouped permutation importance.

## 7. Data Cleaning

The cleaning process is intentionally conservative. The notebook does not silently assume the dataset schema is correct. It defines the expected columns and fails early if required fields are missing. This protects reproducibility because a renamed or incomplete dataset would be detected before model training.

String cleaning is applied to carrier and airport identifiers. Whitespace differences can create artificial categories in one-hot encoding, so stripping these fields is necessary. Numeric coercion is applied to all non-string columns. This ensures that downstream arithmetic and modeling operations behave consistently even if the CSV parser reads unexpected values.

The most important cleaning decision is filtering `arr_flights`. Because the target delay ratio divides by `arr_flights`, records with missing or non-positive arrival flight counts cannot be used. The notebook removed 240 such rows. This is a small fraction of the raw dataset and is methodologically justified. Keeping these rows would create infinite or undefined ratios; imputing the denominator would fabricate the target.

Missing predictor values are not globally dropped. Instead, the model pipelines impute missing categorical values with the most frequent category and numeric values with the median. Pipeline-based imputation is preferable because imputation parameters are learned only from training data during model fitting. This prevents validation or test information from influencing preprocessing.

The missing-value summary is exported to `outputs/tables/missing_values_summary.csv`, and a missingness chart is saved to `outputs/figures/missing_values_bar.png`. This satisfies the data-quality documentation requirement and gives a reproducible artifact for the report.

## 8. Exploratory Data Analysis

The EDA section examines distribution, imbalance, seasonality, carrier differences, airport differences, cause patterns, disruption variables, and numeric correlation. All charts include titles and labeled axes and are saved under `outputs/figures`.

The delay-ratio distribution is right-skewed. Many aggregate records have moderate delay ratios, while a smaller subset shows substantially higher risk. This supports the threshold-based classification framing: rather than treating all delay ratios equally, the project identifies the upper-risk segment. The selected threshold appears as a vertical reference line in the distribution chart.

The class-balance chart shows the approximately 75/25 split between low-risk and high-risk observations. This imbalance is meaningful enough to require precision, recall, F1, and ROC-AUC, but not so severe that specialized resampling is necessary. The Logistic Regression tuning grid includes balanced class weighting to address this imbalance.

Monthly delay patterns show seasonal structure. Some months have higher average delay ratios than others. This is plausible because weather conditions, holiday travel, and congestion vary over the year. Month is retained as a strict predictive feature because it is available before the outcome and does not directly encode the target.

Carrier comparison shows substantial variation in average delay ratio across carriers. This suggests that operational practices, network design, aircraft utilization, and market structure may influence aggregate delay risk. Airport comparison also shows heterogeneity, reflecting local congestion, weather exposure, runway capacity, and operational complexity. These plots justify including carrier and airport identifiers in the feature set.

The correlation heatmap shows strong relationships among delay-count and delay-duration variables, reinforcing the leakage policy: these realized-outcome variables are useful for diagnosis but not defensible as prospective predictors.

Delay-cause breakdown and cancellation/diversion monthly charts provide additional operational context. Carrier, national aviation system, weather, and late-aircraft causes are more prominent than security-related causes. Cancellations and diversions vary over time and are included as strict predictors because they represent disruption signals in the aggregate record. A production system would need to define exactly when these variables are available, but within the aggregate project framing they are less direct leakage risks than delay counts and durations.

## 9. Feature Engineering

The main engineered feature is `delay_ratio`, calculated as `arr_del15 / arr_flights`. This feature is used to create the target label, not as a predictor. The binary target `high_delay_risk` is then created using the selected threshold.

Categorical feature engineering is performed through one-hot encoding of carrier and airport. This allows both models to learn carrier-specific and airport-specific patterns. One-hot encoding is appropriate here because the number of categories is manageable and because Logistic Regression requires numeric inputs. The encoder uses `handle_unknown='ignore'`, which makes the pipeline robust if a validation or test split contains categories not seen in the training split.

Numeric features are standardized after imputation. Standardization is important for Logistic Regression regularization and neural network optimization because features with different scales can otherwise dominate gradients or coefficient penalties. The numeric variables are year, month, arrival flight volume, cancellations, and diversions.

The project deliberately avoids adding features that would create leakage. For example, using `arr_del15`, delay-cause counts, or delay-duration totals would produce artificially strong predictive performance because those values are outcomes or near-outcomes. A weaker project might include them and report inflated metrics. The improved notebook documents excluded columns and saves them in the run summary.

Future feature engineering could add lagged delay ratios, rolling airport congestion indicators, weather forecasts, route distance, aircraft rotations, scheduled departure windows, and holiday indicators. These would likely improve predictive validity while preserving a prospective forecasting interpretation, provided they are measured before the prediction target period.

## 10. Model 1: Logistic Regression

Logistic Regression serves as the interpretable baseline. It is appropriate for this project because the feature matrix includes one-hot encoded categorical variables and standardized numeric predictors. The model is transparent enough for academic interpretation and operational communication.

The hyperparameter grid evaluated 16 configurations across regularization strength, penalty type, and class weighting. The best cross-validated configuration used `C = 1.0`, `l1` penalty, and balanced class weights. L1 regularization can drive some coefficients toward zero, which is useful in a high-dimensional one-hot encoded feature space. Balanced class weights increase the penalty for misclassifying high-risk records, improving sensitivity to the minority class.

On the held-out test set, Logistic Regression achieved accuracy 0.6095, precision 0.3559, recall 0.6936, F1-score 0.4704, and ROC-AUC 0.6874. The model's most notable strength is recall. It captures about 69% of high-risk cases at the default threshold. Its weakness is precision: many records flagged as high risk are false positives. This behavior may still be acceptable when missing a high-risk operational period is more costly than investigating extra alerts.

The train, validation, and test metrics are similar. Training F1 was 0.4757, validation F1 was 0.4731, and test F1 was 0.4704. This indicates low overfitting risk for the Logistic Regression model. The 5-fold stratified CV summary also supports stability: mean F1 was 0.4722 with a standard deviation of 0.0011, and mean ROC-AUC was 0.6884 with a standard deviation of 0.0025.

Coefficient analysis provides interpretability. The largest absolute coefficients are mostly airport indicators, which suggests that specific airports strongly shift delay-risk odds after accounting for carrier, month, year, and disruption variables. This does not prove airport causality, but it identifies operational units associated with higher or lower risk in the model.

## 11. Model 2: Neural Network

The advanced model is a feed-forward neural network. The notebook contains a TensorFlow/Keras Sequential implementation with dense layers, dropout, validation data, and early stopping. The intended Keras architecture includes hidden layers such as 128, 64, and 32 units with ReLU activation and dropout regularization, followed by a sigmoid output layer for binary classification. The model uses binary cross-entropy loss and Adam optimization.

The executed run used the sklearn MLP fallback because TensorFlow was unavailable in the Python 3.13 runtime. This limitation is explicitly recorded as `sklearn_mlp_fallback (ModuleNotFoundError: No module named 'tensorflow')`. The fallback still trains a dense neural classifier with early stopping and validation monitoring, so the notebook remains executable and the advanced-model comparison is reproducible. The requirements file now includes TensorFlow with a Python-version marker for compatible environments.

Neural hyperparameter tuning evaluated four configurations. The tested configurations varied hidden layer count, hidden units, dropout in the Keras design, and learning rate. The best validation F1 in the tuning stage came from the `(128, 64, 32)` configuration with learning rate 0.0005. The validation tuning F1 scores were lower than the final full-training test F1 because tuning used a controlled 30,000-row subset for CPU feasibility, while the final model trained on the full training set.

On the held-out test set, the neural network achieved accuracy 0.8002, precision 0.6331, recall 0.4776, F1-score 0.5445, and ROC-AUC 0.8181. Compared with Logistic Regression, it substantially improved accuracy, precision, F1, and ROC-AUC. It produced fewer false positives and better ranking quality. However, recall was lower than Logistic Regression's recall, meaning it missed more high-risk records at the default 0.5 threshold.

The neural model shows some overfitting but not catastrophic failure. Training F1 was 0.6663, validation F1 was 0.5497, and test F1 was 0.5445. The gap indicates that the neural model fits training data more strongly than validation data, which is expected for a more flexible model. Validation and test performance are close, suggesting the model generalizes reasonably after early stopping. The neural CV proxy achieved mean F1 0.5208 and mean ROC-AUC 0.8121, supporting the conclusion that neural ranking performance is stronger than the logistic baseline.

## 12. Evaluation

The final test comparison is summarized as follows. Logistic Regression achieved accuracy 0.6095, precision 0.3559, recall 0.6936, F1-score 0.4704, and ROC-AUC 0.6874. The neural network achieved accuracy 0.8002, precision 0.6331, recall 0.4776, F1-score 0.5445, and ROC-AUC 0.8181.

These results show that the neural network is better at ranking and precision-oriented classification, while Logistic Regression is better at recall-oriented detection. The ROC-AUC gap is large: 0.8181 for the neural model versus 0.6874 for Logistic Regression. This means the neural model is more effective at assigning higher scores to high-risk observations across thresholds. If the operating threshold were tuned, the neural model might achieve a better balance between recall and precision than it shows at the default 0.5 threshold.

The confusion matrices illustrate the operational tradeoff. Logistic Regression flags many high-risk cases, but it also creates many false alarms. The neural network is more selective, reducing false positives and improving precision, but it misses more true high-risk cases. In aviation operations, the best model depends on the cost structure. If missed high-risk periods cause major disruption, recall may dominate. If mitigation resources are scarce, precision may dominate.

Cross-validation strengthens confidence in the results. Logistic Regression has low variance across folds, indicating stable but moderate performance. The neural CV proxy has higher F1 and ROC-AUC than Logistic Regression, with manageable fold-to-fold variation. Because neural training is more sensitive to initialization and training dynamics, the validation and test results should be interpreted together rather than as a single definitive number.

The project also exports train, validation, and test metrics. This is important because a model that performs much better on training data than validation data may be overfitting. Logistic Regression has minimal train-test gap. The neural model has a larger train-validation gap, but validation and test metrics align closely, which supports acceptable generalization.

## 13. Model Comparison

The model comparison is not a simple winner-take-all result. The neural network is the stronger model by ROC-AUC, F1-score, precision, and accuracy. It is the best default choice when ranking quality and false-positive control matter. Logistic Regression is stronger by recall and interpretability. It is the better choice when the primary goal is to capture as many high-risk cases as possible and when stakeholders need transparent coefficient-level explanations.

From an academic perspective, this comparison is valuable because it demonstrates why multiple metrics are necessary. Accuracy alone would make the neural model look overwhelmingly superior. Recall alone would favor Logistic Regression. ROC-AUC and F1 provide a broader view: the neural model learns more discriminative nonlinear patterns, but the logistic model's balanced class weighting creates a high-sensitivity decision boundary.

The comparison also reflects the nature of tabular data. Neural networks can improve performance, but they are not automatically optimal. The current neural model likely benefits from nonlinear combinations of carrier, airport, month, year, and disruption signals. However, a tree-based gradient boosting model might perform even better on this kind of structured data. That is an important future-work direction.

Operationally, the recommended approach is to use the neural model as the primary scoring model when probability ranking is needed, while retaining Logistic Regression as an interpretable benchmark and communication tool. The final decision threshold should be tuned using business costs. For example, an airline could lower the neural threshold if recall is too low, accepting more false positives in exchange for capturing more high-risk cases.

## 14. Discussion

The results demonstrate that aggregate airline delay risk is predictable to a meaningful degree using a small strict feature set. Carrier, airport, year, month, arrival volume, cancellations, and diversions contain enough signal to distinguish many high-risk records from lower-risk records. The neural model's ROC-AUC of 0.8181 is strong for a restricted aggregate-feature setting.

Grouped neural permutation importance shows that carrier is the most influential feature group, with a mean validation ROC-AUC drop of 0.1469 when permuted. Year is second with a drop of 0.1014, followed by airport at 0.0834 and month at 0.0750. Arrival flights and cancellations also contribute, while diversions have a smaller effect. These results have operational meaning. Carrier effects may reflect network strategy, scheduling practices, fleet utilization, and operational resilience. Year effects may reflect long-term changes, including demand shocks and network restructuring. Airport effects reflect local congestion and infrastructure. Month effects reflect seasonality.

The Logistic Regression coefficient table also emphasizes airport-specific effects. Some airport indicators have large positive or negative coefficients. These coefficients should not be interpreted causally, especially because aggregate airport records may have different traffic volumes and carrier mixes. Still, they are useful for identifying where the model sees strong historical association.

The leakage audit is one of the most important improvements. A naive version of this project could include `arr_del15`, delay-cause counts, or delay-duration totals as predictors. That would inflate performance because those fields describe the outcome after delays occurred. The strict model excludes those variables, making the reported performance more credible. This is a central academic and engineering point: predictive modeling must respect when information becomes available.

The TensorFlow limitation is also handled transparently. A project submitted as a final AI course assignment should not claim to run Keras if the local environment cannot execute it. The notebook contains the Keras implementation and the fallback records why it was used. The requirements file indicates how to use TensorFlow in compatible Python versions. This is a reproducibility strength rather than a weakness because it documents the actual execution environment.

## 15. Limitations

The most important limitation is data granularity. Monthly carrier-airport aggregates cannot represent individual flights, routes, aircraft rotations, gate constraints, crew timing, or hourly weather. The model is therefore useful for aggregate risk screening, not passenger-level flight prediction.

Second, the split strategy is random stratified rather than temporal. This is acceptable for comparing models under a course-project setting, but a real forecasting system should use time-based validation. A model trained on 2013-2021 and tested on 2022-2023 would better measure future generalization. Temporal validation is especially important because aviation operations changed during the COVID-19 period.

Third, the strict feature set is intentionally limited. Excluding leakage-risk variables improves methodological validity, but it also removes information that may explain delays. Future work should add legitimate pre-outcome variables such as weather forecasts, scheduled traffic volume, route distance, airport capacity, holiday indicators, and prior-month delay history.

Fourth, the Keras model was not executed on this machine due to the Python 3.13 TensorFlow compatibility issue. The notebook includes the Keras code path, but the reported neural results come from the sklearn MLP fallback. In a compatible environment, the Keras model should be rerun and compared directly.

Fifth, the classification threshold is policy-dependent. The 75th percentile is statistically convenient and class-balanced, but different organizations may define high risk differently. Threshold optimization should be tied to intervention costs.

## 16. Future Work

Future work should first add temporal validation. A forward-chaining evaluation would reveal whether the model generalizes across years and whether concept drift affects performance. This would make the study closer to a real forecasting deployment.

Second, the project should evaluate tree-based ensemble models such as random forest, gradient boosting, XGBoost, LightGBM, or CatBoost. Literature on tabular flight delay prediction often finds that boosted trees perform strongly. Adding these models would create a more complete benchmark.

Third, external features should be integrated. Weather, airport capacity, route network structure, flight schedules, holidays, and aircraft rotations would likely improve predictive power while preserving prospective validity.

Fourth, the neural network should be rerun in a TensorFlow-compatible Python environment, preferably Python 3.10 or 3.11. The Keras training history would provide explicit training and validation loss curves, and dropout behavior would match the intended architecture.

Fifth, threshold optimization should be expanded. Precision-recall curves, cost curves, and calibration plots would help translate probability scores into operational decisions. A high-recall threshold might be selected for resilience planning, while a high-precision threshold might be selected for expensive interventions.

Sixth, explainability can be deepened with SHAP or accumulated local effects. Grouped permutation importance is useful, but more detailed explanations would help operational stakeholders understand model behavior.

## 17. Conclusion

This project builds a complete, leakage-aware airline delay-risk classification pipeline using a decade of aggregate U.S. airline delay data. The final notebook validates the dataset, cleans the data, constructs a defensible target, performs EDA, trains and tunes Logistic Regression and neural network models, evaluates results on train/validation/test splits, runs stratified cross-validation, exports metrics and figures, and analyzes feature importance.

The neural network achieved the strongest overall predictive performance, with test ROC-AUC 0.8181 and F1-score 0.5445. Logistic Regression achieved lower ROC-AUC and F1, but much higher recall at the default threshold. This creates a practical tradeoff: the neural model is better for ranking and precision, while Logistic Regression is better for broad high-risk detection and interpretability.

The project's strongest methodological contribution is its leakage policy. By excluding `arr_del15`, delay-cause counts, and delay-duration fields from strict predictive modeling, the project avoids inflated results and preserves a realistic forecasting interpretation. The project also documents its environment honestly: TensorFlow/Keras is implemented as the preferred deep learning path, while the executed Python 3.13 run uses an sklearn neural fallback because TensorFlow is unavailable.

From an operational perspective, the model can support carrier-airport-month risk screening, planning prioritization, and resource-allocation discussions. It should not be deployed as a flight-level predictor without additional data and temporal validation. With external pre-outcome features, time-based validation, and expanded model benchmarks, this project could become a stronger applied aviation analytics system.

## 18. References

- Dai, M. (2024). A hybrid machine learning-based model for predicting flight delay through aviation big data. *Scientific Reports*, 14, Article 4603. https://www.nature.com/articles/s41598-024-55217-z
- Chen, J., & Li, M. (2019). Chained predictions of flight delay using machine learning. *AIAA SciTech Forum*. https://doi.org/10.2514/6.2019-1661
- Cheevachaipimol, W., Teinwan, B., & Chutima, P. (2021). Flight delay prediction using a hybrid deep learning method. *Engineering Journal*, 25(8), 99-112. https://doi.org/10.4186/ej.2021.25.8.99
- Lübbe, J., Schäfer, M., & Schultz, M. (2021). Probabilistic flight delay predictions using machine learning and applications to the flight-to-gate assignment problem. *Aerospace*, 8(6), 152. https://www.mdpi.com/2226-4310/8/6/152
- Rodríguez-Sanz, Á., Comendador, V. F. G., Valdés, R. A., & Pérez-Castán, J. A. (2023). Study of delay prediction in the US airport network. *Aerospace*, 10(4), 342. https://www.mdpi.com/2226-4310/10/4/342
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
- Sokolova, M., Japkowicz, N., & Szpakowicz, S. (2006). Beyond accuracy, F-score and ROC: a family of discriminant measures for performance evaluation. *AI 2006*.
- Scikit-learn documentation. LogisticRegression, MLPClassifier, GridSearchCV, ColumnTransformer, and permutation importance. https://scikit-learn.org/stable/
- TensorFlow Keras documentation. Sequential model and callbacks. https://www.tensorflow.org/guide/keras/sequential_model
- Kaggle dataset: Sriharsha Eedala, Airline Delay dataset. https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
