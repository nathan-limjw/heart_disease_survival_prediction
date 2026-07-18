# Coronary Artery Disease Survival Prediction

## Overview

This project predicts patient survival (`Survive`) using medical records of coronary artery disease patients, as part of an AIAP self-practice classification assignment. At least three classification models were evaluated and compared, with the best-performing model tuned and evaluated on a held-out test set.

## Project Structure

```
root/
├── eda.ipynb              # Exploratory data analysis
├── README.md
├── requirements.txt
├── data/
│   └── heart_disease_data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
└── main.py
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This runs the full pipeline end-to-end: data cleaning, preprocessing, patient-grouped stratified train/validation/test split, baseline model training, hyperparameter tuning on the top-performing model, and final test-set evaluation. Progress and metrics are logged to console.

Exploratory analysis (distributions, outlier checks, correlation/significance testing, feature selection rationale) is in `eda.ipynb` and is not re-run by `main.py`.

## Approach

**Data cleaning:** deduplication, negative-value correction (`Age`), category standardization (`Survive`, `Smoke`, `Ejection Fraction` shorthand codes), and removal of features identified as uninformative during EDA.

**Feature selection:** numeric features were assessed via Pearson correlation and Mann-Whitney U tests against the target; categorical features via Chi-square tests. Features with no statistically significant relationship to `Survive` (`Gender`, `Smoke`, `Diabetes`, `Creatine phosphokinase`, `Height`, and the synthetic `Favourite Color` column) were dropped. `Hemoglobin` and `Blood Pressure` were dropped after a follow-up feature-importance check across all three models showed consistently negligible contribution; `Platelets` was retained despite weak linear correlation, as it showed meaningful importance in Random Forest, indicating a likely non-linear relationship.

**Preprocessing:** skewed numeric columns were corrected per-column based on empirical comparison of Log1p, Square Root, and Box-Cox transforms (`Creatinine`, `Sodium` → Box-Cox; `Platelets`, `Weight` → Square Root); `Ejection Fraction` was ordinally encoded (Low < Normal < High); all numeric features were standardized.

**Splitting:** since the dataset contains repeated visits per patient (same `ID`, differing clinical values across rows), a grouped and stratified split (`StratifiedGroupKFold`) was used to ensure no patient appears in more than one of the train/validation/test sets, preventing data leakage, while preserving the target's class distribution across all three sets.

**Models evaluated:** Logistic Regression, Random Forest, and XGBoost, compared at default hyperparameters before tuning.

## Results

### Baseline Model Comparison (Validation Set)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.86 | 0.73 |
| Random Forest | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 |

Logistic Regression performed notably worse than both tree-based models. A diagnostic depth-sweep using a standalone decision tree showed training and validation accuracy tracking closely at every depth (e.g. 0.97/0.96 at depth 10), ruling out overfitting or data leakage as the explanation for the tree-based models' near-perfect scores. This indicates `Survive` follows a near-deterministic, threshold-based relationship with a small set of features — learnable by tree-based splitting but not representable by Logistic Regression's linear decision boundary.

**XGBoost** was selected as the final model for tuning, based on its top baseline performance.

### Hyperparameter Tuning

XGBoost was tuned via `RandomizedSearchCV` with grouped, stratified cross-validation. Given Random Forest and XGBoost were already near-perfect at baseline, tuning effort was focused on XGBoost alone rather than tuning all three models, given limited computational resources and minimal expected headroom for improvement.

Best cross-validated PR-AUC: **1.0000**
Best parameters: `n_estimators=200`, `max_depth=5`, `learning_rate=0.2`, `subsample=1.0`

### Final Test Set Evaluation

| Metric | Score |
|---|---|
| Precision (both classes) | 1.00 |
| Recall (both classes) | 1.00 |
| F1-score (both classes) | 1.00 |
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 |

Test performance closely matches validation performance, indicating the model generalizes well with no evidence of overfitting.

## Notes on Data Characteristics

Several indicators suggest this dataset was synthetically constructed rather than sourced from genuine noisy clinical records:

- `Favourite Color` showed a near-uniform distribution across categories and no relationship to `Survive`, consistent with random generation.
- `Height` displayed a regular, alternating "comb" pattern in its histogram rather than a smooth distribution, suggesting a rounding or generation artifact.
- The near-perfect, deterministic-seeming relationship between the retained features and `Survive` — confirmed via the depth-sweep diagnostic — is not typical of real-world clinical outcome data, which is usually noisier.

These characteristics don't affect the validity of the modeling approach but are worth noting when interpreting the near-perfect final metrics.