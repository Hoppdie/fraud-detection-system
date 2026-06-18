# Model Card — Fraud Detection System

## Model Overview

| Field | Details |
|---|---|
| **Task** | Binary classification — fraudulent vs. legitimate transaction |
| **Algorithm** | XGBoost (gradient boosted trees) |
| **Input** | 30 transaction features (V1–V28 PCA components, Amount, Time) |
| **Output** | Probability score [0, 1] + binary label |
| **Decision Threshold** | 0.5 (tunable via API parameter) |
| **Serving** | FastAPI REST endpoint (`/predict`) |

## Dataset

- **Source**: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Class balance**: 492 fraud (0.17%) vs. 284,315 legitimate — **highly imbalanced**
- **Features**: 28 PCA-anonymised components (V1–V28), raw Amount and Time

## Training Details

```yaml
model: XGBoostClassifier
params:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
  scale_pos_weight: 577  # handles class imbalance
  subsample: 0.8
  colsample_bytree: 0.8
  eval_metric: aucpr
```

Class imbalance is handled via `scale_pos_weight` (ratio of negatives to positives), which biases the model toward catching fraud without synthetic oversampling.

## Evaluation Metrics

| Metric | Value |
|---|---|
| AUROC | ~0.98 |
| AUPRC | ~0.85 |
| Precision @ threshold=0.5 | ~0.91 |
| Recall @ threshold=0.5 | ~0.82 |
| F1 Score | ~0.86 |

> **Why AUPRC over AUROC?** With 0.17% fraud rate, AUROC can be misleadingly high. AUPRC reflects true precision-recall tradeoffs at extreme imbalance.

## SHAP Explainability

The model uses SHAP (SHapley Additive exPlanations) for per-prediction explanations:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Top features driving fraud predictions: **V14**, **V4**, **V12**, **V10**, **Amount**

## Limitations & Bias

- Dataset is from European cardholders (2013) — may not generalise to other regions or modern fraud patterns
- PCA anonymisation prevents direct feature interpretation
- Threshold of 0.5 may need recalibration for different business cost functions (missed fraud vs. false positives)
- Does not handle concept drift — model should be retrained on fresh transaction data periodically

## Serving

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.8, ...]}'
```

## Intended Use

- **In scope**: Internal fraud screening, risk scoring, transaction flagging
- **Out of scope**: Sole automated decision-making without human review; high-stakes automated account suspension

## Contact

Adhiyan Anbazhagan · adhiyan2005@gmail.com
