# Industrial Predictive Maintenance & Risk Modeling (AI4I 2020)

An end-to-end machine learning pipeline for early detection of industrial milling machine failures under severe class imbalance (**3.39% failure rate**), combining domain-informed physical feature engineering, model-aware preprocessing experiments, TreeSHAP diagnostics, and a multi-tier production inference module.

---

## Key Results (Held-Out Test Set, $n = 1,500$)

Evaluated strictly once on the untouched test partition using the tuned **Gradient Boosting Classifier** (`threshold = 0.511`):

| Metric | Score | Operational Impact |
| :--- | :---: | :--- |
| **PR-AUC** | **0.9214** | High precision-recall ranking under 3.39% failure imbalance |
| **ROC-AUC** | **0.9751** | Strong overall class discrimination |
| **Precision** | **1.0000** | **0 false alarms** across 1,449 normal operating cycles |
| **Recall** | **0.8039** | **41 of 51 actual failures caught** prior to breakdown |
| **F1-Score / F2-Score** | **0.8913 / 0.8367** | Balanced precision-recall performance |
| **Overall Accuracy** | **99.33%** | +2.73% lift over the 96.60% majority-class dummy baseline |

---

## Engineering & Modeling Highlights

### 1. Physics-Informed Feature Engineering
Raw sensor streams (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`, and product quality tier `Type`) were transformed into domain-specific physical indicators:
* **`Power_deviation_W` (~33% model importance):** Absolute deviation from the nominal $6,250\text{ W}$ operating midpoint, converting two-sided power failure bounds ($<3,500\text{ W}$ or $>9,000\text{ W}$) into a monotonic risk signal.
* **`Wear_torque_load_Nm*min` (~23% model importance):** Cumulative mechanical overstrain product (`Tool wear × Torque`) targeting Overstrain Failures (`OSF`).
* **`Temp_diff_K` & `Thermal_speed_risk_K/rpm`:** Thermal dissipation gradient (`Process temp − Air temp`) normalized by spindle speed to isolate Heat Dissipation Failures (`HDF`).
* **`Type_wear_limit_ratio`:** Tool wear normalized by quality-tier limits (`L: 200 min`, `M: 220 min`, `H: 240 min`).

### 2. Controlled Preprocessing Experiments
Rather than applying a one-size-fits-all pipeline, preprocessing strategies were benchmarked across linear and tree-based model families:
* **Linear Models (`Logistic Regression`):** `PowerTransformer` (Yeo-Johnson) + `OneHotEncoder(drop="first")` outperformed `StandardScaler` (`0.3906` PR-AUC) and `RobustScaler` (`0.3860` PR-AUC), reaching `0.4161` PR-AUC.
* **Tree Ensembles (`Gradient Boosting`, `LightGBM`, `XGBoost`, `Random Forest`):** `OrdinalEncoder` (`L=0, M=1, H=2`) + raw physical units preserved scale invariance and real engineering units, outperforming linear baselines by **> +0.50 PR-AUC**.

### 3. Error Analysis & SHAP Explainability
* **False Negative Audit:** Inspecting the 10 missed test failures revealed **7 stochastic Tool Wear Failures (`TWF`)** occurring inside the `[200, 240]` minute normal wear window and **3 boundary Heat Dissipation Failures (`HDF`)**—including row `3829` (`Predicted_Prob = 0.5107` vs. `0.5110` cutoff).
* **Multi-Tier Production Alerting (`src/predict.py`):** Translates raw failure probabilities into three actionable maintenance tiers:
  * `NORMAL` ($p < 0.20$)
  * `WATCHLIST` ($0.20 \le p < 0.511$ — captures borderline thermal/wear drift for scheduled inspection)
  * `CRITICAL` ($p \ge 0.511$ — immediate intervention with 100% historical precision)

---

## Repository Structure

```text
├── data/                        # Raw and processed AI4I 2020 dataset splits
├── models/
│   └── gradient_boosting_champion.joblib  # Serialized pipeline, threshold & metadata
├── notebooks/
│   └── Predictive_Maintenance.ipynb       # Full EDA, experiments, tuning & SHAP analysis
├── reports/
│   ├── test_evaluation_summary.png        # Confusion matrix, PR curve & feature importance
│   ├── shap_summary_beeswarm.png          # Global TreeSHAP feature impact
│   └── shap_waterfall_row3829.png         # Local SHAP explanation for borderline HDF case
├── src/
│   └── predict.py                         # Standalone batch/real-time inference script
└── requirements.txt                       # Reproducible Python environment dependencies