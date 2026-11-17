# Intelligent Predictive Maintenance

Author: Harshith Katta

## Business Understanding and Methodology

The objective is to identify machine failure risks and reduce
unnecessary maintenance inspections.

Dataset: UCI AI4I 2020 Predictive Maintenance.
10,000 records, 339 failures, 3.39% failure rate.
No missing values or duplicate records were identified.

Seven domain-based features were engineered from six original
predictors. Identifiers and failure-mode indicators were excluded
to prevent target leakage.

Stratified training, validation and test splits of 70:15:15
were used with preprocessing fitted on training data.

## ANN Development and Research

A baseline ANN and multiple traditional ML models were developed.

ANN experiments investigated dropout, architecture, initialization,
L1/L2 regularization, class weighting and learning rate.

Total recorded experimental runs: 42.

The optimized architecture uses 64 and 32 hidden neurons,
ReLU, Glorot initialization, 20% dropout, L2 regularization
of 0.001 and Adam with a learning rate of 0.001.

Mean validation Average Precision improved from approximately
0.7363 to 0.7905 under the selected optimization sequence.

The experiments do not establish consistent ANN superiority.

## Business Recommendations

Validation-selected model: Gradient Boosting (Tuned)
Decision threshold: 0.011187

Validation cost: INR 74,000
Test cost: INR 90,000
Test recall: 96.08%
Test Average Precision: 0.9214

Lowest observed test cost: Gradient Boosting (Tuned)
Highest observed test Average Precision: Gradient Boosting (Tuned)

Hypothetical costs:
Inspection = INR 500.
Missed failure = INR 10,000.

High-risk records should receive prioritized inspection.
Medium-risk records require additional monitoring.
Low-risk records continue routine maintenance.

Limitations include synthetic data, rare failures, repeated
validation-set reuse, previously inspected test results,
uncalibrated risk scores and hypothetical operational costs.

Real-world deployment requires further prospective validation.


## Dataset
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Execution
1. Install dependencies using pip install -r requirements.txt.
2. Place the dataset in data/ai4i2020.csv.
3. Open notebooks/Predictive_Maintenance.ipynb.
4. Retain the saved Gradient Boosting pipeline for artifact recovery.

## Final Test Results

```
                         Model  Threshold  Accuracy  Precision  Recall  F1-Score  ROC-AUC  PR-AUC   TN  FP  FN  TP  Business Cost
     Gradient Boosting (Tuned)     0.0112    0.9380     0.3500  0.9608    0.5131   0.9751  0.9214 1358  91   2  49          90000
           Optimized ANN (5:1)     0.1319    0.9367     0.3382  0.9020    0.4920   0.9856  0.8070 1359  90   5  46         118000
   Gradient Boosting (Default)     0.0086    0.9347     0.3333  0.9216    0.4896   0.9587  0.8917 1355  94   4  47         110500
           LightGBM (Weighted)     0.3623    0.9720     0.5570  0.8627    0.6769   0.9847  0.8802 1414  35   7  44         109500
      Random Forest (Balanced)     0.0600    0.9453     0.3740  0.9020    0.5287   0.9703  0.9084 1372  77   5  46         111500
          Optimized ANN (10:1)     0.1724    0.9233     0.2975  0.9216    0.4498   0.9832  0.7748 1338 111   4  47         119000
            XGBoost (Weighted)     0.1468    0.9327     0.3239  0.9020    0.4767   0.9860  0.8771 1353  96   5  46         121000
                  Baseline ANN     0.2257    0.9060     0.2581  0.9412    0.4051   0.9773  0.7431 1311 138   3  48         123000
Logistic Regression (Balanced)     0.4575    0.8540     0.1719  0.8627    0.2866   0.9452  0.4696 1237 212   7  44         198000
```
