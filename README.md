# Credit Card Fraud Detection — ML Project
## Credit Card Fraud Detection using Machine Learning
An end-to-end machine learning system that detects fraudulent credit card transactions — with rich feature engineering, XGBoost classification, and SHAP explainability that explains why a transaction was flagged in plain English.
---

## Problem Statement
Credit card fraud costs billions of dollars annually. Traditional rule-based systems fail because they can't adapt to evolving fraud patterns. This project builds an ML pipeline trained on 555,719 real-world transactions to automatically flag suspicious activity — even under extreme class imbalance (only 0.5% of transactions are fraud).
---

## Project Structure
```
fraud-detection-ml/
│
├── data/
│   ├── raw/                        # Original CSVs 
│   └── processed/                  # Feature-engineered data
│
├── models/
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability.ipynb
│   └── model_evaluation_summary.md
│
├── outputs/                        # Charts and evaluation plots
└── README.md
```
---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

| Split | Transactions | Fraud Rate |
|---|---|---|
| Train | 555,719 | ~0.58% |
| Test | 553,574 | ~0.39% |

> Raw data files are excluded via `.gitignore`. Download from Kaggle and place in `data/raw/`.
---

## Feature Engineering

| Feature | Built From | Why It Matters |
|---|---|---|
| `hour`, `is_night` | `trans_date_trans_time` | Fraud spikes sharply at 10pm–3am |
| `age` | `dob` + transaction date | Derived customer age at time of transaction |
| `geo_distance` | `lat/long` + `merch_lat/long` | Haversine distance: home → merchant (km) |
| `amt_zscore` | `amt` + per-card history | Is this amount unusual *for this specific card*? |
| `card_mean_amt` | `cc_num` history | Each cardholder's average spend baseline |
| `category_fraud_rate` | `category` + `is_fraud` | Target encoding: actual fraud rate per merchant type |
---

##  Models & Results

Three models trained and compared:

| Model | Caught Fraud | Missed Fraud | False Alarms | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 1,564 | 581 | 39,727 | 0.92 |
| Random Forest | 1,782 | 363 | 6,924 | 0.99 |
| **XGBoost** | **1,731** | **414** | **3,189** | **0.99** |

**Why XGBoost over Random Forest?**
Random Forest catches slightly more fraud (1,782 vs 1,731) but produces **twice as many false alarms** (6,924 vs 3,189) — meaning thousands of legitimate customers get wrongly blocked. XGBoost offers the better real-world balance between catching fraud and not punishing honest customers.

## SHAP Explainability

Every flagged transaction comes with a human-readable explanation, not just a score.

**Top features by global importance (SHAP):**
1. `amt` — transaction amount (strongest signal by far)
2. `category_fraud_rate` — merchant category risk level
3. `category_enc` — specific merchant category
4. `is_night` — late-night transactions
5. `amt_zscore` — how unusual this amount is for this card

**Example fraud alert output:**
```
FRAUD ALERT:
Fraud probability: 10.2%

Top reasons:
• This category has 0.2% fraud rate
• Late-night transaction
• Transaction amount: $24.84
```
---
## Tech Stack

- `pandas`, `numpy` -> Data wrangling and feature engineering 
- `scikit-learn` -> Preprocessing, Logistic Regression, Random Forest, metrics
- `imbalanced-learn` -> SMOTE resampling (handling imbalance dataset, create synthetic minority class samples)
- `xgboost` -> Primary classification model
- `shap` -> Model explainability
- `matplotlib`, `seaborn` -> Visualization
- `joblib` -> Model saving etc
---
## Key Decisions & Learnings

- **Dropped gender** as a feature — statistically weak and introduces demographic bias
- **Dropped `city_pop` and `unix_time`** after EDA showed near-zero correlation with fraud
- **Chose XGBoost over Random Forest** based on false alarm rate analysis, not just recall
- **Used target encoding** (`category_fraud_rate`) over plain label encoding — more informative for the model
- **PR-AUC over accuracy** — standard accuracy is meaningless on 0.5% fraud rate data
---

# Author

**Tanishka Dass** — CSE Student, Dronacharya Group of Institutions (2028)
