## Credit Card Fraud Detection using Machine Learning
A machine learning system that detects fraudulent credit card transactions in real time. Uses feature engineering + supervised learning to identify fraud patterns more effectively.
# Problem Statement
Credit card fraud costs billions of dollars annually. Traditional rule-based systems miss sophisticated fraud patterns. This project trains ML models on historical labeled transaction data to automatically identify suspicious transactions with high precision and recall — even under extreme class imbalance (~0.5% fraud rate).
# Key Features Engineered
- Hour of Transaction → captures time-based fraud behavior
- Customer Age → derived from date of birth
- Geo Distance → distance between user location and merchant location
- Transaction Categories → encoded into numerical features
# Workflow
Exploratory Data Analysis (EDA)
- Fraud distribution
- Amount patterns
- Time-based trends
Feature Engineering
- Time features
- Age calculation
- Geographic distance (Haversine formula)
- One-hot encoding
Model Training
- Random Forest Classifier
- Train-test split using separate datasets
Evaluation
- Precision, Recall, F1-score.
# Model Performance
- Precision (Fraud): 0.96
- Recall (Fraud): 0.57
- F1 Score: 0.72
Interpretation
- High precision ensures most flagged transactions are truly fraudulent
- Moderate recall indicates some fraud cases are still missed
- This trade-off is expected due to highly imbalanced data
# Project Structure
fraud-detection-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── README.md
└── requirements.txt
# Dataset
- Source: Kaggle Fraud Detection Dataset
- ~555,000 transactions
- ~0.5% fraud cases

Note: Dataset is not included in the repository.
# Author

Tanishka Dass