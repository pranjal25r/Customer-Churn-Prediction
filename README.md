# Customer Churn Prediction for SaaS

## Project Overview

Customer churn is one of the biggest challenges faced by SaaS companies. This project predicts whether a customer will churn using machine learning, and identifies the key drivers behind churn to enable data-driven retention strategies.

The project covers an end-to-end, reproducible ML workflow:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (encoding, scaling)
- Model training and comparison (Random Forest, XGBoost)
- Class imbalance handling via SMOTE
- Model evaluation with business-relevant metrics
- Batch and single-customer prediction

---

## Dataset

- **Source:** Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers, 19 features after preprocessing
- **Class distribution:** 73.5% non-churn / 26.5% churn (imbalanced)
- **Features:** demographics, subscription details, payment methods, tenure, billing

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, joblib

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory data analysis
│
├── src/
│   ├── data_processing.py         # Cleaning, encoding, scaling, train/test split
│   ├── train_model.py             # Training pipeline (RF, RF+SMOTE, XGBoost)
│   └── predict.py                 # Single & batch prediction utilities
│
├── models/
│   ├── churn_model.pkl            # Best trained model + preprocessing artifacts
│   └── feature_importance_*.png   # Feature importance plot for best model
│
├── dashboard/
│   └── app.py                     # Prediction dashboard
│
├── test_prediction.py
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/pranjal25r/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model
```bash
PYTHONPATH=. python src/train_model.py
```
This trains Random Forest, Random Forest + SMOTE, and XGBoost, evaluates all three on a held-out test set, and saves the best model (selected by F1-score) to `models/churn_model.pkl`.

### Explore the data
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Make predictions
```python
from src.predict import predict_customer

result = predict_customer({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
})
```

---

## Results

The dataset is imbalanced (73.5% non-churn vs 26.5% churn). Accuracy alone is misleading here, since a model can score ~74% by always predicting "no churn." Three models were trained and compared on a held-out test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (baseline) | 80.2% | 66.2% | 51.9% | 0.582 | 0.840 |
| **Random Forest + SMOTE (selected)** | 76.2% | 53.7% | **75.9%** | **0.629** | 0.837 |
| XGBoost | 80.1% | 65.6% | 52.4% | 0.582 | 0.836 |

**Random Forest + SMOTE** was selected as the best model. SMOTE oversamples the minority (churn) class during training, which raises churn-class recall from 52% to 76% — a ~46% relative improvement — at the cost of some accuracy and precision. For a churn-prediction use case, this is the right trade-off: missing an at-risk customer (false negative) is more costly to the business than flagging a customer who doesn't actually churn (false positive).

**Key churn drivers** (from feature importance):
- Contract type (month-to-month contracts churn far more than annual/two-year)
- Tenure (newer customers churn more)
- Payment method (electronic check correlates with higher churn)
- Monthly charges

---

## Business Impact

- Identifies high-risk customer segments for proactive retention outreach
- Prioritizes recall over raw accuracy, aligned with the real cost of missing a churner
- Provides interpretable feature importance to guide retention strategy, not just predictions

---

## Future Improvements

- Hyperparameter tuning (grid/random search, Optuna)
- Deploy as a REST API (FastAPI/Flask)
- Threshold tuning based on business cost of false positives vs false negatives
- Experiment with deep learning approaches on larger datasets

---

## Acknowledgements

- Kaggle for the Telco Customer Churn dataset
- Open-source ML community
