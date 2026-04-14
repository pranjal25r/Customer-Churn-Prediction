# Customer Churn Prediction for SaaS

## Project Overview

Customer churn is one of the biggest challenges faced by SaaS companies. This project aims to **predict whether a customer will churn or not** using machine learning techniques and provide actionable insights to improve customer retention.

This project demonstrates an end-to-end data science workflow including:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model building & evaluation
* Business insights

---

## Objective

The main goal of this project is to:

* Predict customer churn using historical data
* Identify key factors influencing churn
* Help businesses take **proactive retention actions**

---

## Dataset

* Dataset used: **Telco Customer Churn Dataset**
* Contains customer information such as:

  * Demographics
  * Subscription details
  * Payment methods
  * Tenure
  * Churn status (Target Variable)

---

## Tech Stack

* **Programming Language:** Python 🐍
* **Libraries Used:**

  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
  * Scikit-learn

---

## Project Workflow

### 1️⃣ Data Cleaning

* Handling missing values
* Converting data types
* Removing inconsistencies

### 2️⃣ Exploratory Data Analysis (EDA)

* Univariate & Bivariate analysis
* Churn distribution visualization
* Feature correlation analysis

### 3️⃣ Feature Engineering

* Encoding categorical variables
* Scaling numerical features
* Feature selection

### 4️⃣ Model Building

Models used:

* Logistic Regression
* Decision Tree
* Random Forest
* (Optional) XGBoost

### 5️⃣ Model Evaluation

* Accuracy
* Precision, Recall, F1-score
* ROC-AUC Score
* Confusion Matrix

---

## 📈 Key Insights

* Customers with **short tenure** are more likely to churn
* **High monthly charges** increase churn probability
* Customers without **long-term contracts** tend to leave more
* **Payment method & service type** significantly impact churn

---

## 📦 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── EDA_and_Model.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│
├── requirements.txt
├── README.md
└── app/ (optional dashboard)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/pranjal25r/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook
```

Or run scripts:

```bash
python src/model_training.py
```

---

## 📊 Results

* Achieved strong performance in predicting churn
* Identified **high-risk customer segments**
* Model can be used for **real-time churn prediction systems**

---

## 💡 Future Improvements

* Deploy model using Flask / FastAPI
* Build interactive dashboard (Streamlit)
* Handle class imbalance using SMOTE
* Hyperparameter tuning
* Use deep learning models

---

## 📌 Business Impact

This project helps SaaS companies:

* Reduce customer churn
* Improve retention strategies
* Increase revenue
* Understand customer behavior

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 🙌 Acknowledgements

* Kaggle for dataset
* Open-source ML community

---
