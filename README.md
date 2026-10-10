# Predicta: Customer Churn Prediction

## The problem

A phone and internet company loses customers every month. This is called churn. Winning a new customer usually costs more than keeping an existing one, so it helps to know who is likely to leave while there is still time to call them or make them an offer.

This project looks at a telecom company's customer list and tries to flag the customers who are likely to leave. Missing a customer who leaves is the costly mistake, so the goal is to catch as many leavers as possible without flagging too many customers who would have stayed anyway.

## Data

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle: 7,043 customers, one row each. 1,869 of them (26.5%) left.

| Columns | What they mean |
|---|---|
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Who the customer is: gender, whether they are 65 or older, whether they have a partner, whether they have dependents |
| `tenure` | How many months they have been a customer |
| `PhoneService`, `MultipleLines` | Whether they have a phone line, and more than one |
| `InternetService` | DSL, fiber optic, or no internet |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Internet add-ons they pay for |
| `Contract` | Month-to-month, one year, or two year |
| `PaperlessBilling`, `PaymentMethod` | How they are billed and how they pay |
| `MonthlyCharges`, `TotalCharges` | What they pay each month, and what they have paid in total |
| `Churn` | Whether they left (the thing to predict) |

`customerID` is dropped because it is only a label.

## What I did

1. **Cleaned the data.** `TotalCharges` is blank for 11 customers. All 11 have a tenure of 0 months (they have not been billed yet), so the blanks are set to 0.
2. **Split the data.** 80% for training (5,634 customers) and 20% for testing (1,409 customers), keeping the share of leavers the same in both.
3. **Prepared the columns the same way for training and prediction.** Text columns are turned into 0/1 columns and number columns are put on the same scale. This preparation and the model are saved together as one file, so a prediction always goes through exactly the steps used in training.
4. **Compared three models, chosen on the training data only.** Logistic Regression, Random Forest and XGBoost. Only about 1 in 4 customers leave, so each model was told to give leavers more weight. Each model was scored 5 times on different slices of the training data (5-fold cross-validation), using F1 for leavers, a single score that balances "how many leavers were caught" against "how many flags were right".
5. **Tested the chosen model once.** The test customers were not used for anything until this final check.

## Results

Cross-validation on the training data (F1 for leavers, average of 5 folds ± standard deviation):

| Model | F1 |
|---|---|
| Logistic Regression (weighted) | 0.628 ± 0.023 |
| Random Forest (weighted) | 0.626 ± 0.019 |
| **XGBoost (weighted)** | **0.632 ± 0.011** |
| Random Forest, no weighting (the "before") | 0.584 ± 0.019 |

XGBoost scored highest and was chosen. The three weighted models are within each other's spread, so this is close to a tie; what clearly matters is the weighting, not which model.

Test set (1,409 customers, 374 of whom left):

| | Before: Random Forest, no weighting | After: XGBoost, weighted |
|---|---|---|
| Accuracy | 80.7% | 75.8% |
| Precision | 67.5% | 53.2% |
| Recall | 52.7% | 73.5% |
| F1 | 0.592 | 0.617 |
| ROC-AUC | 0.840 | 0.836 |

- **Before:** of 374 customers who left, caught 197; flagged 95 customers who stayed.
- **After:** of 374 customers who left, caught 275; flagged 242 customers who stayed.

In plain words: the chosen model catches 78 more leavers, and the price is 147 more false alarms. Accuracy goes down because of those false alarms. ROC-AUC is about the same, which says the two models rank customers about equally well; the weighting mainly moves the line for who gets flagged. Whether that trade is worth it depends on how much a retention call costs compared with losing a customer.

All numbers are in [reports/metrics.json](reports/metrics.json).

## What drives churn

Churn rates across all 7,043 customers, from [sql/churn_analysis.sql](sql/churn_analysis.sql). Full output is in [reports/churn_rates.md](reports/churn_rates.md).

**By contract**

| Contract | Customers | Left | Churn rate |
|---|---|---|---|
| Month-to-month | 3,875 | 1,655 | 42.7% |
| One year | 1,473 | 166 | 11.3% |
| Two year | 1,695 | 48 | 2.8% |

**By tenure**

| Tenure | Customers | Left | Churn rate |
|---|---|---|---|
| 0–12 months | 2,186 | 1,037 | 47.4% |
| 13–24 months | 1,024 | 294 | 28.7% |
| 25–48 months | 1,594 | 325 | 20.4% |
| 49+ months | 2,239 | 213 | 9.5% |

**By payment method**

| Payment method | Customers | Left | Churn rate |
|---|---|---|---|
| Electronic check | 2,365 | 1,071 | 45.3% |
| Mailed check | 1,612 | 308 | 19.1% |
| Bank transfer (automatic) | 1,544 | 258 | 16.7% |
| Credit card (automatic) | 1,522 | 232 | 15.2% |

**By internet service**

| Internet service | Customers | Left | Churn rate |
|---|---|---|---|
| Fiber optic | 3,096 | 1,297 | 41.9% |
| DSL | 2,421 | 459 | 19.0% |
| No internet | 1,526 | 113 | 7.4% |

As a cross-check, the five inputs the chosen model relies on most are: month-to-month contract, fiber optic internet, no online security, two-year contract, and no phone service. The first two match the biggest gaps in the tables above.

## Limits

- **Public dataset.** This is a sample dataset from Kaggle, not a real company's live data. The numbers show the method works here, not what it would achieve elsewhere.
- **One test split.** The test numbers come from a single group of 1,409 customers. A different split would give somewhat different numbers.
- **Patterns are not causes.** Customers who pay by electronic check leave more often, but that does not mean changing how they pay would make them stay.
- **The score is not a true probability.** Because leavers were given extra weight, the model's churn score runs high. Use it to rank customers, not as the real chance that a customer leaves.

## How to run

```bash
git clone https://github.com/pranjal25r/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Train** (compares the models, saves `models/churn_model.pkl` and `reports/metrics.json`):
```bash
python -m src.train_model
```

**SQL analysis** (saves `reports/churn_rates.md`):
```bash
python -m src.run_sql
```

**Predict** for a sample customer:
```bash
python -m src.predict
```

**Dashboard:**
```bash
streamlit run dashboard/app.py
```

The notebook in `notebooks/EDA.ipynb` is a first look at the data (charts and summaries); it is not part of the pipeline.
