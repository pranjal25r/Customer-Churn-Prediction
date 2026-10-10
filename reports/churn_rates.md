# Churn rates

Produced by `python -m src.run_sql` from `sql/churn_analysis.sql`, using all 7,043 customers in the CSV.

## Overall churn rate

| customers | churned | churn_rate_pct |
|---|---|---|
| 7043 | 1869 | 26.5 |

## Churn rate by contract

| contract | customers | churned | churn_rate_pct |
|---|---|---|---|
| Month-to-month | 3875 | 1655 | 42.7 |
| One year | 1473 | 166 | 11.3 |
| Two year | 1695 | 48 | 2.8 |

## Churn rate by tenure band

| tenure_band | customers | churned | churn_rate_pct |
|---|---|---|---|
| 0-12 months | 2186 | 1037 | 47.4 |
| 13-24 months | 1024 | 294 | 28.7 |
| 25-48 months | 1594 | 325 | 20.4 |
| 49+ months | 2239 | 213 | 9.5 |

## Churn rate by payment method

| payment_method | customers | churned | churn_rate_pct |
|---|---|---|---|
| Electronic check | 2365 | 1071 | 45.3 |
| Mailed check | 1612 | 308 | 19.1 |
| Bank transfer (automatic) | 1544 | 258 | 16.7 |
| Credit card (automatic) | 1522 | 232 | 15.2 |

## Churn rate by internet service

| internet_service | customers | churned | churn_rate_pct |
|---|---|---|---|
| Fiber optic | 3096 | 1297 | 41.9 |
| DSL | 2421 | 459 | 19.0 |
| No | 1526 | 113 | 7.4 |

## Cross-check: the model's top 5 features

The 5 inputs the chosen model (XGBClassifier) relies on most. Names are the columns after one-hot encoding, e.g. `Contract_Month-to-month` means "Contract is Month-to-month".

| feature | weight |
|---|---|
| Contract_Month-to-month | 0.312 |
| InternetService_Fiber optic | 0.132 |
| OnlineSecurity_No | 0.056 |
| Contract_Two year | 0.047 |
| MultipleLines_No phone service | 0.037 |
