-- Churn rates from the Telco CSV, loaded into a table called "customers".
-- Run with: python -m src.run_sql
-- The title comment above each query is used by run_sql.py as its heading.

-- title: Overall churn rate
SELECT
    COUNT(*) AS customers,
    SUM(Churn = 'Yes') AS churned,
    ROUND(100.0 * SUM(Churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers;

-- title: Churn rate by contract
SELECT
    Contract AS contract,
    COUNT(*) AS customers,
    SUM(Churn = 'Yes') AS churned,
    ROUND(100.0 * SUM(Churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- title: Churn rate by tenure band
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band,
    COUNT(*) AS customers,
    SUM(Churn = 'Yes') AS churned,
    ROUND(100.0 * SUM(Churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);

-- title: Churn rate by payment method
SELECT
    PaymentMethod AS payment_method,
    COUNT(*) AS customers,
    SUM(Churn = 'Yes') AS churned,
    ROUND(100.0 * SUM(Churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- title: Churn rate by internet service
SELECT
    InternetService AS internet_service,
    COUNT(*) AS customers,
    SUM(Churn = 'Yes') AS churned,
    ROUND(100.0 * SUM(Churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
