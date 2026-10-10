"""
Prediction module for Customer Churn Prediction.

Loads the saved pipeline (preprocessing + model) and makes predictions with it.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Optional

from sklearn.pipeline import Pipeline

from src.data_processing import clean_data, FEATURE_COLS


def load_model(filepath: Optional[str] = None) -> Pipeline:
    """
    Load the fitted pipeline saved by train_model.py.

    Args:
        filepath: Path to the model file. Uses default if None.

    Returns:
        Fitted pipeline (preprocessing + model).
    """
    if filepath is None:
        filepath = str(Path(__file__).parent.parent / 'models' / 'churn_model.pkl')

    return joblib.load(filepath)


def prepare_input(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    """
    Turn raw customer data into the columns the pipeline expects.

    Args:
        data: Input data as DataFrame or dict.

    Returns:
        Cleaned DataFrame with the feature columns.
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    return clean_data(data)[FEATURE_COLS]


def risk_level(probability: float) -> str:
    """Map a churn score to 'Low', 'Medium' or 'High'."""
    if probability >= 0.7:
        return 'High'
    if probability >= 0.4:
        return 'Medium'
    return 'Low'


def predict_customer(
    data: Union[pd.DataFrame, dict],
    model_path: Optional[str] = None
) -> dict:
    """
    Make churn prediction for a single customer.

    Args:
        data: Customer data as DataFrame or dictionary.
        model_path: Path to the model file.

    Returns:
        Dictionary with:
            - prediction: 0 (No Churn) or 1 (Churn)
            - probability: Churn score (0.0 to 1.0)
            - churn_label: 'Yes' or 'No'
            - risk_level: 'Low', 'Medium', or 'High'
    """
    pipeline = load_model(model_path)
    features = prepare_input(data)

    prediction = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0, 1]

    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'churn_label': 'Yes' if prediction == 1 else 'No',
        'risk_level': risk_level(probability)
    }


def predict_batch(
    data: pd.DataFrame,
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Make predictions for multiple customers.

    Args:
        data: DataFrame with customer features.
        model_path: Path to the model file.

    Returns:
        DataFrame with predictions and probabilities added.
    """
    pipeline = load_model(model_path)
    features = prepare_input(data)

    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    result = data.copy()
    result['prediction'] = predictions
    result['probability'] = probabilities
    result['churn_label'] = np.where(predictions == 1, 'Yes', 'No')
    result['risk_level'] = [risk_level(p) for p in probabilities]

    return result


def top_features(pipeline: Pipeline, top_n: int = 5) -> pd.DataFrame:
    """
    List the features the model leans on most.

    Uses absolute coefficients for Logistic Regression and feature importances
    for tree models. Features are named after one-hot encoding, e.g.
    'Contract_Month-to-month'.

    Args:
        pipeline: Fitted pipeline.
        top_n: Number of features to return.

    Returns:
        DataFrame with columns 'feature' and 'weight', largest first.
    """
    model = pipeline.named_steps['model']
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()

    if hasattr(model, 'coef_'):
        weights = model.coef_[0]
    else:
        weights = model.feature_importances_

    df = pd.DataFrame({'feature': feature_names, 'weight': weights.astype(float)})
    order = df['weight'].abs().sort_values(ascending=False).index

    return df.loc[order].head(top_n).reset_index(drop=True)


if __name__ == '__main__':
    # Example usage with a single customer (dict)
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    try:
        # Single customer prediction
        result = predict_customer(sample_customer)
        print("Customer Churn Prediction")
        print("=" * 40)
        print(f"Prediction: {result['prediction']} ({result['churn_label']})")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")

        # Batch prediction example
        print("\n" + "=" * 40)
        print("Batch Prediction Example")
        print("=" * 40)
        batch_df = pd.DataFrame([sample_customer, sample_customer])
        batch_df.loc[1, 'tenure'] = 72  # Long-term customer
        batch_df.loc[1, 'Contract'] = 'Two year'
        batch_results = predict_batch(batch_df)
        print(batch_results[['tenure', 'Contract', 'prediction', 'probability', 'risk_level']])

    except FileNotFoundError:
        print("Model not found. Please run train_model.py first.")
