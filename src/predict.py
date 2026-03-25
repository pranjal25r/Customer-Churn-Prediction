"""
Prediction module for Customer Churn Prediction.

This module provides functions for making predictions using a trained model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Tuple, Any, Optional

from src.data_processing import clean_data, CATEGORICAL_COLS, NUMERICAL_COLS


def load_model(filepath: Optional[str] = None) -> Tuple[Any, dict]:
    """
    Load a trained model and its artifacts.

    Args:
        filepath: Path to the model file. Uses default if None.

    Returns:
        Tuple of (model, artifacts dict).
    """
    if filepath is None:
        filepath = str(Path(__file__).parent.parent / 'models' / 'churn_model.pkl')

    save_object = joblib.load(filepath)
    return save_object['model'], save_object.get('artifacts', {})


def preprocess_input(
    data: Union[pd.DataFrame, dict],
    artifacts: dict
) -> pd.DataFrame:
    """
    Preprocess input data for prediction.

    Args:
        data: Input data as DataFrame or dict.
        artifacts: Preprocessing artifacts from training.

    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    df = data.copy()

    # Clean the data
    df = clean_data(df)

    # Encode categorical features
    encoders = artifacts.get('encoders', {})
    for col in CATEGORICAL_COLS:
        if col in df.columns and col in encoders:
            # Handle unseen categories by using the most frequent encoded value
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    # Scale numerical features
    scaler = artifacts.get('scaler')
    if scaler is not None:
        scale_cols = [col for col in NUMERICAL_COLS if col in df.columns]
        df[scale_cols] = scaler.transform(df[scale_cols])

    # Ensure correct feature order
    feature_names = artifacts.get('feature_names', [])
    if feature_names:
        df = df[feature_names]

    return df


def predict(
    data: Union[pd.DataFrame, dict],
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Make churn predictions on new data.

    Args:
        data: Input data as DataFrame or dict.
        model_path: Path to the model file.

    Returns:
        Array of predictions (0 = No Churn, 1 = Churn).
    """
    model, artifacts = load_model(model_path)
    processed_data = preprocess_input(data, artifacts)
    return model.predict(processed_data)


def predict_proba(
    data: Union[pd.DataFrame, dict],
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Get churn probability predictions.

    Args:
        data: Input data as DataFrame or dict.
        model_path: Path to the model file.

    Returns:
        Array of probabilities for churn class.
    """
    model, artifacts = load_model(model_path)
    processed_data = preprocess_input(data, artifacts)
    return model.predict_proba(processed_data)[:, 1]


def predict_customer(
    data: Union[pd.DataFrame, dict],
    model_path: Optional[str] = None
) -> dict:
    """
    Make churn prediction for customer(s).

    Args:
        data: Customer data as DataFrame or dictionary.
        model_path: Path to the model file.

    Returns:
        Dictionary with:
            - prediction: 0 (No Churn) or 1 (Churn)
            - probability: Churn probability score (0.0 to 1.0)
            - churn_label: 'Yes' or 'No'
            - risk_level: 'Low', 'Medium', or 'High'
    """
    # Load model once
    model, artifacts = load_model(model_path)
    processed_data = preprocess_input(data, artifacts)

    # Get prediction and probability
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0, 1]

    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'churn_label': 'Yes' if prediction == 1 else 'No',
        'risk_level': (
            'High' if probability >= 0.7 else
            'Medium' if probability >= 0.4 else
            'Low'
        )
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
    model, artifacts = load_model(model_path)
    processed_data = preprocess_input(data, artifacts)

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]

    result = data.copy()
    result['prediction'] = predictions
    result['probability'] = probabilities
    result['churn_label'] = np.where(predictions == 1, 'Yes', 'No')
    result['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return result


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
