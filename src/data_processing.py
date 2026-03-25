"""
Data processing module for Customer Churn Prediction.

This module provides functions for loading, cleaning, and preprocessing
the Telco Customer Churn dataset for machine learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional


# Column definitions
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

TARGET_COL = 'Churn'

DROP_COLS = ['customerID']


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw dataset from CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset by handling data types and missing values.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Drop unnecessary columns
    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Fix TotalCharges datatype (contains whitespace strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using Label Encoding.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Tuple of (encoded DataFrame, dict of label encoders).
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target variable
    if TARGET_COL in df.columns:
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
        encoders[TARGET_COL] = le

    return df, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        X_train: Training features.
        X_test: Test features.
        columns: Columns to scale. If None, scales NUMERICAL_COLS.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    if columns is None:
        columns = [col for col in NUMERICAL_COLS if col in X_train.columns]

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    return X_train, X_test, scaler


def prepare_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Complete data preparation pipeline.

    Args:
        filepath: Path to the raw CSV file.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.
        scale: Whether to scale numerical features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, artifacts dict).
    """
    # Load and clean
    df = load_data(filepath)
    df = clean_data(df)

    # Encode
    df, encoders = encode_features(df)

    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale if requested
    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    artifacts = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }

    return X_train, X_test, y_train, y_test, artifacts


def get_processed_data(
    data_dir: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Convenience function to load and process data from default location.

    Args:
        data_dir: Directory containing the data file. If None, uses default.
        **kwargs: Additional arguments passed to prepare_data.

    Returns:
        Same as prepare_data.
    """
    if data_dir is None:
        # Default path relative to this file
        data_dir = Path(__file__).parent.parent / 'data'

    filepath = Path(data_dir) / 'Telco-Customer-Churn.csv'

    return prepare_data(str(filepath), **kwargs)


if __name__ == '__main__':
    # Test the processing pipeline
    X_train, X_test, y_train, y_test, artifacts = get_processed_data()

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Features: {artifacts['feature_names']}")
