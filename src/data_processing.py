"""
Data processing for Customer Churn Prediction.

Loads and cleans the Telco Customer Churn dataset and builds the
preprocessing step that is shared by training and prediction.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).parent.parent / 'data' / 'Telco-Customer-Churn.csv'

# Column definitions
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Already 0/1 in the raw data, so it is passed through unchanged.
PASSTHROUGH_COLS = ['SeniorCitizen']

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS + PASSTHROUGH_COLS

TARGET_COL = 'Churn'

DROP_COLS = ['customerID']

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str = str(DATA_PATH)) -> pd.DataFrame:
    """
    Load the raw dataset from CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.
    """
    return pd.read_csv(filepath)


def check_blank_total_charges(df: pd.DataFrame) -> dict:
    """
    Count rows where TotalCharges is blank and check they all have tenure = 0.

    Args:
        df: Raw DataFrame.

    Returns:
        Dictionary with the blank row count and the tenure check result.
    """
    blank = pd.to_numeric(df['TotalCharges'], errors='coerce').isna()
    return {
        'blank_total_charges_rows': int(blank.sum()),
        'blank_rows_with_tenure_0': int((df.loc[blank, 'tenure'] == 0).sum()),
        'all_blank_rows_have_tenure_0': bool((df.loc[blank, 'tenure'] == 0).all()),
        'tenure_0_rows_total': int((df['tenure'] == 0).sum())
    }


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

    # TotalCharges is blank for customers with tenure = 0 (nothing billed yet),
    # so a blank means 0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing step: one-hot encode categoricals, scale numerics.

    Returns:
        Unfitted ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('flag', 'passthrough', PASSTHROUGH_COLS)
        ],
        verbose_feature_names_out=False
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split of a cleaned DataFrame.

    Args:
        df: Cleaned DataFrame including the target column.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test). Target is 1 = churned.
    """
    X = df[FEATURE_COLS]
    y = (df[TARGET_COL] == 'Yes').astype(int)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == '__main__':
    raw = load_data()
    print(f"Blank TotalCharges check: {check_blank_total_charges(raw)}")

    X_train, X_test, y_train, y_test = split_data(clean_data(raw))
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
