"""
Training pipeline for Customer Churn Prediction.

This module provides a complete training pipeline that:
- Loads and preprocesses data
- Trains RandomForest and XGBoost models
- Evaluates using multiple metrics
- Saves the best model
- Saves feature importance visualization for the best model
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Only RandomForest will be trained.")

from src.data_processing import get_processed_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
}


def get_models() -> Dict[str, Any]:
    """
    Initialize and return all models to be trained.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {
        'random_forest': RandomForestClassifier(**MODEL_CONFIGS['random_forest'])
    }

    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(**MODEL_CONFIGS['xgboost'])
    else:
        logger.warning("XGBoost not available, skipping...")

    return models


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model using multiple metrics.

    Args:
        model: Trained model with predict and predict_proba methods.
        X_test: Test features.
        y_test: True labels.
        model_name: Name for logging purposes.

    Returns:
        Dictionary of metric names to values.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} Evaluation Results")
    logger.info(f"{'='*50}")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.upper():12s}: {value:.4f}")

    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> Any:
    """
    Train a single model.

    Args:
        model: Untrained model instance.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name for logging purposes.

    Returns:
        Trained model.
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training completed.")
    return model


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train and evaluate all models.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Tuple of (trained models dict, metrics dict).
    """
    models = get_models()
    trained_models = {}
    all_metrics = {}

    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train, name)
        metrics = evaluate_model(trained_model, X_test, y_test, name)

        trained_models[name] = trained_model
        all_metrics[name] = metrics

    return trained_models, all_metrics


def select_best_model(
    trained_models: Dict[str, Any],
    all_metrics: Dict[str, Dict[str, float]],
    selection_metric: str = 'f1_score'
) -> Tuple[str, Any]:
    """
    Select the best model based on a specified metric.

    Args:
        trained_models: Dictionary of trained models.
        all_metrics: Dictionary of metrics for each model.
        selection_metric: Metric to use for selection.

    Returns:
        Tuple of (best model name, best model instance).
    """
    best_name = max(
        all_metrics.keys(),
        key=lambda x: all_metrics[x][selection_metric]
    )
    best_score = all_metrics[best_name][selection_metric]

    logger.info(f"\n{'='*50}")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"Selection Metric ({selection_metric}): {best_score:.4f}")
    logger.info(f"{'='*50}")

    return best_name, trained_models[best_name]


def save_model(
    model: Any,
    filepath: str,
    artifacts: Optional[Dict] = None
) -> None:
    """
    Save model and artifacts using joblib.

    Args:
        model: Trained model to save.
        filepath: Path to save the model.
        artifacts: Optional preprocessing artifacts to save with model.
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    save_object = {
        'model': model,
        'artifacts': artifacts
    }

    joblib.dump(save_object, filepath)
    logger.info(f"Model saved to: {filepath}")


def save_feature_importance_plot(
    model: Any,
    feature_names: list,
    filepath: str,
    top_n: int = 15
) -> Optional[str]:
    """
    Save a feature importance bar plot for models that expose feature_importances_.

    Args:
        model: Trained model instance.
        feature_names: List of feature names used in training.
        filepath: Output path for the plot image.
        top_n: Number of top features to visualize.

    Returns:
        Saved file path if plot is created, otherwise None.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not provide feature_importances_. Skipping plot.")
        return None

    importances = model.feature_importances_

    # Keep only top N features for readability.
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    feature_importance_df = feature_importance_df.head(top_n)
    feature_importance_df = feature_importance_df.iloc[::-1]

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.title(f'Top {len(feature_importance_df)} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"Feature importance plot saved to: {filepath}")
    return filepath


def load_model(filepath: str) -> Tuple[Any, Optional[Dict]]:
    """
    Load model and artifacts from file.

    Args:
        filepath: Path to the saved model.

    Returns:
        Tuple of (model, artifacts dict).
    """
    save_object = joblib.load(filepath)
    return save_object['model'], save_object.get('artifacts')


def run_training_pipeline(
    data_dir: Optional[str] = None,
    model_output_path: Optional[str] = None,
    selection_metric: str = 'f1_score'
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Args:
        data_dir: Directory containing the data file.
        model_output_path: Path to save the best model.
        selection_metric: Metric for model selection.

    Returns:
        Dictionary containing training results and metadata.
    """
    logger.info("Starting training pipeline...")

    # Set default model output path
    if model_output_path is None:
        model_output_path = str(
            Path(__file__).parent.parent / 'models' / 'churn_model.pkl'
        )

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, artifacts = get_processed_data(data_dir)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {len(artifacts['feature_names'])}")

    # Train and evaluate all models
    trained_models, all_metrics = train_and_evaluate_all(
        X_train, X_test, y_train, y_test
    )

    # Select best model
    best_name, best_model = select_best_model(
        trained_models, all_metrics, selection_metric
    )

    # Save the best model
    save_model(best_model, model_output_path, artifacts)

    # Save feature importance plot for the selected model
    feature_importance_path = str(
        Path(model_output_path).parent / f'feature_importance_{best_name}.png'
    )
    saved_plot_path = save_feature_importance_plot(
        best_model,
        artifacts.get('feature_names', []),
        feature_importance_path
    )

    # Print comparison summary
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    comparison_df = pd.DataFrame(all_metrics).T
    logger.info(f"\n{comparison_df.to_string()}")

    return {
        'best_model_name': best_name,
        'best_model': best_model,
        'all_metrics': all_metrics,
        'model_path': model_output_path,
        'feature_importance_plot_path': saved_plot_path,
        'artifacts': artifacts
    }


if __name__ == '__main__':
    results = run_training_pipeline()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Saved to: {results['model_path']}")
    if results.get('feature_importance_plot_path'):
        print(f"Feature Importance Plot: {results['feature_importance_plot_path']}")
    print(f"\nMetrics for best model:")
    for metric, value in results['all_metrics'][results['best_model_name']].items():
        print(f"  {metric}: {value:.4f}")
