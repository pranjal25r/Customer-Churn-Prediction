"""
Training pipeline for Customer Churn Prediction.

This module:
- Loads and cleans the data
- Compares models with 5-fold cross-validation on the training set only
- Evaluates the chosen model once on the held-out test set
- Saves the fitted pipeline (preprocessing + model) as one file
- Saves all numbers to reports/metrics.json

Run with: python -m src.train_model
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import (
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    check_blank_total_charges,
    clean_data,
    load_data,
    split_data
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / 'models' / 'churn_model.pkl'
METRICS_PATH = ROOT_DIR / 'reports' / 'metrics.json'

CV_FOLDS = 5

# The "before" reference: no handling of class imbalance. Not a candidate.
BASELINE_NAME = 'random_forest_unweighted'

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    }
}


def get_models(scale_pos_weight: float) -> Dict[str, Pipeline]:
    """
    Build one pipeline (preprocessing + model) per model.

    Args:
        scale_pos_weight: non-churners / churners in the training set (for XGBoost).

    Returns:
        Dictionary mapping model names to unfitted pipelines.
    """
    estimators = {
        'logistic_regression': LogisticRegression(
            class_weight='balanced', **MODEL_CONFIGS['logistic_regression']
        ),
        'random_forest': RandomForestClassifier(
            class_weight='balanced', **MODEL_CONFIGS['random_forest']
        ),
        'xgboost': XGBClassifier(
            scale_pos_weight=scale_pos_weight, **MODEL_CONFIGS['xgboost']
        ),
        BASELINE_NAME: RandomForestClassifier(**MODEL_CONFIGS['random_forest'])
    }

    return {
        name: Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])
        for name, estimator in estimators.items()
    }


def cross_validate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Score each model with stratified k-fold cross-validation on the training set.

    Preprocessing is refitted inside every fold, so no fold sees its own
    validation rows during fitting.

    Args:
        models: Dictionary of unfitted pipelines.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary of model name to churn-class F1 mean, std and fold scores.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        results[name] = {
            'f1_mean': float(scores.mean()),
            'f1_std': float(scores.std()),
            'f1_folds': [float(s) for s in scores]
        }
        logger.info(f"CV F1 {name:26s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline on the test set.

    Args:
        pipeline: Fitted pipeline with predict and predict_proba methods.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Dictionary of metrics, confusion-matrix counts and a plain sentence.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, y_pred).ravel())

    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': {
            'left_and_caught': tp,
            'left_and_missed': fn,
            'stayed_and_flagged': fp,
            'stayed_and_not_flagged': tn
        },
        'confusion_sentence': (
            f"Of {tp + fn} customers who left, caught {tp}; "
            f"flagged {fp} customers who stayed (out of {tn + fp})."
        )
    }


def run_training_pipeline() -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Returns:
        Dictionary with everything that is written to reports/metrics.json.
    """
    logger.info("Loading and cleaning data...")
    raw = load_data()
    blank_check = check_blank_total_charges(raw)
    logger.info(f"Blank TotalCharges check: {blank_check}")

    X_train, X_test, y_train, y_test = split_data(clean_data(raw))
    churners = int(y_train.sum())
    non_churners = int(len(y_train) - churners)
    scale_pos_weight = non_churners / churners

    models = get_models(scale_pos_weight)

    # Choose the model on the training set only
    cv_results = cross_validate_models(models, X_train, y_train)
    candidates = [name for name in models if name != BASELINE_NAME]
    best_name = max(candidates, key=lambda name: cv_results[name]['f1_mean'])
    logger.info(f"Chosen model (highest CV F1): {best_name}")

    # The test set is used once: for the chosen model and the "before" reference
    best_pipeline = models[best_name].fit(X_train, y_train)
    baseline_pipeline = models[BASELINE_NAME].fit(X_train, y_train)
    test_results = {
        best_name: evaluate_model(best_pipeline, X_test, y_test),
        BASELINE_NAME: evaluate_model(baseline_pipeline, X_test, y_test)
    }
    for name, metrics in test_results.items():
        logger.info(f"Test {name}: {metrics['confusion_sentence']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Pipeline saved to: {MODEL_PATH}")

    results = {
        'data': {
            'rows': int(len(raw)),
            'churners': int((raw['Churn'] == 'Yes').sum()),
            **blank_check
        },
        'split': {
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'stratified': True,
            'train_rows': int(len(X_train)),
            'test_rows': int(len(X_test)),
            'train_churners': churners,
            'train_non_churners': non_churners,
            'xgboost_scale_pos_weight': scale_pos_weight
        },
        'cross_validation': {
            'folds': CV_FOLDS,
            'metric': 'F1 for the churn class, training set only',
            'results': cv_results
        },
        'chosen_model': best_name,
        'before_model': BASELINE_NAME,
        'test': test_results
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2) + '\n')
    logger.info(f"Metrics saved to: {METRICS_PATH}")

    return results


if __name__ == '__main__':
    results = run_training_pipeline()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Chosen model: {results['chosen_model']}")
    for name, metrics in results['test'].items():
        print(f"\nTest set - {name}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            print(f"  {metric}: {metrics[metric]:.4f}")
        print(f"  {metrics['confusion_sentence']}")
