"""Training, tuning, and artifact serialization for Random Forest Regressor & Historical Classifier."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import MODELS_DIR, RANDOM_SEED, BASELINE_SELECTED_SENSORS
from src.evaluation import evaluate_predictions


def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = RANDOM_SEED,
) -> Tuple[RandomForestRegressor, Dict[str, Any], np.ndarray]:
    """Train and evaluate Random Forest Regressor for RUL prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_time

    # Inference timing on validation set
    infer_start = time.perf_counter()
    val_preds = model.predict(X_val)
    infer_duration = time.perf_counter() - infer_start
    infer_latency_ms = (infer_duration / len(X_val)) * 1000.0

    metrics = evaluate_predictions(y_val, val_preds, model_name="Random Forest Regressor", split_name="validation")
    metrics["train_duration_sec"] = round(train_duration, 3)
    metrics["inference_latency_ms"] = round(infer_latency_ms, 4)

    # Feature importances
    importances = dict(zip(feature_names, [round(float(v), 5) for v in model.feature_importances_]))
    metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics["hyperparameters"] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state": random_state,
    }

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "random_forest_regressor.joblib"
    joblib.dump(model, model_path)
    metrics["artifact_path"] = str(model_path)
    metrics["artifact_size_kb"] = round(model_path.stat().st_size / 1024, 2)

    return model, metrics, val_preds


def train_historical_rf_classifier(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Reproduces and preserves the Phase 1 Random Forest Classification baseline.
    
    This trains the binary classifier (Safe vs Maintenance Needed: RUL <= 30)
    using the 80:20 split and 100 estimators documented in the Progress Report.
    """
    from sklearn.model_selection import train_test_split

    features = feature_cols or list(BASELINE_SELECTED_SENSORS)
    X = df[features].to_numpy(dtype=float)
    y = df["label_binary"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds).tolist()
    cr = classification_report(y_test, preds, target_names=["Safe (>30)", "Maintenance Needed (<=30)"], output_dict=True)

    # Save baseline artifact
    baseline_path = MODELS_DIR / "rf_classifier_baseline.joblib"
    joblib.dump(clf, baseline_path)

    importances = dict(zip(features, [round(float(v), 5) for v in clf.feature_importances_]))

    return {
        "model_name": "Random Forest Classifier (Phase 1 Baseline)",
        "accuracy": round(acc, 4),
        "accuracy_pct": f"{acc * 100:.2f}%",
        "confusion_matrix": cm,
        "classification_report": cr,
        "feature_importances": dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)),
        "historical_reported_accuracy": "96.05%",
        "artifact_path": str(baseline_path),
        "features_used": features,
    }
