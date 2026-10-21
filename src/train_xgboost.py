"""Training, tuning, and artifact serialization for XGBoost Regressor."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import joblib
import numpy as np
from xgboost import XGBRegressor

from src.config import MODELS_DIR, RANDOM_SEED
from src.evaluation import evaluate_predictions


def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Tuple[XGBRegressor, Dict[str, Any], np.ndarray]:
    """Train and evaluate XGBoost Regressor for RUL prediction."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_duration = time.perf_counter() - start_time

    infer_start = time.perf_counter()
    val_preds = model.predict(X_val)
    infer_duration = time.perf_counter() - infer_start
    infer_latency_ms = (infer_duration / len(X_val)) * 1000.0

    metrics = evaluate_predictions(y_val, val_preds, model_name="XGBoost Regressor", split_name="validation")
    metrics["train_duration_sec"] = round(train_duration, 3)
    metrics["inference_latency_ms"] = round(infer_latency_ms, 4)

    # Feature importances
    importances = dict(zip(feature_names, [round(float(v), 5) for v in model.feature_importances_]))
    metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics["hyperparameters"] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": random_state,
    }

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_regressor.joblib"
    joblib.dump(model, model_path)
    metrics["artifact_path"] = str(model_path)
    metrics["artifact_size_kb"] = round(model_path.stat().st_size / 1024, 2)

    return model, metrics, val_preds
