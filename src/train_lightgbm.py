"""Training, tuning, and artifact serialization for LightGBM Regressor."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import joblib
import numpy as np
from lightgbm import LGBMRegressor

from src.config import MODELS_DIR, RANDOM_SEED
from src.evaluation import evaluate_predictions


def train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_estimators: int = 150,
    num_leaves: int = 31,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Tuple[LGBMRegressor, Dict[str, Any], np.ndarray]:
    """Train and evaluate LightGBM Regressor for RUL prediction."""
    model = LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    train_duration = time.perf_counter() - start_time

    infer_start = time.perf_counter()
    val_preds = model.predict(X_val)
    infer_duration = time.perf_counter() - infer_start
    infer_latency_ms = (infer_duration / len(X_val)) * 1000.0

    metrics = evaluate_predictions(y_val, val_preds, model_name="LightGBM Regressor", split_name="validation")
    metrics["train_duration_sec"] = round(train_duration, 3)
    metrics["inference_latency_ms"] = round(infer_latency_ms, 4)

    # Feature importances (normalized split importance)
    raw_imp = model.feature_importances_
    total_imp = float(np.sum(raw_imp)) if np.sum(raw_imp) > 0 else 1.0
    importances = dict(zip(feature_names, [round(float(v) / total_imp, 5) for v in raw_imp]))
    metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics["hyperparameters"] = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
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
    model_path = MODELS_DIR / "lightgbm_regressor.joblib"
    joblib.dump(model, model_path)
    metrics["artifact_path"] = str(model_path)
    metrics["artifact_size_kb"] = round(model_path.stat().st_size / 1024, 2)

    return model, metrics, val_preds
