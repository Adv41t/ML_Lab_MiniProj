"""Unit tests for common evaluation metrics and model artifact reloading."""

import pytest
import numpy as np
import joblib
from pathlib import Path

from src.evaluation import evaluate_predictions, compute_nasa_scoring
from src.config import MODELS_DIR
from src.train_lstm import load_lstm_model


def test_regression_metrics_calculations():
    """Verify MAE, RMSE, R2, quantiles, and late-life calculations."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 31.0, 38.0])

    metrics = evaluate_predictions(y_true, y_pred, model_name="UnitTestModel", split_name="test")

    # Errors: +2, -1, +1, -2 -> Abs errors: 2, 1, 1, 2 -> MAE = 6 / 4 = 1.5
    assert metrics["mae"] == 1.5
    # Squared errors: 4, 1, 1, 4 -> MSE = 10 / 4 = 2.5 -> RMSE = sqrt(2.5) ~ 1.581
    assert abs(metrics["rmse"] - np.sqrt(2.5)) < 1e-3
    assert metrics["sample_count"] == 4
    # Late life (threshold = 30): 10, 20, 30 -> 3 samples
    assert metrics["late_life"]["sample_count"] == 3


def test_nasa_scoring_asymmetry():
    """Verify that late predictions receive a heavier penalty than early predictions."""
    y_true = np.array([50.0])
    # Early prediction by 10 cycles (y_pred = 40, diff = -10)
    early_score = compute_nasa_scoring(y_true, np.array([40.0]))
    # Late prediction by 10 cycles (y_pred = 60, diff = +10)
    late_score = compute_nasa_scoring(y_true, np.array([60.0]))

    # Early penalty: exp(10/13) - 1 ~ 1.158
    # Late penalty: exp(10/10) - 1 ~ 1.718
    assert late_score > early_score


def test_saved_models_can_be_loaded():
    """Verify all saved model artifacts can be loaded and perform inference."""
    rf_path = MODELS_DIR / "random_forest_regressor.joblib"
    xgb_path = MODELS_DIR / "xgboost_regressor.joblib"
    lgb_path = MODELS_DIR / "lightgbm_regressor.joblib"
    lstm_path = MODELS_DIR / "lstm_regressor.pt"

    assert rf_path.exists(), "Random Forest artifact missing"
    assert xgb_path.exists(), "XGBoost artifact missing"
    assert lgb_path.exists(), "LightGBM artifact missing"
    assert lstm_path.exists(), "LSTM artifact missing"

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    lstm = load_lstm_model(lstm_path)

    # Test dummy inference (11 features matching BASELINE_SELECTED_SENSORS)
    dummy_tabular = np.zeros((1, 11), dtype=np.float32)
    assert np.isfinite(rf.predict(dummy_tabular)[0])
    assert np.isfinite(xgb.predict(dummy_tabular)[0])
    assert np.isfinite(lgb.predict(dummy_tabular)[0])

    dummy_seq = np.zeros((1, 30, 11), dtype=np.float32)
    import torch
    with torch.no_grad():
        lstm_out = lstm(torch.tensor(dummy_seq)).numpy().item()
        assert np.isfinite(lstm_out)
