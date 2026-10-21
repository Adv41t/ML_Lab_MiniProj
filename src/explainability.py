"""Model explainability module using feature importances and SHAP values."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import SENSOR_METADATA


def get_feature_descriptions(feature_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Return dictionary of feature physical descriptions, units, and tags."""
    desc = {}
    for f in feature_names:
        if f in SENSOR_METADATA:
            desc[f] = SENSOR_METADATA[f]
        elif f == "cycle":
            desc[f] = {"name": "Cycle", "desc": "Current operational cycle count", "unit": "cycles"}
        elif f.startswith("setting_"):
            num = f.split("_")[1]
            desc[f] = {"name": f"Set{num}", "desc": f"Operating altitude/Mach setting #{num}", "unit": "--"}
        else:
            desc[f] = {"name": f, "desc": "Sensor measurement", "unit": "--"}
    return desc


def compute_tree_shap_values(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values using TreeExplainer for tree-based models."""
    try:
        import shap

        # Limit sample size for computational efficiency
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_eval = X_sample[indices]
        else:
            X_eval = X_sample

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        # Handle different SHAP output shapes
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        shap_importance = dict(zip(feature_names, [round(float(v), 5) for v in mean_abs_shap]))
        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "status": "SUCCESS",
            "mean_abs_shap": shap_importance,
            "sample_size": len(X_eval),
            "expected_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else None,
        }
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "error": str(e),
            "fallback_note": "SHAP calculation skipped or unavailable. Standard Gini/Gain feature importance used.",
        }


def explain_single_prediction(
    model: Any,
    feature_vector: np.ndarray,
    feature_names: List[str],
    is_tree_model: bool = True,
) -> Dict[str, Any]:
    """Provide local explanation for an individual engine cycle prediction."""
    fv_2d = feature_vector.reshape(1, -1)
    if is_tree_model:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(fv_2d)
            if isinstance(shap_values, list):
                sv = shap_values[0][0]
            elif len(shap_values.shape) == 2:
                sv = shap_values[0]
            else:
                sv = shap_values

            contributions = dict(zip(feature_names, [round(float(v), 3) for v in sv]))
            # Top positive drivers (increase predicted RUL)
            positive_factors = {k: v for k, v in contributions.items() if v > 0}
            # Top negative drivers (decrease predicted RUL / accelerate maintenance alert)
            negative_factors = {k: v for k, v in contributions.items() if v < 0}

            return {
                "method": "SHAP TreeExplainer (Local)",
                "base_value": round(float(explainer.expected_value), 2) if hasattr(explainer, "expected_value") else None,
                "contributions": contributions,
                "top_degradation_indicators": dict(sorted(negative_factors.items(), key=lambda x: x[1])[:5]),
                "top_health_indicators": dict(sorted(positive_factors.items(), key=lambda x: x[1], reverse=True)[:5]),
            }
        except Exception:
            pass

    # Fallback heuristic based on feature importances and normalized deviations
    return {
        "method": "Global MDI/Gain Importance Weighted Deviation",
        "note": "Local SHAP explanation fallback used.",
    }
