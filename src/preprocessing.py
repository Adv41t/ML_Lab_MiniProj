"""Data preprocessing, engine-level splitting, feature scaling, and leakage prevention."""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import (
    CONSTANT_SENSORS_FD001,
    BASELINE_SELECTED_SENSORS,
    ALL_INFORMATIVE_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
)


def detect_constant_columns(df: pd.DataFrame, threshold: float = 1e-5) -> List[str]:
    """Identify columns with variance or standard deviation below the threshold."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stds = df[numeric_cols].std()
    constant_cols = stds[stds < threshold].index.tolist()
    return constant_cols


def split_train_val_by_engine(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Engine-aware validation split to strictly prevent data leakage.
    
    All cycles for a given engine are kept strictly together in either
    the training set or the validation set.
    """
    unique_engines = np.sort(df["engine_id"].unique())
    num_val_engines = int(np.round(len(unique_engines) * val_ratio))

    rng = np.random.default_rng(seed)
    shuffled_engines = rng.permutation(unique_engines)

    val_engine_ids = sorted(shuffled_engines[:num_val_engines].tolist())
    train_engine_ids = sorted(shuffled_engines[num_val_engines:].tolist())

    train_df = df[df["engine_id"].isin(train_engine_ids)].copy().reset_index(drop=True)
    val_df = df[df["engine_id"].isin(val_engine_ids)].copy().reset_index(drop=True)

    return train_df, val_df, train_engine_ids, val_engine_ids


class Preprocessor:
    """Handles feature selection and strictly training-fitted scaling."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        scaler_type: str = "minmax",
    ):
        self.feature_cols = feature_cols or list(BASELINE_SELECTED_SENSORS)
        self.scaler_type = scaler_type.lower()
        self.scaler = MinMaxScaler() if self.scaler_type == "minmax" else StandardScaler()
        self.is_fitted = False
        self.feature_min_: Optional[Dict[str, float]] = None
        self.feature_max_: Optional[Dict[str, float]] = None

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Fit scaler strictly on training engine data."""
        missing_features = [col for col in self.feature_cols if col not in train_df.columns]
        if missing_features:
            raise KeyError(f"Training data is missing required feature columns: {missing_features}")

        X_train = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        self.scaler.fit(X_train)
        self.is_fitted = True

        if hasattr(self.scaler, "data_min_") and hasattr(self.scaler, "data_max_"):
            self.feature_min_ = dict(zip(self.feature_cols, self.scaler.data_min_))
            self.feature_max_ = dict(zip(self.feature_cols, self.scaler.data_max_))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling while preserving dataframe structure."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")

        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing required feature columns: {missing}")

        df_out = df.copy()
        X_scaled = self.scaler.transform(df[self.feature_cols].to_numpy(dtype=np.float32))

        for idx, col in enumerate(self.feature_cols):
            df_out[col] = X_scaled[:, idx]

        return df_out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train and transform."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Optional[Path | str] = None) -> str:
        """Serialize preprocessor to disk."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return str(filepath)

    @classmethod
    def load(cls, filepath: Optional[Path | str] = None) -> "Preprocessor":
        """Load preprocessor from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")

        obj = joblib.load(filepath)
        return obj
