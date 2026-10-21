"""Sliding-window sequence generation for sequential/temporal deep learning (LSTM)."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import SEQUENCE_LENGTH


def build_sequences_for_engine(
    engine_df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = SEQUENCE_LENGTH,
    target_col: Optional[str] = "RUL_clipped",
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    """Generate chronological sliding-window sequences for a single engine.
    
    Guarantees that sequences never bridge across different engines.
    Target associated with each window is the RUL at the LAST cycle of the window.
    """
    sorted_df = engine_df.sort_values(by="cycle").reset_index(drop=True)
    num_records = len(sorted_df)

    if num_records < sequence_length:
        return (
            np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32) if target_col else None,
            [],
        )

    feature_matrix = sorted_df[feature_cols].to_numpy(dtype=np.float32)
    engine_id = int(sorted_df["engine_id"].iloc[0])
    cycles = sorted_df["cycle"].to_numpy(dtype=int)

    targets = sorted_df[target_col].to_numpy(dtype=np.float32) if target_col and target_col in sorted_df else None

    sequences = []
    seq_targets = []
    metadata = []

    num_windows = num_records - sequence_length + 1
    for i in range(num_windows):
        window = feature_matrix[i : i + sequence_length]
        sequences.append(window)
        final_cycle = int(cycles[i + sequence_length - 1])

        if targets is not None:
            seq_targets.append(targets[i + sequence_length - 1])

        metadata.append({
            "engine_id": engine_id,
            "start_cycle": int(cycles[i]),
            "end_cycle": final_cycle,
        })

    X_arr = np.array(sequences, dtype=np.float32)
    y_arr = np.array(seq_targets, dtype=np.float32) if targets is not None else None

    return X_arr, y_arr, metadata


def build_all_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = SEQUENCE_LENGTH,
    target_col: Optional[str] = "RUL_clipped",
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    """Build chronological sequences across all engines, isolating each engine completely."""
    all_X = []
    all_y = []
    all_meta = []

    for engine_id, engine_df in df.groupby("engine_id"):
        X_eng, y_eng, meta_eng = build_sequences_for_engine(
            engine_df=engine_df,
            feature_cols=feature_cols,
            sequence_length=sequence_length,
            target_col=target_col,
        )
        if len(X_eng) > 0:
            all_X.append(X_eng)
            if y_eng is not None:
                all_y.append(y_eng)
            all_meta.extend(meta_eng)

    if not all_X:
        raise ValueError(
            f"No valid sequences could be generated with sequence_length={sequence_length}. "
            "Engines may have fewer cycles than the required sequence length."
        )

    X_out = np.vstack(all_X)
    y_out = np.concatenate(all_y) if all_y else None

    return X_out, y_out, all_meta


def build_test_final_sequences(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """For each test engine, extract exactly ONE sequence: the final window of length sequence_length.
    
    This matches the NASA C-MAPSS official benchmark evaluation protocol,
    yielding 1 sequence per test engine corresponding to its maximum observed cycle.
    """
    final_seqs = []
    meta_list = []

    unique_engines = sorted(test_df["engine_id"].unique())
    for engine_id in unique_engines:
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values(by="cycle").reset_index(drop=True)
        num_cycles = len(engine_df)

        if num_cycles < sequence_length:
            raise ValueError(
                f"Test engine {engine_id} has {num_cycles} cycles, which is less than "
                f"required sequence_length {sequence_length}."
            )

        # Slice the last sequence_length cycles
        window_df = engine_df.iloc[-sequence_length:]
        window_features = window_df[feature_cols].to_numpy(dtype=np.float32)

        final_seqs.append(window_features)
        meta_list.append({
            "engine_id": int(engine_id),
            "max_cycle": int(engine_df["cycle"].max()),
            "start_cycle": int(window_df["cycle"].min()),
            "end_cycle": int(window_df["cycle"].max()),
        })

    return np.array(final_seqs, dtype=np.float32), meta_list
