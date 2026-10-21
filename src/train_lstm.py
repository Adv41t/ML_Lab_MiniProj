"""Training, evaluation, and artifact serialization for PyTorch LSTM Regressor."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import MODELS_DIR, RANDOM_SEED, SEQUENCE_LENGTH
from src.evaluation import evaluate_predictions


class TurbofanLSTM(nn.Module):
    """Deep Recurrent Neural Network with LSTM layers for RUL regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim_1,
            hidden_size=hidden_dim_2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim_2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, (hn, cn) = self.lstm2(out1)
        out2 = self.dropout2(out2)
        # Take the output of the last time step
        last_step = out2[:, -1, :]
        h = self.relu(self.fc1(last_step))
        out = self.fc2(h)
        return out.squeeze(-1)


def train_lstm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    sequence_length: int = SEQUENCE_LENGTH,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    hidden_dim_1: int = 64,
    hidden_dim_2: int = 32,
    dropout: float = 0.2,
    patience: int = 7,
    random_state: int = RANDOM_SEED,
) -> Tuple[TurbofanLSTM, Dict[str, Any], np.ndarray]:
    """Train PyTorch LSTM Regressor with early stopping, recording loss curves."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TurbofanLSTM(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_abs_accum = 0.0
        total_train_samples = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            batch_count = len(by)
            train_loss_accum += loss.item() * batch_count
            train_abs_accum += torch.abs(preds - by).sum().item()
            total_train_samples += batch_count

        epoch_train_loss = train_loss_accum / total_train_samples
        epoch_train_mae = train_abs_accum / total_train_samples

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_abs_accum = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)

                batch_count = len(by)
                val_loss_accum += loss.item() * batch_count
                val_abs_accum += torch.abs(preds - by).sum().item()
                total_val_samples += batch_count

        epoch_val_loss = val_loss_accum / total_val_samples
        epoch_val_mae = val_abs_accum / total_val_samples

        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_mae"].append(round(epoch_train_mae, 4))
        history["val_mae"].append(round(epoch_val_mae, 4))

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_duration = time.perf_counter() - start_time

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(device)

    # Validation evaluation & inference timing
    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(val_tensor).cpu().numpy()
    infer_duration = time.perf_counter() - infer_start
    infer_latency_ms = (infer_duration / len(X_val)) * 1000.0

    metrics = evaluate_predictions(y_val, val_preds, model_name="LSTM Regressor", split_name="validation")
    metrics["train_duration_sec"] = round(train_duration, 3)
    metrics["inference_latency_ms"] = round(infer_latency_ms, 4)
    metrics["training_device"] = str(device).upper()
    metrics["training_history"] = history
    metrics["stopped_epoch"] = len(history["epochs"])

    metrics["hyperparameters"] = {
        "sequence_length": sequence_length,
        "input_dim": input_dim,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "device": str(device),
    }

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lstm_regressor.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "dropout": dropout,
        "feature_names": feature_names,
        "sequence_length": sequence_length,
    }, model_path)

    metrics["artifact_path"] = str(model_path)
    metrics["artifact_size_kb"] = round(model_path.stat().st_size / 1024, 2)

    return model, metrics, val_preds


def load_lstm_model(model_path: Optional[Path | str] = None, device: Optional[str] = None) -> TurbofanLSTM:
    """Load persisted PyTorch LSTM model."""
    if model_path is None:
        model_path = MODELS_DIR / "lstm_regressor.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model checkpoint not found at: {model_path}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location=dev)

    model = TurbofanLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim_1=checkpoint["hidden_dim_1"],
        hidden_dim_2=checkpoint["hidden_dim_2"],
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(dev)
    model.eval()
    return model
