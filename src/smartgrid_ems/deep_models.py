"""Optional PyTorch models for anomaly detection and time-series forecasting experiments."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[deep-learning]'") from exc
    return torch, nn


def build_torch_autoencoder(input_dim: int, latent_dim: int = 4):
    """Construct a compact autoencoder for edge/cloud anomaly experiments."""
    _, nn = _torch()

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = max(latent_dim * 2, input_dim // 2)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
            )

        def forward(self, values):
            return self.decoder(self.encoder(values))

    return Autoencoder()


def build_lstm_forecaster(input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
    """Construct a compact LSTM regressor for multivariate load forecasting."""
    _, nn = _torch()

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

        def forward(self, values):
            output, _ = self.lstm(values)
            return self.head(output[:, -1, :]).squeeze(-1)

    return LSTMForecaster()


def _make_sequences(x: np.ndarray, y: np.ndarray, sequence_length: int):
    xs, ys = [], []
    for idx in range(sequence_length, len(x)):
        xs.append(x[idx - sequence_length : idx])
        ys.append(y[idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_lstm_forecaster(
    data: pd.DataFrame,
    seed: int = 42,
    sequence_length: int = 12,
    hidden_dim: int = 32,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """Train/evaluate a reproducible LSTM on the chronological holdout split.

    Scaling is fitted only on the training period. The sequence split preserves
    temporal ordering and does not shuffle the validation/test horizon.
    """
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    split = max(sequence_length + 2, min(len(data) - 1, int(len(data) * 0.8)))
    train = data.iloc[:split].copy()
    # Include a short history window before the holdout boundary for causal context.
    test_context = data.iloc[split - sequence_length :].copy()

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train_scaled = x_scaler.fit_transform(train[FEATURES])
    y_train_scaled = y_scaler.fit_transform(train[["load_kw"]]).ravel()
    x_test_scaled = x_scaler.transform(test_context[FEATURES])
    y_test_raw = test_context["load_kw"].to_numpy()

    x_train, y_train = _make_sequences(x_train_scaled, y_train_scaled, sequence_length)
    x_test, y_test = _make_sequences(x_test_scaled, y_test_raw, sequence_length)

    model = build_lstm_forecaster(len(FEATURES), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    model.train()

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    x_test_tensor = torch.from_numpy(x_test)
    with torch.no_grad():
        pred_scaled = model(x_test_tensor).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # Warm-up and median latency estimate.
    with torch.no_grad():
        model(x_test_tensor[: min(8, len(x_test_tensor))])
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            model(x_test_tensor)
            elapsed.append(time.perf_counter() - start)
    latency = float(np.median(elapsed) * 1000 / max(len(x_test_tensor), 1))
    size_kb = float(len(pickle.dumps(model.state_dict())) / 1024)

    return {
        "experiment": "full_features",
        "model": "lstm",
        "n_features": len(FEATURES),
        "mae_kw": float(mean_absolute_error(y_test, pred)),
        "rmse_kw": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
        "inference_ms_per_sample": latency,
        "serialized_size_kb": size_kb,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
    }
