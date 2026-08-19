from __future__ import annotations

import pandas as pd

FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "temperature_c",
    "solar_irradiance_wm2",
    "wind_speed_ms",
    "voltage_pu",
    "frequency_hz",
    "battery_soc",
    "renewable_kw",
    "power_deviation_kw",
    "load_rate_of_change",
    "communication_latency_ms",
    "message_frequency_hz",
    "device_state_transition",
    "load_lag_1",
    "load_lag_4",
    "load_rolling_4",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create causal time-series features shared by edge and cloud models."""
    import numpy as np

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    hour = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["load_lag_1"] = data["load_kw"].shift(1)
    data["load_lag_4"] = data["load_kw"].shift(4)
    data["load_rolling_4"] = data["load_kw"].shift(1).rolling(4).mean()
    data["load_rate_of_change"] = data["load_kw"].shift(1).diff()
    data["power_deviation_kw"] = data["load_lag_1"] - data["renewable_kw"]
    return data.dropna().reset_index(drop=True)
