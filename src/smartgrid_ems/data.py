from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_grid_data(
    rows: int = 2880,
    interval_minutes: int = 15,
    anomaly_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible EMS telemetry; intended for demos, not grid studies."""
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2025-01-01", periods=rows, freq=f"{interval_minutes}min", tz="UTC")
    hour = timestamp.hour.to_numpy() + timestamp.minute.to_numpy() / 60
    day = np.arange(rows) * interval_minutes / (24 * 60)
    temperature = 18 + 8 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 1.5, rows)
    sunlight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    irradiance = 850 * sunlight + rng.normal(0, 30, rows)
    irradiance = np.clip(irradiance, 0, None)
    wind = np.clip(5 + 2 * np.sin(2 * np.pi * day / 3) + rng.normal(0, 1.2, rows), 0, None)
    solar_kw = irradiance * 0.22
    wind_kw = 12 * wind
    renewable = solar_kw + wind_kw
    morning = 90 * np.exp(-((hour - 8) / 2.2) ** 2)
    evening = 150 * np.exp(-((hour - 19) / 2.8) ** 2)
    cooling = 4 * np.maximum(temperature - 23, 0)
    load = 260 + morning + evening + cooling + rng.normal(0, 12, rows)
    battery_soc = np.clip(0.55 + 0.25 * np.sin(2 * np.pi * (hour - 11) / 24), 0.1, 0.95)
    frequency = 50 + rng.normal(0, 0.035, rows)
    voltage = 1.0 + rng.normal(0, 0.012, rows)
    latency = np.clip(rng.lognormal(mean=3.2, sigma=0.35, size=rows), 2, 500)
    message_frequency = np.clip(rng.normal(4.0, 0.15, rows), 0.1, None)
    device_state_transition = rng.binomial(1, 0.025, rows)
    anomaly = np.zeros(rows, dtype=int)
    count = max(1, int(rows * anomaly_rate))
    indices = rng.choice(np.arange(8, rows), size=count, replace=False)
    anomaly[indices] = 1
    split = count // 3
    load[indices[:split]] *= rng.uniform(1.35, 1.8, split)
    frequency[indices[split : 2 * split]] += rng.choice([-1, 1], split) * rng.uniform(0.55, 1.0, split)
    remaining = indices[2 * split :]
    voltage[remaining] += rng.choice([-1, 1], len(remaining)) * rng.uniform(0.12, 0.25, len(remaining))
    latency[indices] *= rng.uniform(2, 8, count)
    message_frequency[indices] *= rng.uniform(0.1, 2.5, count)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "load_kw": load,
            "renewable_kw": renewable,
            "temperature_c": temperature,
            "solar_irradiance_wm2": irradiance,
            "wind_speed_ms": wind,
            "voltage_pu": voltage,
            "frequency_hz": frequency,
            "battery_soc": battery_soc,
            "communication_latency_ms": latency,
            "message_frequency_hz": message_frequency,
            "device_state_transition": device_state_transition,
            "is_anomaly": anomaly,
        }
    )


def save_data(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
