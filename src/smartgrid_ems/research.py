"""Research-style benchmarking, ablation, and efficiency experiments.

This module is intentionally separate from the runtime EMS path so that model
comparison code does not complicate the lightweight edge deployment package.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .anomaly import build_anomaly_detector
from .features import FEATURES


@dataclass
class ForecastResult:
    experiment: str
    model: str
    n_features: int
    mae_kw: float
    rmse_kw: float
    r2: float
    inference_ms_per_sample: float
    serialized_size_kb: float


@dataclass
class AnomalyResult:
    detector: str
    precision: float
    recall: float
    f1: float
    predicted_anomaly_rate: float


def chronological_split(data: pd.DataFrame, train_fraction: float = 0.8):
    """Chronological train/test split to avoid future-to-past leakage."""
    split = max(1, min(len(data) - 1, int(len(data) * train_fraction)))
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def _regression_metrics(y_true, y_pred) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred) ** 0.5),
        float(r2_score(y_true, y_pred)),
    )


def _benchmark_inference(model, x: pd.DataFrame, repeats: int = 5) -> float:
    if len(x) == 0:
        return 0.0
    sample = x.iloc[: min(512, len(x))]
    model.predict(sample.iloc[: min(8, len(sample))])  # warm-up
    times = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000 / len(sample))


def _serialized_size_kb(model) -> float:
    return float(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024)


def build_forecast_models(seed: int = 42) -> dict[str, object]:
    """Representative baselines spanning linear, bagging, and boosting families."""
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.1,
            random_state=seed,
        ),
    }


def evaluate_forecast_models(
    data: pd.DataFrame,
    feature_names: Iterable[str] = FEATURES,
    seed: int = 42,
    experiment: str = "full_features",
) -> list[ForecastResult]:
    """Compare persistence and trainable forecasting baselines on a time split."""
    features = list(feature_names)
    train, test = chronological_split(data)
    y_train, y_test = train["load_kw"], test["load_kw"]
    x_train, x_test = train[features], test[features]
    results: list[ForecastResult] = []

    # A hard-to-beat sanity baseline for short-horizon load forecasting.
    persistence = test["load_lag_1"].to_numpy()
    mae, rmse, r2 = _regression_metrics(y_test, persistence)
    results.append(
        ForecastResult(experiment, "persistence_lag1", len(features), mae, rmse, r2, 0.0, 0.0)
    )

    for name, model in build_forecast_models(seed).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mae, rmse, r2 = _regression_metrics(y_test, pred)
        results.append(
            ForecastResult(
                experiment=experiment,
                model=name,
                n_features=len(features),
                mae_kw=mae,
                rmse_kw=rmse,
                r2=r2,
                inference_ms_per_sample=_benchmark_inference(model, x_test),
                serialized_size_kb=_serialized_size_kb(model),
            )
        )
    return results


ABLATIONS = {
    "full_features": FEATURES,
    "no_temporal_features": [
        f for f in FEATURES
        if f not in {"hour_sin", "hour_cos", "day_of_week", "load_lag_1", "load_lag_4", "load_rolling_4", "load_rate_of_change"}
    ],
    "no_grid_state_features": [
        f for f in FEATURES
        if f not in {"voltage_pu", "frequency_hz", "battery_soc", "renewable_kw", "power_deviation_kw"}
    ],
    "no_communication_features": [
        f for f in FEATURES
        if f not in {"communication_latency_ms", "message_frequency_hz", "device_state_transition"}
    ],
}


def run_ablation_study(data: pd.DataFrame, seed: int = 42) -> list[ForecastResult]:
    """Measure how feature groups affect an Extra Trees forecasting baseline."""
    train, test = chronological_split(data)
    results: list[ForecastResult] = []
    for experiment, features in ABLATIONS.items():
        model = ExtraTreesRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        model.fit(train[features], train["load_kw"])
        pred = model.predict(test[features])
        mae, rmse, r2 = _regression_metrics(test["load_kw"], pred)
        results.append(
            ForecastResult(
                experiment=experiment,
                model="extra_trees",
                n_features=len(features),
                mae_kw=mae,
                rmse_kw=rmse,
                r2=r2,
                inference_ms_per_sample=_benchmark_inference(model, test[features]),
                serialized_size_kb=_serialized_size_kb(model),
            )
        )
    return results


def evaluate_anomaly_detectors(data: pd.DataFrame, seed: int = 42) -> list[AnomalyResult]:
    """Compare unsupervised detectors against held-out synthetic anomaly labels."""
    train, test = chronological_split(data)
    normal_train = train.loc[train["is_anomaly"] == 0, FEATURES]
    y_true = test["is_anomaly"].astype(int).to_numpy()
    results: list[AnomalyResult] = []
    for name in ("statistical", "isolation_forest", "one_class_svm"):
        detector = build_anomaly_detector(name, seed=seed).fit(normal_train)
        # sklearn novelty/outlier convention: -1 = anomaly, +1 = normal
        y_pred = (detector.predict(test[FEATURES]) == -1).astype(int)
        results.append(
            AnomalyResult(
                detector=name,
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                predicted_anomaly_rate=float(np.mean(y_pred)),
            )
        )
    return results




def run_time_series_backtest(data: pd.DataFrame, seed: int = 42, n_splits: int = 4) -> list[dict]:
    """Rolling-origin backtest to quantify stability across multiple time periods."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    x = data[FEATURES]
    y = data["load_kw"]
    model_builders = {
        "ridge": lambda: Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.1,
            random_state=seed,
        ),
    }
    per_model: dict[str, list[tuple[float, float, float]]] = {name: [] for name in model_builders}
    for train_idx, test_idx in splitter.split(x):
        for name, builder in model_builders.items():
            model = builder()
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(x.iloc[test_idx])
            per_model[name].append(_regression_metrics(y.iloc[test_idx], pred))

    rows = []
    for name, scores in per_model.items():
        values = np.asarray(scores)
        rows.append({
            "model": name,
            "folds": n_splits,
            "mae_mean_kw": float(values[:, 0].mean()),
            "mae_std_kw": float(values[:, 0].std(ddof=0)),
            "rmse_mean_kw": float(values[:, 1].mean()),
            "rmse_std_kw": float(values[:, 1].std(ddof=0)),
            "r2_mean": float(values[:, 2].mean()),
            "r2_std": float(values[:, 2].std(ddof=0)),
        })
    return rows


def compute_feature_importance(data: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Train an interpretable tree ensemble and rank feature contribution."""
    train, _ = chronological_split(data)
    model = ExtraTreesRegressor(
        n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    model.fit(train[FEATURES], train["load_kw"])
    rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(FEATURES, model.feature_importances_)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def environment_metadata() -> dict:
    """Capture lightweight reproducibility metadata without adding dependencies."""
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_research_suite(
    data: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    include_lstm: bool = False,
    lstm_epochs: int = 12,
) -> dict:
    """Run model comparison, ablation, anomaly, efficiency, and optional LSTM experiments."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    forecast = evaluate_forecast_models(data, seed=seed)
    ablation = run_ablation_study(data, seed=seed)
    anomaly = evaluate_anomaly_detectors(data, seed=seed)
    backtest = run_time_series_backtest(data, seed=seed)
    importance = compute_feature_importance(data, seed=seed)

    forecast_rows = [asdict(r) for r in forecast]
    ablation_rows = [asdict(r) for r in ablation]
    anomaly_rows = [asdict(r) for r in anomaly]

    if include_lstm:
        from .deep_models import train_lstm_forecaster

        lstm = train_lstm_forecaster(data, seed=seed, epochs=lstm_epochs)
        forecast_rows.append(lstm)

    pd.DataFrame(forecast_rows).to_csv(output / "forecast_benchmarks.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(output / "ablation_study.csv", index=False)
    pd.DataFrame(anomaly_rows).to_csv(output / "anomaly_benchmarks.csv", index=False)
    pd.DataFrame(backtest).to_csv(output / "time_series_backtest.csv", index=False)
    pd.DataFrame(importance).to_csv(output / "feature_importance.csv", index=False)
    (output / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2), encoding="utf-8"
    )

    best_forecast = min(forecast_rows, key=lambda row: row["mae_kw"])
    best_anomaly = max(anomaly_rows, key=lambda row: row["f1"])
    summary = {
        "best_forecast_model": best_forecast,
        "best_anomaly_detector": best_anomaly,
        "forecast_benchmarks": forecast_rows,
        "ablation_study": ablation_rows,
        "anomaly_benchmarks": anomaly_rows,
        "time_series_backtest": backtest,
        "feature_importance": importance,
    }
    (output / "research_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
