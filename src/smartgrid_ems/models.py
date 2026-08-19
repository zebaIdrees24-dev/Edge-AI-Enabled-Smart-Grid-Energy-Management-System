from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from .anomaly import build_anomaly_detector
from .features import FEATURES


@dataclass
class ModelBundle:
    edge_forecaster: RandomForestRegressor
    cloud_forecaster: ExtraTreesRegressor
    anomaly_detector: Pipeline
    edge_residual_scale: float


def train_models(data: pd.DataFrame, config: dict) -> tuple[ModelBundle, dict[str, float]]:
    split = max(1, int(len(data) * 0.8))
    train, test = data.iloc[:split], data.iloc[split:]
    x_train, y_train = train[FEATURES], train["load_kw"]
    x_test, y_test = test[FEATURES], test["load_kw"]
    model_cfg = config["models"]
    seed = config["seed"]
    edge = RandomForestRegressor(
        n_estimators=model_cfg["edge_trees"], max_depth=8, min_samples_leaf=3,
        random_state=seed, n_jobs=-1,
    )
    cloud = ExtraTreesRegressor(
        n_estimators=model_cfg["cloud_trees"], max_depth=model_cfg["cloud_max_depth"],
        min_samples_leaf=2, random_state=seed, n_jobs=-1,
    )
    edge.fit(x_train, y_train)
    cloud.fit(x_train, y_train)
    normal = train.loc[train["is_anomaly"] == 0, FEATURES]
    anomaly = build_anomaly_detector(model_cfg.get("anomaly_detector", "isolation_forest"), seed)
    anomaly.fit(normal)
    edge_train_residual = np.abs(y_train.to_numpy() - edge.predict(x_train))
    bundle = ModelBundle(edge, cloud, anomaly, float(np.quantile(edge_train_residual, 0.95) + 1e-6))
    edge_pred, cloud_pred = edge.predict(x_test), cloud.predict(x_test)
    metrics = {
        "edge_mae_kw": float(mean_absolute_error(y_test, edge_pred)),
        "edge_rmse_kw": float(mean_squared_error(y_test, edge_pred) ** 0.5),
        "cloud_mae_kw": float(mean_absolute_error(y_test, cloud_pred)),
        "cloud_rmse_kw": float(mean_squared_error(y_test, cloud_pred) ** 0.5),
    }
    return bundle, metrics


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)
