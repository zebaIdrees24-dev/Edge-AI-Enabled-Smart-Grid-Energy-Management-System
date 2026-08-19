from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import FEATURES
from .models import ModelBundle


@dataclass
class EMSDecision:
    timestamp: str
    forecast_kw: float
    actual_load_kw: float
    anomaly_score: float
    escalated_to_cloud: bool
    action: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveEMSController:
    def __init__(self, bundle: ModelBundle, config: dict):
        self.bundle = bundle
        self.config = config["controller"]

    def decide(self, row: pd.Series) -> EMSDecision:
        x = pd.DataFrame([row[FEATURES].to_dict()])
        edge_prediction = float(self.bundle.edge_forecaster.predict(x)[0])
        x_values = x.to_numpy()
        tree_predictions = np.array(
            [tree.predict(x_values)[0] for tree in self.bundle.edge_forecaster.estimators_]
        )
        uncertainty = float(np.std(tree_predictions) / max(abs(edge_prediction), 1.0))
        raw_score = float(-self.bundle.anomaly_detector.decision_function(x)[0])
        anomaly_probability = float(1 / (1 + np.exp(-5 * raw_score)))
        c = self.config
        safety_violation = not (
            c["grid_frequency_min_hz"] <= row["frequency_hz"] <= c["grid_frequency_max_hz"]
            and c["voltage_min_pu"] <= row["voltage_pu"] <= c["voltage_max_pu"]
        )
        escalate = bool(
            c["cloud_available"]
            and (uncertainty >= c["uncertainty_threshold"] or anomaly_probability >= c["anomaly_threshold"] or safety_violation)
        )
        forecast = float(self.bundle.cloud_forecaster.predict(x)[0]) if escalate else edge_prediction
        net_demand = forecast - float(row["renewable_kw"])
        if safety_violation:
            action, reason = "island_or_protect", "voltage/frequency safety limit violated"
        elif row["battery_soc"] <= c["battery_low_soc"]:
            action, reason = "preserve_battery", "battery state of charge is low"
        elif net_demand > 120 and row["battery_soc"] > 0.25:
            action, reason = "discharge_battery", "forecast net demand is high"
        elif net_demand < -30 and row["battery_soc"] < c["battery_high_soc"]:
            action, reason = "charge_battery", "renewable surplus is available"
        else:
            action, reason = "hold", "supply and demand are within operating band"
        if escalate and not safety_violation:
            reason += "; cloud model selected due to edge uncertainty/anomaly"
        return EMSDecision(
            timestamp=str(row["timestamp"]), forecast_kw=forecast,
            actual_load_kw=float(row["load_kw"]), anomaly_score=anomaly_probability,
            escalated_to_cloud=escalate, action=action, reason=reason,
        )


def run_controller(data: pd.DataFrame, bundle: ModelBundle, config: dict) -> pd.DataFrame:
    """Vectorized batch simulation equivalent to the single-sample edge policy."""
    x = data[FEATURES]
    x_values = x.to_numpy()
    edge_predictions = bundle.edge_forecaster.predict(x)
    tree_predictions = np.vstack(
        [tree.predict(x_values) for tree in bundle.edge_forecaster.estimators_]
    )
    uncertainty = np.std(tree_predictions, axis=0) / np.maximum(np.abs(edge_predictions), 1.0)
    raw_scores = -bundle.anomaly_detector.decision_function(x)
    anomaly_probability = 1 / (1 + np.exp(np.clip(-5 * raw_scores, -50, 50)))
    c = config["controller"]
    safety = ~(
        data["frequency_hz"].between(c["grid_frequency_min_hz"], c["grid_frequency_max_hz"])
        & data["voltage_pu"].between(c["voltage_min_pu"], c["voltage_max_pu"])
    ).to_numpy()
    escalate = c["cloud_available"] & (
        (uncertainty >= c["uncertainty_threshold"])
        | (anomaly_probability >= c["anomaly_threshold"])
        | safety
    )
    forecasts = edge_predictions.copy()
    if np.any(escalate):
        forecasts[escalate] = bundle.cloud_forecaster.predict(x.loc[escalate])
    records = []
    for position, (_, row) in enumerate(data.iterrows()):
        net_demand = forecasts[position] - float(row["renewable_kw"])
        if safety[position]:
            action, reason = "island_or_protect", "voltage/frequency safety limit violated"
        elif row["battery_soc"] <= c["battery_low_soc"]:
            action, reason = "preserve_battery", "battery state of charge is low"
        elif net_demand > 120 and row["battery_soc"] > 0.25:
            action, reason = "discharge_battery", "forecast net demand is high"
        elif net_demand < -30 and row["battery_soc"] < c["battery_high_soc"]:
            action, reason = "charge_battery", "renewable surplus is available"
        else:
            action, reason = "hold", "supply and demand are within operating band"
        if escalate[position] and not safety[position]:
            reason += "; cloud model selected due to edge uncertainty/anomaly"
        records.append(
            EMSDecision(
                timestamp=str(row["timestamp"]),
                forecast_kw=float(forecasts[position]),
                actual_load_kw=float(row["load_kw"]),
                anomaly_score=float(anomaly_probability[position]),
                escalated_to_cloud=bool(escalate[position]),
                action=action,
                reason=reason,
            ).to_dict()
        )
    return pd.DataFrame(records)
