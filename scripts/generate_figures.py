"""Generate portfolio-ready figures for the Smart Grid Edge AI project.

Run from the repository root:
    python scripts/generate_figures.py

The script uses the repository's reproducible synthetic telemetry and trained
models. Figures are written to results/figures/ for embedding in README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartgrid_ems.config import load_config
from smartgrid_ems.controller import run_controller
from smartgrid_ems.data import generate_synthetic_grid_data
from smartgrid_ems.features import FEATURES, engineer_features
from smartgrid_ems.models import train_models
from smartgrid_ems.anomaly import build_anomaly_detector

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    raw = generate_synthetic_grid_data(seed=config["seed"], **config["data"])
    data = engineer_features(raw)
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    # 1) Smart-grid telemetry overview (first week)
    week = raw.iloc[: 7 * 24 * 4].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(week["timestamp"], week["load_kw"], label="Load")
    ax.plot(week["timestamp"], week["renewable_kw"], label="Renewable generation")
    ax.set_title("Smart-grid telemetry: load and renewable generation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "01_grid_telemetry.png")

    # 2) Actual vs predicted load for two representative ML models
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=3,
        random_state=config["seed"], n_jobs=-1,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=config["seed"],
    )
    rf.fit(train[FEATURES], train["load_kw"])
    hgb.fit(train[FEATURES], train["load_kw"])
    n = min(240, len(test))
    sample = test.iloc[:n]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(sample["timestamp"], sample["load_kw"], label="Actual load", linewidth=2)
    ax.plot(sample["timestamp"], rf.predict(sample[FEATURES]), label="Random Forest")
    ax.plot(sample["timestamp"], hgb.predict(sample[FEATURES]), label="HistGradientBoosting")
    ax.set_title("Actual vs predicted electrical load")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (kW)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "02_actual_vs_predicted.png")

    # Load persisted research results for exact benchmark numbers.
    research_dir = ROOT / "reports" / "research"
    forecast = pd.read_csv(research_dir / "forecast_benchmarks.csv")
    anomaly = pd.read_csv(research_dir / "anomaly_benchmarks.csv")
    importance = pd.read_csv(research_dir / "feature_importance.csv")

    # 3) Forecasting model MAE comparison
    chart = forecast.sort_values("mae_kw")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(chart["model"], chart["mae_kw"])
    ax.set_title("Forecasting model comparison")
    ax.set_ylabel("Holdout MAE (kW) — lower is better")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "03_forecast_model_comparison.png")

    # 4) Accuracy-efficiency trade-off. Exclude persistence (not a trained model).
    eff = forecast.loc[forecast["model"] != "persistence_lag1"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter(eff["inference_ms_per_sample"], eff["mae_kw"], s=80)
    for _, row in eff.iterrows():
        ax.annotate(
            row["model"].replace("_", " "),
            (row["inference_ms_per_sample"], row["mae_kw"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_title("Forecast accuracy vs inference latency")
    ax.set_xlabel("Inference latency (ms/sample, log scale)")
    ax.set_ylabel("MAE (kW) — lower is better")
    ax.grid(alpha=0.25)
    save(fig, "04_accuracy_latency_tradeoff.png")

    # 5) Feature importance (top 10)
    top = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top forecasting features")
    ax.set_xlabel("Extra Trees feature importance")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "05_feature_importance.png")

    # 6) Anomaly detector F1 comparison
    a = anomaly.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(a["detector"], a["f1"])
    ax.set_ylim(0, 1)
    ax.set_title("Unsupervised anomaly detector comparison")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Detector")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "06_anomaly_detector_comparison.png")

    # 7) Adaptive edge-cloud routing result
    bundle, _ = train_models(data, config)
    decisions = run_controller(test, bundle, config)
    cloud_rate = 100 * float(decisions["escalated_to_cloud"].mean())
    edge_rate = 100 - cloud_rate
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(["Edge inference", "Cloud escalation"], [edge_rate, cloud_rate])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of decisions (%)")
    ax.set_title("Adaptive edge–cloud routing")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate([edge_rate, cloud_rate]):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")
    save(fig, "07_edge_cloud_routing.png")

    # 8) Anomaly score timeline with synthetic ground-truth anomaly markers
    normal_train = train.loc[train["is_anomaly"] == 0, FEATURES]
    detector = build_anomaly_detector("one_class_svm", seed=config["seed"]).fit(normal_train)
    raw_scores = -detector.decision_function(test[FEATURES])
    # normalize only for visualization; this is not used as a calibrated probability
    score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-12)
    # Select a window containing several ground-truth anomalies.
    idxs = np.flatnonzero(test["is_anomaly"].to_numpy() == 1)
    center = int(idxs[len(idxs)//2]) if len(idxs) else min(200, len(test)//2)
    start = max(0, center - 120)
    stop = min(len(test), start + 260)
    window = test.iloc[start:stop]
    s = score[start:stop]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(window["timestamp"], s, label="Normalized anomaly score")
    mask = window["is_anomaly"].to_numpy().astype(bool)
    if mask.any():
        ax.scatter(window.loc[mask, "timestamp"], s[mask], marker="x", s=65, label="Injected anomaly")
    ax.set_title("Anomaly score over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "08_anomaly_timeline.png")

    print(f"Generated 8 figures in {OUT}")


if __name__ == "__main__":
    main()
