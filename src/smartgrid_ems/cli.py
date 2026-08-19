from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .controller import run_controller
from .data import generate_synthetic_grid_data, save_data
from .features import engineer_features
from .models import load_bundle, save_bundle, train_models
from .research import run_research_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive edge-cloud smart-grid EMS")
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate synthetic smart-grid telemetry")
    generate.add_argument("--output", default="data/grid_telemetry.csv")
    train = sub.add_parser("train", help="Train edge, cloud, and anomaly models")
    train.add_argument("--data", default="data/grid_telemetry.csv")
    train.add_argument("--model", default="artifacts/models.joblib")
    train.add_argument("--metrics", default="reports/training_metrics.json")
    simulate = sub.add_parser("simulate", help="Run adaptive edge-cloud decisions")
    simulate.add_argument("--data", default="data/grid_telemetry.csv")
    simulate.add_argument("--model", default="artifacts/models.joblib")
    simulate.add_argument("--output", default="reports/decisions.csv")
    research = sub.add_parser("research", help="Run ML research benchmarks and ablations")
    research.add_argument("--data", default="data/grid_telemetry.csv")
    research.add_argument("--output", default="reports/research")
    research.add_argument("--include-lstm", action="store_true")
    research.add_argument("--lstm-epochs", type=int, default=12)
    sub.add_parser("demo", help="Run generation, training, and simulation")
    return parser


def _generate(config: dict, output: str) -> pd.DataFrame:
    params = config["data"] | {"seed": config["seed"]}
    frame = generate_synthetic_grid_data(**params)
    save_data(frame, output)
    return frame


def _train(config: dict, data_path: str, model_path: str, metrics_path: str):
    frame = engineer_features(pd.read_csv(data_path))
    bundle, metrics = train_models(frame, config)
    save_bundle(bundle, model_path)
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _simulate(config: dict, data_path: str, model_path: str, output: str):
    frame = engineer_features(pd.read_csv(data_path))
    decisions = run_controller(frame, load_bundle(model_path), config)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(path, index=False)
    summary = {
        "rows": len(decisions),
        "cloud_escalation_rate": float(decisions["escalated_to_cloud"].mean()),
        "actions": decisions["action"].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        _generate(config, args.output)
    elif args.command == "train":
        _train(config, args.data, args.model, args.metrics)
    elif args.command == "simulate":
        _simulate(config, args.data, args.model, args.output)
    elif args.command == "research":
        frame = engineer_features(pd.read_csv(args.data))
        summary = run_research_suite(
            frame,
            args.output,
            seed=config["seed"],
            include_lstm=args.include_lstm,
            lstm_epochs=args.lstm_epochs,
        )
        print(json.dumps({
            "best_forecast_model": summary["best_forecast_model"],
            "best_anomaly_detector": summary["best_anomaly_detector"],
            "output_dir": args.output,
        }, indent=2))
    elif args.command == "demo":
        _generate(config, "data/grid_telemetry.csv")
        _train(config, "data/grid_telemetry.csv", "artifacts/models.joblib", "reports/training_metrics.json")
        _simulate(config, "data/grid_telemetry.csv", "artifacts/models.joblib", "reports/decisions.csv")


if __name__ == "__main__":
    main()
