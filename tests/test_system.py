import numpy as np

from smartgrid_ems.anomaly import build_anomaly_detector
from smartgrid_ems.audit import append_security_event
from smartgrid_ems.controller import AdaptiveEMSController, run_controller
from smartgrid_ems.data import generate_synthetic_grid_data
from smartgrid_ems.features import FEATURES, engineer_features
from smartgrid_ems.models import train_models


def config():
    return {
        "seed": 7,
        "models": {"edge_trees": 12, "cloud_trees": 20, "cloud_max_depth": 8},
        "controller": {
            "uncertainty_threshold": 0.25,
            "anomaly_threshold": 0.55,
            "grid_frequency_min_hz": 49.5,
            "grid_frequency_max_hz": 50.5,
            "voltage_min_pu": 0.9,
            "voltage_max_pu": 1.1,
            "battery_low_soc": 0.15,
            "battery_high_soc": 0.9,
            "cloud_available": True,
        },
    }


def test_generation_and_features_are_reproducible():
    first = generate_synthetic_grid_data(rows=200, seed=3)
    second = generate_synthetic_grid_data(rows=200, seed=3)
    assert first.equals(second)
    featured = engineer_features(first)
    assert set(FEATURES).issubset(featured.columns)
    assert not featured[FEATURES].isna().any().any()


def test_training_and_controller_end_to_end():
    data = engineer_features(generate_synthetic_grid_data(rows=500, anomaly_rate=0.05, seed=7))
    bundle, metrics = train_models(data, config())
    assert all(np.isfinite(value) and value >= 0 for value in metrics.values())
    decisions = run_controller(data.tail(20), bundle, config())
    assert len(decisions) == 20
    assert decisions["forecast_kw"].gt(0).all()
    assert set(decisions["action"]).issubset(
        {"charge_battery", "discharge_battery", "preserve_battery", "hold", "island_or_protect"}
    )


def test_safety_violation_has_priority():
    data = engineer_features(generate_synthetic_grid_data(rows=350, seed=9))
    bundle, _ = train_models(data, config())
    unsafe = data.iloc[-1].copy()
    unsafe["frequency_hz"] = 48.5
    decision = AdaptiveEMSController(bundle, config()).decide(unsafe)
    assert decision.action == "island_or_protect"
    assert decision.escalated_to_cloud


def test_detector_options_and_audit_log(tmp_path):
    data = engineer_features(generate_synthetic_grid_data(rows=120, seed=11))
    for name in ("isolation_forest", "one_class_svm", "statistical"):
        detector = build_anomaly_detector(name, seed=11).fit(data[FEATURES])
        assert len(detector.decision_function(data[FEATURES].head(2))) == 2
    event = append_security_event("anomaly", {"score": 0.91}, tmp_path / "events.jsonl", "pi-01")
    assert event["device_ref"] != "pi-01"
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_research_benchmarks_and_ablations(tmp_path):
    from smartgrid_ems.research import run_research_suite

    data = engineer_features(generate_synthetic_grid_data(rows=500, anomaly_rate=0.06, seed=13))
    summary = run_research_suite(data, tmp_path, seed=13, include_lstm=False)
    assert len(summary["forecast_benchmarks"]) >= 5
    assert len(summary["ablation_study"]) == 4
    assert len(summary["anomaly_benchmarks"]) == 3
    assert (tmp_path / "forecast_benchmarks.csv").exists()
    assert (tmp_path / "ablation_study.csv").exists()
    assert (tmp_path / "anomaly_benchmarks.csv").exists()
    assert (tmp_path / "research_summary.json").exists()
    assert np.isfinite(summary["best_forecast_model"]["mae_kw"])


def test_lstm_forecaster_smoke():
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from smartgrid_ems.deep_models import train_lstm_forecaster

    data = engineer_features(generate_synthetic_grid_data(rows=220, anomaly_rate=0.04, seed=17))
    result = train_lstm_forecaster(
        data, seed=17, sequence_length=8, hidden_dim=8, epochs=1, batch_size=32
    )
    assert result["model"] == "lstm"
    assert np.isfinite(result["mae_kw"])
    assert result["inference_ms_per_sample"] >= 0
