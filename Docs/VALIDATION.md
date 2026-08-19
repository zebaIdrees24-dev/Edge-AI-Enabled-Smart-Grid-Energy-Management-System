# Reproducibility and validation

The default synthetic workflow and the expanded ML research suite were executed on the packaged 2,880-sample synthetic telemetry dataset (2,876 samples after causal lag construction). These results validate software behavior and experimental reproducibility only; they are **not** evidence of real-grid performance.

## Runtime edge/cloud models

| Measure | Result |
|---|---:|
| Edge Random Forest MAE | 13.80 kW |
| Edge Random Forest RMSE | 26.28 kW |
| Cloud Extra Trees MAE | 13.69 kW |
| Cloud Extra Trees RMSE | 25.88 kW |
| Cloud escalation rate | 4.90% |

## Forecast model comparison

| Model | MAE (kW) | RMSE (kW) | R² | Approx. serialized size |
|---|---:|---:|---:|---:|
| Persistence (lag-1) | 21.40 | 42.24 | 0.411 | — |
| Ridge | 17.08 | 29.83 | 0.706 | 1.8 KB |
| Random Forest | 13.78 | 26.29 | 0.772 | 1.5 MB |
| Extra Trees | 13.68 | 25.78 | 0.781 | 24.0 MB |
| HistGradientBoosting | **13.28** | **22.50** | **0.833** | 0.72 MB |
| LSTM (8 epochs) | 14.73 | 30.36 | 0.696 | **30.7 KB** |

The holdout results show why a research-style comparison is useful: the larger Extra Trees model is not automatically the most accurate, and the compact LSTM exposes a distinct model-footprint trade-off.

## Rolling-origin time-series backtest

| Model | Mean MAE ± SD (kW) | Mean RMSE (kW) | Mean R² |
|---|---:|---:|---:|
| Ridge | 16.05 ± 0.62 | 26.79 | 0.736 |
| Random Forest | **13.68 ± 0.15** | 24.69 | 0.776 |
| HistGradientBoosting | 13.90 ± 0.36 | **22.61** | **0.811** |

Random Forest produced the lowest mean MAE across the four time folds, while HistGradientBoosting produced the strongest RMSE and R². This is a useful example of metric-dependent model selection rather than declaring one universal winner.

## Feature ablation

Removing communications telemetry increased Extra Trees holdout MAE from 13.68 to 14.51 kW and reduced R² from 0.781 to 0.706. Removing temporal features also degraded performance. Feature-importance analysis ranked `load_rolling_4`, `load_lag_1`, and cyclic hour features among the strongest predictors.

## Anomaly-detector comparison

On the held-out synthetic anomaly labels, One-Class SVM produced the strongest F1 among the three tested unsupervised methods (0.48), followed by Isolation Forest (0.34) and robust statistical thresholding (0.20). The synthetic anomaly generator is deliberately simple, so these scores should be used to demonstrate evaluation methodology rather than real-world detector quality.

## Software checks

- 6 automated tests passed, including reproducible generation, end-to-end training/control, safety precedence, detector factories, audit logging, research outputs, and a PyTorch LSTM smoke test.
- Python compilation checks passed for the package modules.
- GitHub Actions remains configured to run `ruff` and `pytest` on Python 3.10 and 3.12. (`ruff` was not installed in the local execution runtime used to create this validation snapshot.)

See [`RESEARCH_EXPERIMENTS.md`](RESEARCH_EXPERIMENTS.md) for experiment design and commands.
