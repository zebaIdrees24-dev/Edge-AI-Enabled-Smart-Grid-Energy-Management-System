# Model Card: Adaptive Edge–Cloud Smart-Grid ML Prototype

## Intended use

Research/portfolio prototype for studying multivariate time-series forecasting, unsupervised anomaly detection, edge/cloud model selection, and energy-management decision orchestration. It is not certified for protection, dispatch, billing, or safety-critical grid control.

## Data

The repository ships a reproducible synthetic telemetry generator with 15-minute observations for load, renewable generation, weather, voltage, frequency, battery state of charge, communications latency/frequency, device-state transitions, and injected abnormal states. Real deployment requires governed utility/industrial data and domain-specific validation.

## Candidate models

Forecasting experiments cover a persistence baseline, Ridge, Random Forest, Extra Trees, HistGradientBoosting, and an optional PyTorch LSTM. Anomaly experiments cover robust statistical thresholding, Isolation Forest, One-Class SVM, and an optional autoencoder architecture.

## Evaluation design

- chronological 80/20 holdout;
- rolling-origin `TimeSeriesSplit` backtesting;
- MAE, RMSE, and R² for forecasting;
- precision, recall, and F1 for anomaly detection;
- feature-group ablation;
- feature importance;
- per-sample inference latency and serialized model-size measurements;
- deterministic seeds and captured package/runtime metadata.

## Limitations

Synthetic data cannot represent the full distribution shift, device failures, adversarial behavior, topology, market dynamics, weather uncertainty, or protection constraints found in real grids. Model metrics in this repository should therefore be interpreted as software/research-workflow validation only.

## Safety and governance

No model output should directly actuate real switchgear. A production system would require authentication/authorization, data governance, drift and calibration monitoring, human/operator controls, hardware/software-in-the-loop validation, regional grid-code checks, independent fail-safe protection, and change-management/audit processes.
