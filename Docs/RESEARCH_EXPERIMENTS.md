# ML Research Experiments

This repository includes a research-oriented evaluation path in addition to the runtime EMS demo. The goal is to demonstrate the workflow expected of an applied ML / ML research engineer: define baselines, preserve temporal causality, compare model families, measure trade-offs, and test whether feature groups actually contribute.

## 1. Forecasting baselines

`smartgrid-ems research` compares:

- persistence (`load_lag_1`) as a non-learned sanity baseline;
- Ridge regression as a linear baseline;
- Random Forest as the lightweight edge-family model;
- Extra Trees as the higher-capacity tree ensemble;
- HistGradientBoosting as a boosting baseline;
- optional PyTorch LSTM for multivariate sequence forecasting.

Metrics include MAE, RMSE, R², median inference latency per sample, and serialized model size. The latter two make the accuracy/efficiency trade-off visible rather than reporting accuracy alone.

## 2. Ablation study

The Extra Trees model is retrained with four feature configurations:

- full feature set;
- no temporal features;
- no grid-state features;
- no communication features.

This quantifies whether temporal context, operating-state measurements, and communications telemetry materially improve the load-forecasting task.

## 3. Anomaly-detector comparison

Three unsupervised methods are fitted only on normal training examples and evaluated against held-out synthetic anomaly labels:

- robust statistical thresholding;
- Isolation Forest;
- One-Class SVM.

Precision, recall, F1, and predicted anomaly rate are reported.

## 4. Deep time-series experiment

With the `deep-learning` extra installed, use:

```bash
smartgrid-ems research --include-lstm --lstm-epochs 12
```

The LSTM uses a causal rolling sequence, training-only standardization, and a chronological holdout. It is intentionally compact so the experiment can be repeated on CPU and can serve as a starting point for edge/cloud model-compression work.

## 5. Reproducibility

All stochastic models use the configured seed. The forecasting split is chronological, scalers are fitted on the training period only, and experiment outputs are written as CSV/JSON under `reports/research/`.

Generated outputs:

- `forecast_benchmarks.csv`
- `ablation_study.csv`
- `anomaly_benchmarks.csv`
- `research_summary.json`

The included synthetic dataset is for software validation only; performance values should not be presented as evidence of real-grid accuracy.
