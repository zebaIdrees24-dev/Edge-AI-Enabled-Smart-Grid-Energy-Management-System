from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class StatisticalThresholdDetector(BaseEstimator):
    """Robust multivariate thresholding using median absolute deviation."""

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def fit(self, x, y=None):
        values = np.asarray(x)
        self.median_ = np.median(values, axis=0)
        self.scale_ = np.median(np.abs(values - self.median_), axis=0) * 1.4826 + 1e-9
        return self

    def decision_function(self, x):
        z = np.abs((np.asarray(x) - self.median_) / self.scale_)
        return self.threshold - np.max(z, axis=1)

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0, 1, -1)


def build_anomaly_detector(kind: str, seed: int = 42):
    """Build Isolation Forest, One-Class SVM, or statistical detector."""
    if kind == "isolation_forest":
        model = IsolationForest(contamination="auto", random_state=seed)
    elif kind == "one_class_svm":
        model = OneClassSVM(kernel="rbf", nu=0.04, gamma="scale")
    elif kind == "statistical":
        model = StatisticalThresholdDetector()
    else:
        raise ValueError(f"Unknown anomaly detector: {kind}")
    return Pipeline([("scale", StandardScaler()), ("model", model)])

