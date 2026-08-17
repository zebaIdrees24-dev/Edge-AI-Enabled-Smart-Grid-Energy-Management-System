This repository presents the implemetation of GridEdge-ML: Adaptive Edge–Cloud AI for Smart Grid Energy Management

GridEdge-ML is an end-to-end machine learning, Edge AI, and MLOps platform for intelligent smart-grid energy management. It analyzes multivariate grid telemetry to support load forecasting, anomaly detection, operational monitoring, and adaptive edge–cloud inference while balancing prediction accuracy, latency, model size, and computational cost.

The project benchmarks multiple ML and deep-learning approaches, including Ridge Regression, Random Forest, Extra Trees, HistGradientBoosting, PyTorch LSTM, Isolation Forest, One-Class SVM, statistical thresholding, and autoencoder-based anomaly detection. It also includes time-series backtesting, feature-ablation studies, feature-importance analysis, inference-latency benchmarking, and reproducible model evaluation.

The architecture integrates Raspberry Pi edge gateways with MQTT, REST APIs, Modbus TCP/RTU, and CAN bus for real-time industrial telemetry. Lightweight models can execute locally at the edge, while uncertain, anomalous, or computationally intensive cases are escalated to cloud services.

Cloud workflows use AWS IoT Core, AWS Lambda, Amazon S3, and EC2 for device connectivity, event-driven processing, data storage, and scalable ML analytics.

The repository also demonstrates practical MLOps and production ML practices through modular Python pipelines, Docker containerization, pytest-based testing, GitHub Actions CI, model monitoring, event logging, reproducible experiments, and deployment-oriented workflows.

Technologies: Python, pandas, NumPy, scikit-learn, PyTorch, Jupyter, Raspberry Pi, AWS IoT Core, Lambda, S3, EC2, MQTT, Modbus TCP/RTU, CAN bus, REST APIs, Docker, pytest, GitHub Actions, Git, YAML, joblib.
