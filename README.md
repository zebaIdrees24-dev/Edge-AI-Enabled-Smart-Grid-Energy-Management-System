Smart Grid / Energy Management System — Edge AI & AWS Cloud

This work develops a secure edge-to-cloud smart energy management architecture for industrial and smart-grid environments. Field devices, smart meters, sensors, and energy assets are connected to Raspberry Pi edge gateways, where lightweight ML models perform real-time energy monitoring, anomaly detection, and local decision-making. Computationally intensive analytics, historical processing, forecasting, and system-wide monitoring are handled through AWS cloud services, enabling scalable and low-latency energy management.

Architecture: Designed an edge–cloud architecture integrating smart meters, industrial field devices, Raspberry Pi gateways, Edge AI inference, and AWS-based energy-data processing.
Tools & Platforms: Python, Jupyter Notebook, Raspberry Pi 4B, AWS IoT Core, AWS Lambda, Amazon S3, Amazon EC2, pandas, NumPy, scikit-learn, PyTorch, and TensorFlow.
Industrial Connectivity: Integrated real-time energy and equipment data using MQTT, REST APIs, Modbus TCP/RTU, and CAN bus, enabling communication between field devices, edge gateways, and cloud applications.
Energy Analytics & Feature Engineering: Developed features including power deviation, rate-of-change, rolling statistics, energy-consumption patterns, device-state transitions, communication latency, and message frequency for operational monitoring and intelligent energy analysis.
AI/ML Techniques: Applied load/energy-consumption prediction, anomaly detection, Isolation Forest, One-Class SVM, statistical thresholding, and autoencoder-based approaches to identify abnormal energy usage and equipment operating conditions.
Edge AI: Deployed lightweight PyTorch/TensorFlow inference models on Raspberry Pi for near-real-time analysis, reducing cloud dependency, communication latency, and unnecessary data transmission.
Cloud Processing: Used AWS IoT Core for secure device connectivity, Lambda for event-driven processing, S3 for historical energy-data storage, and EC2 for cloud-hosted analytics and ML services.
Cybersecurity: Implemented authenticated bidirectional edge–cloud communication, anomaly/threat monitoring, configurable alerting, event logging, device traceability, and security controls for connected energy assets.
Output: Enabled real-time energy monitoring, abnormal-consumption detection, load and demand analysis, edge-based intelligent decision-making, secure device management, and cloud-based energy analytics for dynamic industrial and smart-grid environments.
