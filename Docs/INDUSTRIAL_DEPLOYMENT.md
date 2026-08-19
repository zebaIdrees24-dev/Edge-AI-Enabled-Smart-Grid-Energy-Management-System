# Industrial deployment profile

This profile maps the prototype to a ConnectX-style industrial energy stack while keeping all physical-device operations read-only and cloud integrations opt-in.

## Data path

1. Field devices expose operational measurements through Modbus TCP/RTU or CAN.
2. A Raspberry Pi 4B gateway performs normalization, feature engineering, anomaly scoring, and the lightweight forecast.
3. Events are published with mutual-TLS device identity to AWS IoT Core over MQTT QoS 1. REST is available for managed service integration.
4. An IoT topic rule invokes Lambda; Lambda validates the event envelope and archives it in an encrypted, versioned S3 bucket.
5. EC2 or another cloud compute service can run the high-capacity forecast, retraining, fleet analytics, and investigation workflows.

The repository includes protocol boundaries in `smartgrid_ems.adapters` and an AWS SAM template in `infra/`. The default `mock` mode requires no credentials and does not contact devices or AWS.

## Feature families

- Physical: voltage, frequency, battery SoC, load, renewable generation, and power deviation
- Temporal: load rate-of-change, lagged demand, and rolling statistics
- Communications: latency, message frequency, missing/late data (extension point)
- Device behavior: state-transition indicator and future command/state consistency checks

## Detector choices

Set `models.anomaly_detector` to `isolation_forest`, `one_class_svm`, or `statistical`. The optional `deep_models.build_torch_autoencoder` provides a compact PyTorch architecture for reconstruction-error experiments. Train, calibrate, export, and benchmark it separately before selecting it on a Raspberry Pi.

## Raspberry Pi guidance

- Use Raspberry Pi OS 64-bit, a dedicated non-root service account, read-only root filesystem where practical, and systemd restart limits.
- Export compact models, pin dependency versions, record model/file hashes, and benchmark latency, CPU, memory, and temperature.
- Queue signed events locally during cloud loss; never make basic protection depend on WAN connectivity.
- Store private keys in a hardware-backed keystore where available; rotate device certificates and reject expired credentials.

## Security and SOC 2-aligned evidence

The JSONL audit helper emits UTC timestamps, unique event IDs, hashed device references, event type, and structured payload. This supports evidence collection for logical access, change traceability, monitoring, and incident response, but **does not make the system SOC 2 compliant**. A real program also needs documented controls, owners, approvals, retention policies, access reviews, alert handling, vendor management, backups, tests, and independent audit evidence.

Recommended controls include least-privilege IAM, per-device X.509 certificates, TLS 1.2+, encrypted S3, CloudTrail/CloudWatch, immutable or lock-governed retention, secrets rotation, signed model releases, SBOM and dependency scanning, and alarm-to-ticket integration.

