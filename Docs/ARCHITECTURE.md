# Architecture and design

## Edge tier

The edge tier receives recent telemetry, creates only causal lag/rolling features, estimates near-term load with a bounded-depth random forest, and computes an anomaly score. It remains capable of local decisions when cloud connectivity is unavailable.

## Adaptive collaboration

A sample is sent to the cloud when at least one condition holds:

1. dispersion across edge decision trees exceeds the uncertainty threshold;
2. the Isolation Forest score exceeds the anomaly threshold; or
3. voltage/frequency is outside configured limits.

Cloud availability is an explicit configuration input. When offline, the edge prediction and hard safety policy remain active.

## Cloud tier

The cloud model uses a larger Extra Trees ensemble. In a production deployment this tier could also perform fleet-wide optimization, longer-horizon forecasting, model retraining, and drift analysis. Those network services are intentionally outside this safe local prototype.

## EMS policy

The controller computes forecast net demand (`forecast load - renewable generation`) and emits one recommendation. Safety-limit violations take precedence, followed by low battery protection, high-demand discharge, renewable-surplus charging, and hold.

## Design limitations

- Synthetic telemetry is illustrative and not a power-flow simulation.
- The forecaster predicts the current target using lagged observations as a compact demonstration. Production systems should explicitly align a future horizon target.
- Isolation Forest is unsupervised; validate its threshold for each site.
- Policy outputs are recommendations, not actuator commands.

