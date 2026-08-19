"""Safe industrial protocol boundaries with a mock-first default.

Production protocol clients are optional dependencies. No adapter writes to a
physical device; control remains a separate, authenticated integration concern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class TelemetrySource(Protocol):
    def read(self) -> dict[str, Any]: ...


class EventSink(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...


@dataclass
class MockTelemetrySource:
    sample: dict[str, Any]

    def read(self) -> dict[str, Any]:
        return dict(self.sample)


class MQTTSink:
    """Publish JSON to a TLS-configured MQTT client supplied by the caller."""

    def __init__(self, client: Any, topic: str, qos: int = 1):
        self.client, self.topic, self.qos = client, topic, qos

    def publish(self, event: dict[str, Any]) -> None:
        result = self.client.publish(self.topic, json.dumps(event), qos=self.qos)
        if getattr(result, "rc", 0) != 0:
            raise RuntimeError(f"MQTT publish failed with code {result.rc}")


class RESTSink:
    """Send an event through an authenticated requests-compatible session."""

    def __init__(self, session: Any, endpoint: str, timeout_seconds: float = 5):
        self.session, self.endpoint, self.timeout = session, endpoint, timeout_seconds

    def publish(self, event: dict[str, Any]) -> None:
        response = self.session.post(self.endpoint, json=event, timeout=self.timeout)
        response.raise_for_status()


class ModbusReader:
    """Read-only wrapper around a configured pymodbus-compatible client."""

    def __init__(self, client: Any, address: int, count: int, device_id: int = 1):
        self.client, self.address, self.count, self.device_id = client, address, count, device_id

    def read(self) -> dict[str, Any]:
        result = self.client.read_holding_registers(
            address=self.address, count=self.count, device_id=self.device_id
        )
        if result.isError():
            raise RuntimeError(f"Modbus read failed: {result}")
        return {"registers": list(result.registers)}


class CANReader:
    """Read one CAN frame from a python-can compatible bus."""

    def __init__(self, bus: Any, timeout_seconds: float = 1):
        self.bus, self.timeout = bus, timeout_seconds

    def read(self) -> dict[str, Any]:
        message = self.bus.recv(timeout=self.timeout)
        if message is None:
            raise TimeoutError("No CAN frame received")
        return {
            "arbitration_id": message.arbitration_id,
            "data_hex": bytes(message.data).hex(),
            "timestamp": message.timestamp,
        }

