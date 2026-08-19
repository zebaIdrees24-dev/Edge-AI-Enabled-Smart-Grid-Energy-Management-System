from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def append_security_event(
    event_type: str,
    payload: dict[str, Any],
    path: str | Path,
    device_id: str = "unknown",
    hash_identifier: bool = True,
) -> dict[str, Any]:
    """Append a traceable JSON event without storing raw device identity by default."""
    identity = hashlib.sha256(device_id.encode()).hexdigest() if hash_identifier else device_id
    event = {
        "event_id": str(uuid4()),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "device_ref": identity,
        "payload": payload,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event

