"""Safe local structured execution logging for Runtime v0."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

_LOGGER = logging.getLogger("resilient_ai_runtime.execution")


def emit_execution_record(record: Mapping[str, Any]) -> None:
    """Emit a machine-readable execution record without interrupting callers."""

    try:
        _LOGGER.info(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))
    except Exception:
        # Observability must not turn a completed generation into a failure.
        return
