"""JSON application logs with stable run and step context."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .config import RuntimeConfig

_STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Keep structured extras without losing exceptions or timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize standard log fields and experiment context as one JSON object."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _STANDARD_FIELDS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, allow_nan=False)


class RunContext(logging.Filter):
    """Attach a default run ID while preserving explicit per-record context."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Supply the configured run identity when the record has no override."""
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


def configure_logging(level: str | None = None, run_id: str | None = None) -> None:
    """Install one JSON stderr handler. Repeated setup does not duplicate output."""
    cfg = RuntimeConfig.from_env()
    root = logging.getLogger()
    handler = next((h for h in root.handlers if h.name == "geotrace-json"), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.name = "geotrace-json"
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter())
    handler.filters.clear()
    handler.addFilter(RunContext(run_id or cfg.run_id))
    root.setLevel(level or cfg.log_level)
