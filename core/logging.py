"""Structured logging with request-ID correlation.

Provides ``configure_logging`` (called by each entrypoint) and a ``request_id``
context variable that a middleware sets per request. When ``LOG_FORMAT=json``
every log line is emitted as a single JSON object including the active
request_id, so logs from one request can be traced across the whole pipeline
(rewrite → retrieve → rerank → generate) in a log aggregator.
"""

import contextvars
import json
import logging

# Set per request by api.middleware; empty outside a request (e.g. CLI, tests).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    """Attach the active request_id to every record (empty if none)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] — %(message)s"


def configure_logging(log_format: str = "text", level: int = logging.INFO) -> None:
    """Configure the root logger's handler, format, and request-id filter.

    Idempotent: replaces existing handlers so repeated calls (e.g. across
    entrypoints or test reloads) don't stack duplicate output.

    Args:
        log_format: "json" for structured logs, anything else for human-readable
            text. Both include the request_id.
        level: Root log level.
    """
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
