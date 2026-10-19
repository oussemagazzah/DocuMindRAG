"""Structured logging configuration for DocuMindRAG."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        request_id = REQUEST_ID.get()
        prefix = f"{color}{timestamp} [{record.levelname}]{self.RESET}"
        req_part = f" [{request_id[:8]}]" if request_id != "-" else ""
        return f"{prefix}{req_part} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, use JSON format (for production).
                     If False, use colored readable format (for development).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root.level)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ReadableFormatter())

    root.addHandler(handler)

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(f"documind.{name}")
