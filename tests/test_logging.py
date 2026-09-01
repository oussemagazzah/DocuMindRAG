"""Tests for logging configuration."""

import logging
import json

from app.logging_config import setup_logging, get_logger, JSONFormatter, ReadableFormatter, REQUEST_ID


def test_setup_logging_creates_handler():
    setup_logging(level="INFO", json_output=False)
    root = logging.getLogger()
    assert len(root.handlers) > 0


def test_get_logger_returns_named_logger():
    logger = get_logger("test_module")
    assert logger.name == "documind.test_module"
    assert isinstance(logger, logging.Logger)


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "test message"
    assert "timestamp" in parsed


def test_readable_formatter():
    formatter = ReadableFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "hello" in output
    assert "INFO" in output


def test_request_id_context_var():
    REQUEST_ID.set("test-123")
    assert REQUEST_ID.get() == "test-123"
    REQUEST_ID.set("-")


def test_json_formatter_with_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=sys.exc_info(),
        )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]
