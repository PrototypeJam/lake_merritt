"""
Telemetry utilities for observability and monitoring.
Currently provides stubs for future OpenTelemetry integration.
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


_tracer = None


def init_telemetry(
    service_name: str = "ai-eval-workbench",
    enable: bool = None,
    endpoint: Optional[str] = None,
) -> None:
    """Initialize telemetry/tracing."""
    import os

    if enable is None:
        enable = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"

    if not enable:
        logger.info("Telemetry disabled")
        return

    try:
        logger.info(f"Telemetry initialized for service: {service_name}")
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")


@contextmanager
def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a trace span (context manager)."""
    start_time = time.time()

    try:
        logger.debug(f"Starting span: {name}")
        if attributes:
            logger.debug(f"Span attributes: {attributes}")
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Completed span: {name} (duration: {duration:.3f}s)")


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace function execution."""

    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            with create_span(span_name, attributes):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with create_span(span_name, attributes):
                return await func(*args, **kwargs)

        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def log_metric(
    name: str,
    value: float,
    unit: str = "",
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Log a metric value."""
    metric_info = {
        "name": name,
        "value": value,
        "unit": unit,
    }
    if tags:
        metric_info["tags"] = tags

    logger.info(f"Metric: {metric_info}")


def trace_llm_call(
    provider: str,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    duration: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Trace an LLM API call."""
    span_attrs = {
        "llm.provider": provider,
        "llm.model": model,
    }

    if input_tokens is not None:
        span_attrs["llm.input_tokens"] = input_tokens
    if output_tokens is not None:
        span_attrs["llm.output_tokens"] = output_tokens
    if duration is not None:
        span_attrs["llm.duration_ms"] = int(duration * 1000)
    if error:
        span_attrs["llm.error"] = error

    logger.info(f"LLM call trace: {span_attrs}")

    if duration is not None:
        log_metric(
            "llm.call.duration",
            duration,
            "seconds",
            {"provider": provider, "model": model},
        )

    if input_tokens is not None:
        log_metric(
            "llm.tokens.input",
            input_tokens,
            "tokens",
            {"provider": provider, "model": model},
        )

    if output_tokens is not None:
        log_metric(
            "llm.tokens.output",
            output_tokens,
            "tokens",
            {"provider": provider, "model": model},
        )


class TelemetryContext:
    """Context for collecting telemetry data during an operation."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.attributes = {}
        self.metrics = {}
        self.events = []

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Started telemetry context: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type:
            self.add_event("error", {"type": str(exc_type), "message": str(exc_val)})

        logger.info(
            f"Completed telemetry context: {self.operation_name}",
            extra={
                "duration": duration,
                "attributes": self.attributes,
                "metrics": self.metrics,
                "events": self.events,
            },
        )

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_metric(self, name: str, value: float, unit: str = "") -> None:
        self.metrics[name] = {"value": value, "unit": unit}

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        event = {"name": name, "timestamp": time.time()}
        if attributes:
            event["attributes"] = attributes
        self.events.append(event)
