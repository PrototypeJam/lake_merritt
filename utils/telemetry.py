"""Telemetry placeholders."""
from contextlib import contextmanager


@contextmanager
def trace_llm_call(name: str):
    yield
