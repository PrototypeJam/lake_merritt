"""Reporting helpers."""
from .data_models import RunResult


def to_dict(result: RunResult) -> dict:
    return result.dict()
