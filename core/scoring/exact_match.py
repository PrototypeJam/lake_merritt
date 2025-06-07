"""Exact match scorer."""
from core.data_models import EvalRecord, Score


def score(record: EvalRecord, cfg: dict) -> Score:
    value = 1.0 if record.output == record.expected else 0.0
    return Score(value=value)
