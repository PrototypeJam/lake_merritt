"""Fuzzy match scorer."""
from difflib import SequenceMatcher
from core.data_models import EvalRecord, Score


def score(record: EvalRecord, cfg: dict) -> Score:
    ratio = SequenceMatcher(None, record.expected, record.output or "").ratio()
    return Score(value=ratio)
