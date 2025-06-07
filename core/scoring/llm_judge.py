"""LLM-based scorer placeholder."""
from core.data_models import EvalRecord, Score


def score(record: EvalRecord, cfg: dict) -> Score:
    # TODO: integrate with real LLM
    return Score(value=0.0)
