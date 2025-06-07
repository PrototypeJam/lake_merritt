"""Evaluation orchestration."""
from typing import Iterable, List
from .data_models import EvalRecord, Score, RunMetadata, RunResult
from .scoring import exact_match


def run_evaluation(records: Iterable[EvalRecord]) -> RunResult:
    scores: List[Score] = []
    for record in records:
        scores.append(exact_match.score(record, {}))
    metadata = RunMetadata(scorer="exact_match")
    return RunResult(records=list(records), scores=scores, metadata=metadata)
