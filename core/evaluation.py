"""Evaluation orchestration."""
from typing import Iterable, List
from .data_models import (
    EvalRecord,
    Score,
    RunMetadata,
    RunResult,
    EvaluationItem,
)
from .scoring import create_scorer


def run_evaluation(records: Iterable[EvalRecord]) -> RunResult:
    scores: List[Score] = []
    scorer = create_scorer("exact_match")
    for idx, record in enumerate(records):
        item = EvaluationItem(
            id=idx,
            input=record.prompt,
            expected_output=record.expected,
            output=record.output or "",
        )
        result = scorer.score(item)
        scores.append(Score(value=result.score, meta={"passed": result.passed}))
    metadata = RunMetadata(scorer="exact_match")
    return RunResult(records=list(records), scores=scores, metadata=metadata)
