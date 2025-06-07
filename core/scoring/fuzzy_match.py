"""Fuzzy string matching scorer."""
from __future__ import annotations

from typing import Any, Dict, Optional
from difflib import SequenceMatcher

from core.data_models import EvaluationItem, ScorerResult


class FuzzyMatchScorer:
    """Compute similarity ratio between output and expected output."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.threshold = cfg.get("threshold", 0.8)

    def score(self, item: EvaluationItem) -> ScorerResult:
        if not item.output:
            ratio = 0.0
        else:
            ratio = SequenceMatcher(None, item.expected_output, item.output).ratio()
        passed = ratio >= self.threshold
        reasoning = "Similarity above threshold" if passed else "Similarity below threshold"
        return ScorerResult(
            scorer_name="fuzzy_match",
            score=round(ratio, 4),
            passed=passed,
            reasoning=reasoning,
            details={"threshold": self.threshold},
        )
