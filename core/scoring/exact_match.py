"""Exact match scoring utilities."""
from __future__ import annotations

from typing import Any, Dict, Optional
import re

from core.data_models import EvaluationItem, ScorerResult


class ExactMatchScorer:
    """Scorer that checks for strict string equality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def score(self, item: EvaluationItem) -> ScorerResult:
        if not item.output:
            return ScorerResult(
                scorer_name="exact_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
                details={
                    "output_length": 0,
                    "expected_length": len(item.expected_output.strip()),
                    "stripped_match": False,
                },
            )

        output_stripped = item.output.strip()
        expected_stripped = item.expected_output.strip()
        match = output_stripped == expected_stripped
        reasoning = "Exact match found" if match else "Output does not exactly match expected output"
        return ScorerResult(
            scorer_name="exact_match",
            score=1.0 if match else 0.0,
            passed=match,
            reasoning=reasoning,
            details={
                "output_length": len(output_stripped),
                "expected_length": len(expected_stripped),
                "stripped_match": match,
            },
        )


class CaseInsensitiveExactMatchScorer(ExactMatchScorer):
    """Exact match scorer that ignores case."""

    def score(self, item: EvaluationItem) -> ScorerResult:
        base = super().score(item)
        if base.score == 1.0:
            base.details.update(
                {
                    "case_sensitive_match": True,
                    "case_insensitive_match": True,
                }
            )
            return base

        if item.output is None:
            base.details.update(
                {
                    "case_sensitive_match": False,
                    "case_insensitive_match": False,
                }
            )
            return base

        case_insensitive_match = item.output.strip().lower() == item.expected_output.strip().lower()
        passed = case_insensitive_match
        reasoning = (
            "Case-insensitive exact match found"
            if passed
            else "Output does not match expected output (case-insensitive)"
        )
        return ScorerResult(
            scorer_name="case_insensitive_exact_match",
            score=1.0 if passed else 0.0,
            passed=passed,
            reasoning=reasoning,
            details={
                "case_sensitive_match": item.output.strip() == item.expected_output.strip(),
                "case_insensitive_match": case_insensitive_match,
            },
        )


class NormalizedExactMatchScorer(ExactMatchScorer):
    """Exact match scorer with text normalization."""

    def _normalize(self, text: str) -> str:
        text = text.strip()
        # Replace smart quotes/apostrophes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        if self.config.get("ignore_trailing_punctuation"):
            text = re.sub(r"[.!?]+$", "", text)
        return text.lower()

    def score(self, item: EvaluationItem) -> ScorerResult:
        if not item.output:
            return ScorerResult(
                scorer_name="normalized_exact_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        normalized_output = self._normalize(item.output)
        normalized_expected = self._normalize(item.expected_output)
        passed = normalized_output == normalized_expected
        reasoning = (
            "Normalized match found" if passed else "Normalized output does not match expected"
        )
        return ScorerResult(
            scorer_name="normalized_exact_match",
            score=1.0 if passed else 0.0,
            passed=passed,
            reasoning=reasoning,
            details={
                "normalized_output": normalized_output,
                "normalized_expected": normalized_expected,
                "raw_match": item.output.strip() == item.expected_output.strip(),
            },
        )
