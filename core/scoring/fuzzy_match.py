# core/scoring/fuzzy_match.py
"""
Fuzzy match scorer - uses string similarity algorithms to score matches.
"""

from typing import Any, Dict
from rapidfuzz import fuzz
from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer

class FuzzyMatchScorer(BaseScorer):
    """
    Scorer that uses fuzzy string matching to calculate similarity.
    """
    
    requires_api_key = False  # No API key needed for fuzzy matching

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.8)
        self.algorithm = self.config.get("algorithm", "token_sort_ratio")

    @property
    def name(self) -> str:
        return "Fuzzy Match"

    @property
    def description(self) -> str:
        return f"Uses fuzzy string matching with {self.algorithm} algorithm (threshold: {self.threshold})"

    def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        """
        Score an item using fuzzy string matching.

        Args:
            item: The evaluation item to score
            stage_config: Stage-specific configuration (threshold, algorithm)

        Returns:
            ScorerResult with similarity score (0.0 to 1.0)
        """
        threshold = stage_config.get("threshold", self.threshold)
        algorithm = stage_config.get("algorithm", self.algorithm)

        if item.output is None:
            return ScorerResult(
                scorer_name="fuzzy_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Get the appropriate fuzzy matching function
        if algorithm == "ratio":
            similarity = fuzz.ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif algorithm == "partial_ratio":
            similarity = fuzz.partial_ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif algorithm == "token_sort_ratio":
            similarity = fuzz.token_sort_ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif algorithm == "token_set_ratio":
            similarity = fuzz.token_set_ratio(
                item.output, item.expected_output, processor=str.lower
            )
        else:
            # Default to token_sort_ratio
            similarity = fuzz.token_sort_ratio(
                item.output, item.expected_output, processor=str.lower
            )

        # Convert to 0-1 scale
        score = similarity / 100.0
        passed = score >= threshold

        # Generate reasoning
        if passed:
            reasoning = f"Similarity score {score:.2f} meets threshold {threshold}"
        else:
            reasoning = f"Similarity score {score:.2f} below threshold {threshold}"

        return ScorerResult(
            scorer_name="fuzzy_match",
            score=score,
            passed=passed,
            reasoning=reasoning,
            details={
                "algorithm": algorithm,
                "threshold": threshold,
                "raw_similarity": similarity,
                "all_scores": {
                    "ratio": fuzz.ratio(
                        item.output, item.expected_output, processor=str.lower
                    ),
                    "partial_ratio": fuzz.partial_ratio(
                        item.output, item.expected_output, processor=str.lower
                    ),
                    "token_sort_ratio": fuzz.token_sort_ratio(
                        item.output, item.expected_output, processor=str.lower
                    ),
                    "token_set_ratio": fuzz.token_set_ratio(
                        item.output, item.expected_output, processor=str.lower
                    ),
                },
            },
        )
