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

    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score an item using fuzzy string matching.

        Args:
            item: The evaluation item to score

        Returns:
            ScorerResult with similarity score (0.0 to 1.0)
        """
        if item.output is None:
            return ScorerResult(
                scorer_name="fuzzy_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Get the appropriate fuzzy matching function
        if self.algorithm == "ratio":
            similarity = fuzz.ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif self.algorithm == "partial_ratio":
            similarity = fuzz.partial_ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif self.algorithm == "token_sort_ratio":
            similarity = fuzz.token_sort_ratio(
                item.output, item.expected_output, processor=str.lower
            )
        elif self.algorithm == "token_set_ratio":
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
        passed = score >= self.threshold

        # Generate reasoning
        if passed:
            reasoning = f"Similarity score {score:.2f} meets threshold {self.threshold}"
        else:
            reasoning = f"Similarity score {score:.2f} below threshold {self.threshold}"

        return ScorerResult(
            scorer_name="fuzzy_match",
            score=score,
            passed=passed,
            reasoning=reasoning,
            details={
                "algorithm": self.algorithm,
                "threshold": self.threshold,
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
