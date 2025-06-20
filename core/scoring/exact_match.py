"""
Exact match scorer - checks if output exactly matches expected output.
"""

from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """
    Scorer that checks for exact string match between output and expected output.
    """

    @property
    def name(self) -> str:
        return "Exact Match"

    @property
    def description(self) -> str:
        return (
            "Checks if the output exactly matches the expected output (case-sensitive)"
        )

    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score an item based on exact string match.

        Args:
            item: The evaluation item to score

        Returns:
            ScorerResult with binary score (1.0 for match, 0.0 for no match)
        """
        if item.output is None:
            return ScorerResult(
                scorer_name="exact_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Check for exact match
        is_match = item.output.strip() == item.expected_output.strip()

        return ScorerResult(
            scorer_name="exact_match",
            score=1.0 if is_match else 0.0,
            passed=is_match,
            reasoning=(
                "Exact match found"
                if is_match
                else "Output does not exactly match expected"
            ),
            details={
                "output_length": len(item.output),
                "expected_length": len(item.expected_output),
                "stripped_match": item.output.strip() == item.expected_output.strip(),
            },
        )


class CaseInsensitiveExactMatchScorer(BaseScorer):
    """
    Scorer that checks for exact match ignoring case.
    """

    @property
    def name(self) -> str:
        return "Case-Insensitive Exact Match"

    @property
    def description(self) -> str:
        return "Checks if the output matches expected output ignoring case differences"

    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score an item based on case-insensitive exact match.

        Args:
            item: The evaluation item to score

        Returns:
            ScorerResult with binary score
        """
        if item.output is None:
            return ScorerResult(
                scorer_name="case_insensitive_exact_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Check for case-insensitive match
        is_match = item.output.strip().lower() == item.expected_output.strip().lower()

        return ScorerResult(
            scorer_name="case_insensitive_exact_match",
            score=1.0 if is_match else 0.0,
            passed=is_match,
            reasoning=(
                "Case-insensitive match found"
                if is_match
                else "Output does not match (case-insensitive)"
            ),
            details={
                "case_sensitive_match": item.output.strip()
                == item.expected_output.strip(),
                "case_insensitive_match": is_match,
            },
        )


class NormalizedExactMatchScorer(BaseScorer):
    """
    Scorer that normalizes text before checking for exact match.
    Normalization includes: lowercasing, removing extra whitespace, punctuation normalization.
    """

    @property
    def name(self) -> str:
        return "Normalized Exact Match"

    @property
    def description(self) -> str:
        return "Checks for match after normalizing whitespace, case, and punctuation"

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        import re

        # Convert to lowercase
        text = text.lower()

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Normalize common punctuation
        # Replace smart quotes/apostrophes with straight ones
        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2019", "'")
            .replace("\u2018", "'")
        )

        # Remove trailing punctuation if configured
        if self.config.get("ignore_trailing_punctuation", False):
            text = re.sub(r"[.!?;,]+$", "", text)

        return text

    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score an item based on normalized exact match.

        Args:
            item: The evaluation item to score

        Returns:
            ScorerResult with binary score
        """
        if item.output is None:
            return ScorerResult(
                scorer_name="normalized_exact_match",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )

        # Normalize both texts
        normalized_output = self.normalize_text(item.output)
        normalized_expected = self.normalize_text(item.expected_output)

        is_match = normalized_output == normalized_expected

        return ScorerResult(
            scorer_name="normalized_exact_match",
            score=1.0 if is_match else 0.0,
            passed=is_match,
            reasoning=(
                "Normalized match found" if is_match else "No match after normalization"
            ),
            details={
                "raw_match": item.output == item.expected_output,
                "normalized_output": normalized_output,
                "normalized_expected": normalized_expected,
            },
        )
