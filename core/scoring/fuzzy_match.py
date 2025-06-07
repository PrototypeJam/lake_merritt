"""
Fuzzy match scorer - uses string similarity algorithms to score matches.
"""
from fuzzywuzzy import fuzz
from typing import Dict, Any

from core.scoring.base import BaseScorer
from core.data_models import EvaluationItem, ScorerResult


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
            similarity = fuzz.ratio(item.output, item.expected_output)
        elif self.algorithm == "partial_ratio":
            similarity = fuzz.partial_ratio(item.output, item.expected_output)
        elif self.algorithm == "token_sort_ratio":
            similarity = fuzz.token_sort_ratio(item.output, item.expected_output)
        elif self.algorithm == "token_set_ratio":
            similarity = fuzz.token_set_ratio(item.output, item.expected_output)
        else:
            # Default to token_sort_ratio
            similarity = fuzz.token_sort_ratio(item.output, item.expected_output)
        
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
                    "ratio": fuzz.ratio(item.output, item.expected_output),
                    "partial_ratio": fuzz.partial_ratio(item.output, item.expected_output),
                    "token_sort_ratio": fuzz.token_sort_ratio(item.output, item.expected_output),
                    "token_set_ratio": fuzz.token_set_ratio(item.output, item.expected_output),
                }
            }
        )


class LevenshteinScorer(BaseScorer):
    """
    Scorer that uses Levenshtein distance for similarity calculation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.8)
        self.normalize = self.config.get("normalize", True)
    
    @property
    def name(self) -> str:
        return "Levenshtein Distance"
    
    @property
    def description(self) -> str:
        return f"Uses Levenshtein distance to measure similarity (threshold: {self.threshold})"
    
    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score using Levenshtein distance.
        
        Args:
            item: The evaluation item to score
        
        Returns:
            ScorerResult with similarity score
        """
        if item.output is None:
            return ScorerResult(
                scorer_name="levenshtein",
                score=0.0,
                passed=False,
                reasoning="No output provided",
            )
        
        import Levenshtein
        
        output = item.output
        expected = item.expected_output
        
        # Optionally normalize
        if self.normalize:
            output = output.lower().strip()
            expected = expected.lower().strip()
        
        # Calculate distance and ratio
        distance = Levenshtein.distance(output, expected)
        max_len = max(len(output), len(expected))
        
        # Convert to similarity score (0-1)
        if max_len == 0:
            score = 1.0
        else:
            score = 1.0 - (distance / max_len)
        
        passed = score >= self.threshold
        
        return ScorerResult(
            scorer_name="levenshtein",
            score=score,
            passed=passed,
            reasoning=f"Edit distance: {distance}, similarity: {score:.2f}",
            details={
                "edit_distance": distance,
                "max_length": max_len,
                "threshold": self.threshold,
                "normalized": self.normalize,
            }
        )
