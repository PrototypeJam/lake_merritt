"""LLM-based scorer placeholder."""
from core.scoring import BaseScorer
from core.data_models import EvaluationItem, ScorerResult


class LLMJudgeScorer(BaseScorer):
    """Placeholder scorer that would use an LLM to judge outputs."""

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def description(self) -> str:
        return "Scores using an LLM as a judge (not implemented)"

    def score(self, item: EvaluationItem) -> ScorerResult:
        """Return a dummy score as the implementation placeholder."""
        return ScorerResult(
            scorer_name="llm_judge",
            score=0.0,
            passed=False,
            reasoning="LLM judge not implemented",
        )
