"""Scoring module containing various evaluation scorers."""

from typing import Any, Dict, Type

from core.scoring.base import BaseScorer
# Import specific scorers - do this after BaseScorer is defined
from core.scoring.exact_match import ExactMatchScorer
from core.scoring.fuzzy_match import FuzzyMatchScorer
from core.scoring.llm_judge import LLMJudgeScorer
from core.scoring.otel.criteria_selection_judge import \
    CriteriaSelectionJudgeScorer

# Try to import optional scorer variants
try:
    from core.scoring.exact_match import (CaseInsensitiveExactMatchScorer,
                                          NormalizedExactMatchScorer)
except ImportError:
    CaseInsensitiveExactMatchScorer = None
    NormalizedExactMatchScorer = None

try:
    from core.scoring.llm_judge import StructuredLLMJudgeScorer
except ImportError:
    StructuredLLMJudgeScorer = None


# Registry of available scorers
SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "exact_match": ExactMatchScorer,
    "fuzzy_match": FuzzyMatchScorer,
    "llm_judge": LLMJudgeScorer,
    "criteria_selection_judge": CriteriaSelectionJudgeScorer,
}

# Add optional scorers if available
if CaseInsensitiveExactMatchScorer:
    SCORER_REGISTRY["case_insensitive_exact_match"] = CaseInsensitiveExactMatchScorer
if NormalizedExactMatchScorer:
    SCORER_REGISTRY["normalized_exact_match"] = NormalizedExactMatchScorer
if StructuredLLMJudgeScorer:
    SCORER_REGISTRY["structured_llm_judge"] = StructuredLLMJudgeScorer


def get_available_scorers() -> Dict[str, Dict[str, Any]]:
    """Return metadata about each registered scorer.

    The function iterates over :data:`SCORER_REGISTRY`, instantiates each
    scorer class, and collects human-friendly information. The returned
    dictionary maps the scorer's registration name to its class object,
    display name and description.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of scorer names to metadata.
    """
    scorers_info = {}

    for name, scorer_class in SCORER_REGISTRY.items():
        scorer = scorer_class()
        scorers_info[name] = {
            "class": scorer_class,
            "display_name": scorer.name,
            "description": scorer.description,
        }

    return scorers_info


def create_scorer(name: str, config: Dict[str, Any] = None) -> BaseScorer:
    """
    Create a scorer instance by name.

    Args:
        name: Name of the scorer
        config: Configuration for the scorer

    Returns:
        Scorer instance

    Raises:
        ValueError: If scorer name is not found
    """
    if name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer: {name}. Available: {list(SCORER_REGISTRY.keys())}"
        )

    scorer_class = SCORER_REGISTRY[name]
    return scorer_class(config)


def register_scorer(name: str, scorer_class: Type[BaseScorer]) -> None:
    """
    Register a new scorer class.

    Args:
        name: Name to register the scorer under
        scorer_class: The scorer class
    """
    if not issubclass(scorer_class, BaseScorer):
        raise TypeError("Scorer class must inherit from BaseScorer")

    SCORER_REGISTRY[name] = scorer_class


__all__ = [
    "BaseScorer",
    "ExactMatchScorer",
    "FuzzyMatchScorer",
    "LLMJudgeScorer",
    "get_available_scorers",
    "create_scorer",
    "register_scorer",
]
