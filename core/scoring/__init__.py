"""
Scoring module containing various evaluation scorers.
"""
from typing import Dict, Any, Type
from abc import ABC, abstractmethod

from core.data_models import EvaluationItem, ScorerResult


class BaseScorer(ABC):
    """Abstract base class for all scorers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the scorer with configuration.
        
        Args:
            config: Scorer-specific configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def score(self, item: EvaluationItem) -> ScorerResult:
        """
        Score an evaluation item.
        
        Args:
            item: The evaluation item to score
        
        Returns:
            ScorerResult with score and details
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this scorer."""
        pass
    
    @property
    def description(self) -> str:
        """Return a description of this scorer."""
        return f"{self.name} scorer"


# Import specific scorers
from core.scoring.exact_match import ExactMatchScorer
from core.scoring.fuzzy_match import FuzzyMatchScorer
from core.scoring.llm_judge import LLMJudgeScorer


# Registry of available scorers
SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "exact_match": ExactMatchScorer,
    "fuzzy_match": FuzzyMatchScorer,
    "llm_judge": LLMJudgeScorer,
}


def get_available_scorers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available scorers.
    
    Returns:
        Dictionary mapping scorer names to their information
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
        raise ValueError(f"Unknown scorer: {name}. Available: {list(SCORER_REGISTRY.keys())}")
    
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
