"""Base scorer class definition."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from core.data_models import EvaluationItem, ScorerResult


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def score(self, item: EvaluationItem) -> ScorerResult:
        """Score an evaluation item and return the result."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human friendly name for the scorer."""

    @property
    def description(self) -> str:
        return f"{self.name} scorer"
