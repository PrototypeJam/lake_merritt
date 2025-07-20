"""Base scorer class definition."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from core.data_models import EvaluationItem, ScorerResult


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    requires_api_key: bool = False

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """
        The __init__ config is now only for default or shared settings.
        Stage-specific config will be passed directly to the score method.
        """
        self.config = config or {}

    ### FIX #1: MODIFIED METHOD SIGNATURE ###
    # The score method now accepts a stage_config dictionary, making scorers
    # stateless and aware of the specific pipeline stage they are executing for.
    @abstractmethod
    def score(self, item: EvaluationItem, stage_config: Dict[str, Any]) -> ScorerResult:
        """Score an evaluation item using stage-specific configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human friendly name for the scorer."""

    @property
    def description(self) -> str:
        return f"{self.name} scorer"