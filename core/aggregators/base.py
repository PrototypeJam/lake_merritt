# core/aggregators/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.data_models import EvaluationResults

class BaseAggregator(ABC):
    """Base for dataset-level metrics calculators."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def aggregate(self, results: EvaluationResults) -> Dict[str, Any]:
        """Return a flat dict of aggregate metrics."""
        raise NotImplementedError