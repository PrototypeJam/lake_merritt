# core/ingestion/base.py
from abc import ABC, abstractmethod
from typing import List, Any
from core.data_models import EvaluationItem

class BaseIngester(ABC):
    @abstractmethod
    def ingest(self, data: Any, config: dict) -> List[EvaluationItem]:
        """Parses raw data into a list of evaluation items."""
        pass