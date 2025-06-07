"""Pydantic models used throughout the app."""
from __future__ import annotations
from typing import List, Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
try:
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover - fallback for limited env
    from dataclasses import dataclass, asdict

    class BaseModel:
        """Very small subset of Pydantic's BaseModel."""

        def dict(self):
            return asdict(self)

    def dataclass(_cls=None, **kwargs):  # type: ignore
        from dataclasses import dataclass as real_dataclass

        def wrap(cls):
            return real_dataclass(cls)

        return wrap if _cls is None else wrap(_cls)


@dataclass
class EvalRecord(BaseModel):
    prompt: str
    expected: str
    output: str | None = None


@dataclass
class Score(BaseModel):
    value: float
    meta: Any | None = None


@dataclass
class RunMetadata(BaseModel):
    scorer: str


@dataclass
class RunResult(BaseModel):
    records: List[EvalRecord]
    scores: List[Score]
    metadata: RunMetadata


class EvaluationMode(str, Enum):
    """Modes for running evaluations."""

    EVALUATE_EXISTING = "evaluate_existing"
    GENERATE_THEN_EVALUATE = "generate_then_evaluate"


@dataclass
class ScorerResult(BaseModel):
    scorer_name: str
    score: float
    passed: bool
    reasoning: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationItem(BaseModel):
    id: Any
    input: str
    expected_output: str
    output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: List[ScorerResult] = field(default_factory=list)


@dataclass
class EvaluationResults(BaseModel):
    items: List[EvaluationItem]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def model_dump_json(self, indent: int | None = None) -> str:
        import json
        return json.dumps(self.dict(), indent=indent)

