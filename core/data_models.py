"""Pydantic models used throughout the app."""
from __future__ import annotations
from typing import List, Any
from dataclasses import dataclass
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

