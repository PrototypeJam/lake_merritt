"""Actor output generation."""
from typing import Iterable
from .data_models import EvalRecord


def generate(records: Iterable[EvalRecord]) -> Iterable[EvalRecord]:
    for record in records:
        # placeholder generation
        record.output = record.prompt
        yield record
