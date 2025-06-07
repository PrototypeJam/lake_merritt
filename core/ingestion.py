"""CSV ingestion utilities."""
from typing import List
import csv
from .data_models import EvalRecord


def load_csv(fp) -> List[EvalRecord]:
    reader = csv.DictReader(fp)
    records = [EvalRecord(**row) for row in reader]
    return records
