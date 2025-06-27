# core/ingestion/csv_ingester.py
import pandas as pd
from typing import List, Dict, Any, Union, IO
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class CSVIngester(BaseIngester):
    """Ingests rows from a CSV whose header contains at least: input, expected_output[, output]."""
    REQUIRED = {"input", "expected_output"}

    def ingest(self, data: Union[str, IO, pd.DataFrame], config: Dict) -> List[EvaluationItem]:
        df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data
        missing = self.REQUIRED.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")
        mode = config.get("mode", "evaluate_existing")
        items: List[EvaluationItem] = []
        for idx, row in df.iterrows():
            items.append(
                EvaluationItem(
                    id=str(row.get("id", idx + 1)),
                    input=str(row["input"]),
                    output=str(row.get("output", "")) if mode == "evaluate_existing" else None,
                    expected_output=str(row["expected_output"]),
                    metadata={c: row[c] for c in df.columns if c not in {"id","input","output","expected_output"}}
                )
            )
        return items