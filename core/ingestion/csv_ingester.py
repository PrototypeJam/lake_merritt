# core/ingestion/csv_ingester.py
import pandas as pd
from typing import List, Dict, Any, Union, IO
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class CSVIngester(BaseIngester):
    """Ingests rows from a CSV whose header contains at least: input, expected_output[, output]."""

    def ingest(self, data: Union[str, IO, pd.DataFrame], config: Dict) -> List[EvaluationItem]:
        """
        Ingests a CSV file or DataFrame and returns a list of EvaluationItem objects.
        - In Mode A (evaluate_existing): requires 'input' and 'expected_output' columns.
        - In Mode B (generate_then_evaluate): only requires 'input' column.
        """
        df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data
        mode = config.get("mode", "evaluate_existing")

        # --- Mode-aware column requirements ---
        if mode == "generate_then_evaluate":
            required_columns = {"input"}
        else:
            required_columns = {"input", "expected_output"}

        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required column(s) for mode '{mode}': {', '.join(missing)}")

        items: List[EvaluationItem] = []
        for idx, row in df.iterrows():
            # Defensive: always ensure metadata is a dict, never a Series or NaN
            metadata = {c: row[c] for c in df.columns if c not in {"id","input","output","expected_output"}}
            if not isinstance(metadata, dict):
                metadata = dict(metadata)
            items.append(
                EvaluationItem(
                    id=str(row.get("id", idx + 1)),
                    input=str(row["input"]),
                    output=str(row.get("output", "")) if mode == "evaluate_existing" else None,
                    # For generation, expected_output may be missing, so default to empty.
                    expected_output=str(row.get("expected_output", "")),
                    metadata=metadata
                )
            )
        return items
