# In file: core/ingestion/csv_ingester.py

import pandas as pd
from typing import List, Dict, Any, Union, IO
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class CSVIngester(BaseIngester):
    """
    Ingests data from a CSV file. It requires an 'input' column and will
    ingest 'output' and 'expected_output' columns if they are present.
    
    This ingester is stateless and simple - it only maps CSV columns to
    EvaluationItem fields without enforcing workflow-specific validation.
    """

    def ingest(self, data: Union[str, IO, pd.DataFrame], config: Dict[str, Any]) -> List[EvaluationItem]:
        """
        Parses a CSV into a list of EvaluationItem objects.

        Args:
            data: The CSV data source (file path, file object, or DataFrame).
            config: Configuration dictionary (currently unused).

        Returns:
            A list of EvaluationItem objects.

        Raises:
            ValueError: If the 'input' column is missing or the CSV is empty.
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            if hasattr(data, 'seek'):
                data.seek(0)
            df = pd.read_csv(data)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        if "input" not in df.columns:
            raise ValueError("CSV is missing the required 'input' column.")

        items: List[EvaluationItem] = []
        for idx, row in df.iterrows():
            metadata = {
                c: row[c] for c in df.columns if c not in {"id", "input", "output", "expected_output"}
            }

            items.append(
                EvaluationItem(
                    id=str(row.get("id", idx + 1)),
                    input=str(row["input"]),
                    output=str(row["output"]) if "output" in row and pd.notna(row["output"]) else None,
                    expected_output=str(row["expected_output"]) if "expected_output" in row and pd.notna(row["expected_output"]) else None,
                    metadata=metadata
                )
            )
        return items