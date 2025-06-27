# core/ingestion/json_ingester.py
import json
from typing import List, Dict, Any, Union, IO
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class JSONIngester(BaseIngester):
    """Ingests evaluation data from JSON format."""
    REQUIRED = {"input", "expected_output"}

    def ingest(self, data: Union[str, IO, Dict, List], config: Dict) -> List[EvaluationItem]:
        # Parse data if it's a string or file-like object
        if isinstance(data, str):
            json_data = json.loads(data)
        elif hasattr(data, 'read'):
            json_data = json.load(data)
        else:
            json_data = data
        
        # Ensure data is a list
        if isinstance(json_data, dict):
            json_data = [json_data]
        elif not isinstance(json_data, list):
            raise ValueError("JSON data must be a list of objects or a single object")
        
        mode = config.get("mode", "evaluate_existing")
        items: List[EvaluationItem] = []
        
        for idx, item in enumerate(json_data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {idx} is not a dictionary")
            
            # Check required fields
            missing = self.REQUIRED.difference(item.keys())
            if missing:
                raise ValueError(f"Item at index {idx} missing required field(s): {', '.join(missing)}")
            
            # Extract core fields
            eval_item = EvaluationItem(
                id=str(item.get("id", idx + 1)),
                input=str(item["input"]),
                output=str(item.get("output", "")) if mode == "evaluate_existing" and "output" in item else None,
                expected_output=str(item["expected_output"]),
                metadata={k: v for k, v in item.items() if k not in {"id", "input", "output", "expected_output"}}
            )
            items.append(eval_item)
        
        return items