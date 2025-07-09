# In file: core/ingestion/generic_otel_ingester.py

import json
from typing import List, Dict, Any, Union, IO, Optional

from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class GenericOtelIngester(BaseIngester):
    """
    A trace-aware ingester for standard OpenTelemetry JSON traces.
    It groups spans by trace_id and creates one EvaluationItem per trace,
    searching across all spans in that trace to find the specified fields.
    """

    def ingest(self, data: Union[str, IO, Dict], config: Dict) -> List[EvaluationItem]:
        """Ingest OTEL trace data, group by trace, and extract evaluation items."""
        # --- Configuration from Eval Pack ---
        input_field_path = config.get("input_field", "attributes.input")
        output_field_path = config.get("output_field", "attributes.output")
        expected_output_field_path = config.get("expected_output_field")
        default_expected_output = config.get("default_expected_output", "No expected output specified")
        id_field_path = config.get("id_field", "trace_id")
        include_trace_context = config.get("include_trace_context", True)

        # --- Load and Parse Data ---
        if isinstance(data, str):
            raw_data = json.loads(data)
        elif hasattr(data, 'getvalue'): # Handles Streamlit's UploadedFile
            raw_data = json.loads(data.getvalue().decode("utf-8"))
        elif hasattr(data, 'read'): # Handles standard file objects
            data.seek(0)
            raw_data = json.load(data)
        else:
            raw_data = data
        
        # --- Group Spans by Trace ID ---
        all_spans = self._get_all_spans_from_payload(raw_data)
        traces: Dict[str, List[Dict[str, Any]]] = {}
        for span in all_spans:
            trace_id = span.get("traceId")
            if trace_id:
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(span)

        # --- Create One EvaluationItem Per Trace ---
        items: List[EvaluationItem] = []
        for trace_id, span_list in traces.items():
            input_value = self._find_field_in_trace(span_list, input_field_path)
            
            # If we can't find the primary input, we can't create a useful item.
            if input_value is None:
                continue

            output_value = self._find_field_in_trace(span_list, output_field_path)
            expected_output = self._find_field_in_trace(span_list, expected_output_field_path)
            
            item_id = self._find_field_in_trace(span_list, id_field_path) or trace_id

            metadata = {"trace_id": trace_id}
            if include_trace_context:
                metadata["otel_trace"] = {"resourceSpans": [{"scopeSpans": [{"spans": span_list}]}]}

            items.append(
                EvaluationItem(
                    id=str(item_id),
                    input=str(input_value),
                    output=str(output_value) if output_value is not None else "",
                    expected_output=str(expected_output) if expected_output is not None else default_expected_output,
                    metadata=metadata
                )
            )
        
        return items

    def _get_all_spans_from_payload(self, data: Dict) -> List[Dict[str, Any]]:
        """Extracts a flat list of all spans from a raw OTLP/JSON payload."""
        spans = []
        for rs in data.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                spans.extend(ss.get("spans", []))
        return spans

    def _find_field_in_trace(self, span_list: List[Dict], path: Optional[str]) -> Optional[Any]:
        """Search all spans in a trace for the first occurrence of a field specified by dot notation."""
        if not path:
            return None
        
        for span in span_list:
            value = self._extract_field_from_span(span, path)
            if value is not None:
                return value
        return None

    def _extract_field_from_span(self, span: Dict, path: str) -> Optional[Any]:
        """Extracts a field from a single span using dot notation, handling the OTel attribute format."""
        parts = path.split('.')
        current_obj = span
        
        for i, part in enumerate(parts):
            if current_obj is None:
                return None

            # Special handling for the 'attributes' part of the path
            if part == 'attributes' and isinstance(current_obj, list):
                 # This happens when we are already inside an attribute list traversal
                return None # Path is invalid, e.g., attributes.attributes.
            
            if part == 'attributes':
                # The rest of the path must be found within the attributes list
                remaining_path = '.'.join(parts[i+1:])
                attr_list = current_obj.get('attributes', [])
                if not isinstance(attr_list, list):
                    return None # Attributes format is not the expected list
                
                for attr in attr_list:
                    if attr.get('key') == remaining_path:
                        # OTel values are nested, e.g., {"stringValue": "..."}
                        value_obj = attr.get('value', {})
                        return next(iter(value_obj.values()), None) # Return the first value in the dict
                return None # Attribute not found
            
            # Standard dictionary access
            if isinstance(current_obj, dict):
                current_obj = current_obj.get(part)
            else:
                return None
        
        return current_obj
