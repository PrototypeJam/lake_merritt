# In file: core/ingestion/generic_otel_ingester.py

import json
from typing import List, Dict, Any, Union, IO, Optional

from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class GenericOtelIngester(BaseIngester):
    """
    A trace-aware ingester for standard OpenTelemetry JSON traces.
    It can handle single JSON objects, a list of objects, or newline-delimited
    JSON (ndjson). It groups spans by trace_id and creates one EvaluationItem
    per trace, searching across all spans in that trace to find specified fields.
    """

    def ingest(self, data: Union[str, IO, Dict], config: Dict) -> List[EvaluationItem]:
        # --- Configuration from Eval Pack ---
        input_field_path = config.get("input_field", "attributes.input")
        output_field_path = config.get("output_field", "attributes.output")
        expected_output_field_path = config.get("expected_output_field")
        default_expected_output = config.get("default_expected_output", "No expected output specified")
        id_field_path = config.get("id_field", "trace_id")
        include_trace_context = config.get("include_trace_context", True)

        # --- Load and Parse Data ---
        # This now handles single, list, and newline-delimited JSON.
        if hasattr(data, 'getvalue'):
            content = data.getvalue().decode("utf-8")
        elif hasattr(data, 'read'):
            data.seek(0)
            content = data.read()
        else:
            content = data if isinstance(data, str) else json.dumps(data)

        raw_trace_objects = self._parse_json_input(content)
        
        # --- Group Spans by Trace ID ---
        all_spans = self._get_all_spans_from_payload(raw_trace_objects)
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
            
            if input_value is None:
                continue

            output_value = self._find_field_in_trace(span_list, output_field_path)
            expected_output = self._find_field_in_trace(span_list, expected_output_field_path)
            item_id = self._find_field_in_trace(span_list, id_field_path) or trace_id

            metadata = {"trace_id": trace_id}
            if include_trace_context:
                # Reconstruct a valid, single trace object for metadata
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

    def _parse_json_input(self, content: str) -> List[Dict]:
        """Parses a string that could be a single JSON object, a JSON array, or ndjson."""
        content = content.strip()
        if not content:
            return []
        
        # Try parsing as a single JSON array
        if content.startswith('[') and content.endswith(']'):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to ndjson if array parsing fails (e.g., malformed)
                pass

        # Try parsing as ndjson (or a single object)
        objects = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            try:
                obj, end_pos = decoder.raw_decode(content[pos:])
                objects.append(obj)
                pos += end_pos
                # Skip whitespace and newlines
                while pos < len(content) and content[pos].isspace():
                    pos += 1
            except json.JSONDecodeError:
                # This can happen if there's trailing non-JSON data, which we can ignore
                # if we have already parsed at least one object.
                if objects:
                    break 
                else:
                    raise # Re-raise if we couldn't parse any object at all.
        return objects


    def _get_all_spans_from_payload(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Extracts a flat list of all spans from a list of OTLP/JSON trace objects."""
        spans = []
        for trace_obj in data:
            for rs in trace_obj.get("resourceSpans", []):
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
            
            if part == 'attributes':
                remaining_path_key = '.'.join(parts[i+1:])
                attr_list = current_obj.get('attributes', [])
                if not isinstance(attr_list, list):
                    return None
                
                for attr in attr_list:
                    if attr.get('key') == remaining_path_key:
                        value_obj = attr.get('value', {})
                        return next(iter(value_obj.values()), None)
                return None # Attribute not found in this span
            
            if isinstance(current_obj, dict):
                current_obj = current_obj.get(part)
            else:
                return None
        
        return current_obj
