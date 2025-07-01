# In file: core/ingestion/generic_otel_ingester.py

import json
from typing import List, Dict, Any, Union, IO, Optional
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class GenericOtelIngester(BaseIngester):
    """
    Generic ingester for OpenTelemetry traces.
    
    This ingester can handle various OTEL trace formats and extract evaluation items
    based on configurable span filters and field mappings from the Eval Pack.
    """
    
    # The __init__ method is removed. Configuration is handled in ingest().

    def ingest(self, data: Union[str, IO, Dict, List], config: Dict) -> List[EvaluationItem]:
        """Ingest OTEL trace data and convert to evaluation items."""

        # --- FIX 1: PARSE CONFIGURATION AT RUNTIME ---
        # This ensures the settings from the Eval Pack are used.
        span_kind_filter = config.get("span_kind_filter", [])
        input_field = config.get("input_field", "attributes.input")
        output_field = config.get("output_field", "attributes.output")
        expected_output_field = config.get("expected_output_field")
        default_expected_output = config.get("default_expected_output", "No expected output specified")
        id_field = config.get("id_field", "span_id")
        include_trace_context = config.get("include_trace_context", True)
        metadata_fields = config.get("metadata_fields", {})
        
        # --- FIX 2: ROBUST JSON PARSING ---
        if isinstance(data, str):
            trace_data = json.loads(data)
        elif hasattr(data, 'getvalue'): # Handles Streamlit's UploadedFile (BytesIO)
            trace_data = json.loads(data.getvalue().decode("utf-8"))
        elif hasattr(data, 'read'): # Handles standard file objects
            data.seek(0)
            trace_data = json.load(data)
        else:
            trace_data = data
        
        # Handle both single trace and list of traces
        if isinstance(trace_data, dict):
            traces = [trace_data]
        elif isinstance(trace_data, list):
            traces = trace_data
        else:
            raise ValueError("OTEL data must be a trace object or list of traces")
        
        items: List[EvaluationItem] = []
        
        for trace in traces:
            spans = self._extract_spans(trace)
            
            for span in spans:
                if span_kind_filter:
                    span_kind = span.get("kind", "").upper()
                    if span_kind not in span_kind_filter:
                        continue
                
                input_value = self._extract_field(span, input_field)
                if not input_value:
                    continue
                
                output_value = self._extract_field(span, output_field)
                expected_output = self._extract_field(span, expected_output_field) if expected_output_field else default_expected_output
                
                item_id = str(self._extract_field(span, id_field) or span.get("span_id") or span.get("id"))
                
                metadata = {
                    "span_kind": span.get("kind"), "span_name": span.get("name"),
                    "trace_id": trace.get("trace_id") or span.get("trace_id"),
                    "parent_span_id": span.get("parent_span_id"),
                    "attributes": span.get("attributes", {})
                }
                
                if include_trace_context:
                    metadata["otel_trace"] = trace
                
                for key, path in metadata_fields.items():
                    value = self._extract_field(span, path)
                    if value is not None:
                        metadata[key] = value
                
                items.append(
                    EvaluationItem(
                        id=item_id,
                        input=str(input_value),
                        output=str(output_value) if output_value is not None else "",
                        expected_output=str(expected_output),
                        metadata=metadata
                    )
                )
        
        return items

    def _extract_spans(self, trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts a list of spans from a raw OTEL trace object."""
        # Standard OTLP/JSON format with resource spans
        if "resource_spans" in trace_data:
            spans = []
            for rs in trace_data.get("resource_spans", []):
                for ss in rs.get("scope_spans", []):
                    spans.extend(ss.get("spans", []))
            return spans
        # Agento-style trace with a root "traces" key
        if "traces" in trace_data and isinstance(trace_data["traces"], list):
            all_spans = []
            for trace in trace_data["traces"]:
                all_spans.extend(self._extract_spans(trace))
            return all_spans
        # Agento-style trace with a root "steps" key
        if "steps" in trace_data:
            return trace_data["steps"]
        # A simple list of spans
        if isinstance(trace_data, list):
            return trace_data
        return []

    def _extract_field(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        """Extract field from nested dict using dot notation."""
        if not path:
            return None
        
        parts = path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current