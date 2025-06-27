# core/ingestion/generic_otel_ingester.py
import json
from typing import List, Dict, Any, Union, IO, Optional
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem
from core.eval_pack.schema import SpanKind

class GenericOtelIngester(BaseIngester):
    """
    Generic ingester for OpenTelemetry traces.
    
    This ingester can handle various OTEL trace formats and extract evaluation items
    based on configurable span filters and field mappings.
    
    Configuration options:
    - span_kind_filter: Only process spans of specific kinds (e.g., "LLM", "TOOL")
    - input_field: Path to extract input from span attributes (dot notation supported)
    - output_field: Path to extract output from span attributes
    - expected_output_field: Path to extract expected output (optional)
    - default_expected_output: Default value if expected output not found
    - id_field: Path to extract ID (defaults to span_id)
    - include_trace_context: Whether to include full trace in metadata
    """
    
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.span_kind_filter = self.config.get("span_kind_filter", [])
        self.input_field = self.config.get("input_field", "attributes.input")
        self.output_field = self.config.get("output_field", "attributes.output")
        self.expected_output_field = self.config.get("expected_output_field")
        self.default_expected_output = self.config.get("default_expected_output", "No expected output specified")
        self.id_field = self.config.get("id_field", "span_id")
        self.include_trace_context = self.config.get("include_trace_context", True)
    
    def ingest(self, data: Union[str, IO, Dict, List], config: Dict) -> List[EvaluationItem]:
        """Ingest OTEL trace data and convert to evaluation items."""
        # Parse data
        if isinstance(data, str):
            trace_data = json.loads(data)
        elif hasattr(data, 'read'):
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
            # Extract spans from trace
            spans = self._extract_spans(trace)
            
            for span in spans:
                # Apply span kind filter if configured
                if self.span_kind_filter:
                    span_kind = span.get("kind", "").upper()
                    if span_kind not in self.span_kind_filter:
                        continue
                
                # Extract fields using dot notation
                input_value = self._extract_field(span, self.input_field)
                if not input_value:
                    continue  # Skip spans without input
                
                output_value = self._extract_field(span, self.output_field)
                expected_output = None
                
                if self.expected_output_field:
                    expected_output = self._extract_field(span, self.expected_output_field)
                
                if not expected_output:
                    expected_output = self.default_expected_output
                
                # Extract ID
                item_id = str(self._extract_field(span, self.id_field) or span.get("span_id") or span.get("id"))
                
                # Build metadata
                metadata = {
                    "span_kind": span.get("kind"),
                    "span_name": span.get("name"),
                    "trace_id": trace.get("trace_id") or span.get("trace_id"),
                    "parent_span_id": span.get("parent_span_id"),
                    "duration_ms": span.get("duration_ms"),
                    "status": span.get("status"),
                    "attributes": span.get("attributes", {})
                }
                
                if self.include_trace_context:
                    metadata["otel_trace"] = trace
                
                # Add any custom metadata fields from config
                for key, path in self.config.get("metadata_fields", {}).items():
                    value = self._extract_field(span, path)
                    if value is not None:
                        metadata[key] = value
                
                items.append(
                    EvaluationItem(
                        id=item_id,
                        input=str(input_value),
                        output=str(output_value) if output_value else None,
                        expected_output=str(expected_output),
                        metadata=metadata
                    )
                )
        
        return items
    
    def _extract_spans(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract spans from various trace formats."""
        # Standard OTEL format
        if "spans" in trace:
            return trace["spans"]
        
        # Resource spans format
        elif "resource_spans" in trace:
            spans = []
            for resource_span in trace["resource_spans"]:
                for scope_span in resource_span.get("scope_spans", []):
                    spans.extend(scope_span.get("spans", []))
            return spans
        
        # Single span as trace
        elif "span_id" in trace or "id" in trace:
            return [trace]
        
        # Custom formats (e.g., steps-based)
        elif "steps" in trace:
            # Convert steps to span-like format
            spans = []
            for idx, step in enumerate(trace["steps"]):
                span = {
                    "id": f"step_{idx}",
                    "name": step.get("stage", "unknown"),
                    "kind": self._infer_span_kind(step),
                    "attributes": step.get("outputs", {}),
                    "timestamp": step.get("timestamp")
                }
                # Map common step fields
                if "inputs" in step:
                    span["attributes"]["input"] = step["inputs"]
                if "outputs" in step:
                    span["attributes"]["output"] = step["outputs"]
                spans.append(span)
            return spans
        
        return []
    
    def _infer_span_kind(self, step: Dict[str, Any]) -> str:
        """Infer span kind from step data."""
        stage = step.get("stage", "").lower()
        if "llm" in stage or "generate" in stage:
            return "LLM"
        elif "tool" in stage:
            return "TOOL"
        elif "agent" in stage:
            return "AGENT"
        elif "retrieve" in stage:
            return "RETRIEVER"
        else:
            return "CHAIN"
    
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