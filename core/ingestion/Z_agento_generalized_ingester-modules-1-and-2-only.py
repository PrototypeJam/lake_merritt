# Ingests Agento OTEL traces and yields one EvaluationItem per agent action.
# FILE: core/ingestion/agento_generalized_ingester.py

import json
from typing import Dict, Any, Iterable, Optional
from core.data_models import EvaluationItem

def get_attribute_value(span: Dict, key: str) -> Optional[Any]:
    """Helper to safely extract an attribute value from a span's attribute list."""
    for attr in span.get("attributes", []):
        if attr.get("key") == key:
            # The value is nested inside a type key, e.g., {"stringValue": "..."}
            return next(iter(attr.get("value", {}).values()), None)
    return None

def ingest_agento_trace(config: Dict[str, Any]) -> Iterable[EvaluationItem]:
    """
    Parses an Agento OTEL trace file (NDJSON format) and yields one
    EvaluationItem for each major agentic step (plan, draft, critique)
    by using semantic attributes, not indices.
    """
    trace_file = config.get("trace_file")
    if not trace_file:
        raise ValueError("Config for Python ingester must include 'trace_file' path or object.")

    # In Streamlit, the uploaded file is an in-memory object with a getvalue method
    if hasattr(trace_file, 'getvalue'):
        content = trace_file.getvalue().decode('utf-8')
        lines = content.splitlines()
    else: # For local file paths during testing
        with open(trace_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    all_spans = []
    for line in lines:
        if line.strip():
            try:
                data = json.loads(line)
                for rs in data.get("resourceSpans", []):
                    for ss in rs.get("scopeSpans", []):
                        all_spans.extend(ss.get("spans", []))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON line: {line[:100]}...")
                continue

    # Find the user_goal from a root agent span for global context
    user_goal = "Goal not found"
    for span in all_spans:
        if get_attribute_value(span, "openinference.span.kind") == "AGENT":
            goal = get_attribute_value(span, "user_goal")
            if goal:
                user_goal = goal
                break

    # Iterate through all spans and create an EvaluationItem for each labeled action
    for span in all_spans:
        step_type = get_attribute_value(span, "agento.step_type")
        if not step_type:
            continue

        # Defensively provide a default for output_content, though it's not strictly validated
        output_content = get_attribute_value(span, "gen_ai.response.content") or "No output content provided."

        if step_type == "plan":
            # The 'input' field is validated and cannot be empty.
            plan_input = user_goal or "No user goal provided in trace."
            yield EvaluationItem(
                id=span.get("spanId", "plan_span"),
                input=plan_input,
                output=output_content,
                expected_output="A sound, logical, and comprehensive project plan.",
                metadata={"step_type": "plan", "user_goal": user_goal}
            )
        
        elif step_type == "draft":
            step_name = get_attribute_value(span, "agento.step_name")
            
            # Both 'input' (instructions) and 'expected_output' (criteria) are validated.
            draft_input = get_attribute_value(span, "agento.instructions") or "No instructions provided in trace."
            criteria = get_attribute_value(span, "agento.criteria") or "No specific criteria provided in trace."

            if not get_attribute_value(span, "agento.instructions"):
                print(f"Warning: Missing instructions for draft step '{step_name}', using default.")
            if not get_attribute_value(span, "agento.criteria"):
                print(f"Warning: Missing criteria for draft step '{step_name}', using default.")

            yield EvaluationItem(
                id=span.get("spanId", f"draft_{step_name}"),
                input=draft_input,
                output=output_content,
                expected_output=criteria,
                metadata={"step_type": "draft", "step_name": step_name, "user_goal": user_goal}
            )
            
        elif step_type == "critique":
            step_name = get_attribute_value(span, "agento.step_name")
            
            # The 'input' for a critique is the draft content, which is validated.
            draft_content = get_attribute_value(span, "agento.draft_content") or "No draft content provided in trace."

            if not get_attribute_value(span, "agento.draft_content"):
                 print(f"Warning: Missing draft content for critique step '{step_name}', using default.")

            yield EvaluationItem(
                id=span.get("spanId", f"critique_{step_name}"),
                input=draft_content,
                output=output_content,
                expected_output="Insightful, actionable feedback to improve the draft.",
                metadata={"step_type": "critique", "step_name": step_name, "user_goal": user_goal}
            )