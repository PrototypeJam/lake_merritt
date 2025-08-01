# Ingests Agento OTEL traces and yields one EvaluationItem per agent action.
# FILE: core/ingestion/agento_generalized_ingester.py

import json
from typing import Dict, Any, Iterable, Optional
from core.data_models import EvaluationItem

def get_attribute_value(span: Dict, key: str) -> Optional[Any]:
    """Helper to safely extract an attribute value from a span's attribute list."""
    for attr in span.get("attributes", []):
        if attr.get("key") == key:
            return next(iter(attr.get("value", {}).values()), None)
    return None

def ingest_agento_trace(config: Dict[str, Any]) -> Iterable[EvaluationItem]:
    """
    Parses an Agento OTEL trace file (NDJSON format) and yields one
    EvaluationItem for each major agentic step by using semantic attributes.
    """
    trace_file = config.get("trace_file")
    if not trace_file:
        raise ValueError("Config for Python ingester must include 'trace_file' path or object.")

    if hasattr(trace_file, 'getvalue'):
        content = trace_file.getvalue().decode('utf-8')
        lines = content.splitlines()
    else:
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

    user_goal = "Goal not found"
    for span in all_spans:
        if get_attribute_value(span, "openinference.span.kind") == "AGENT":
            goal = get_attribute_value(span, "user_goal")
            if goal:
                user_goal = goal
                break

    for span in all_spans:
        step_type = get_attribute_value(span, "agento.step_type")
        if not step_type:
            continue

        output_content = get_attribute_value(span, "gen_ai.response.content") or get_attribute_value(span, "agento.final_content") or "No output content provided."

        if step_type == "plan":
            yield EvaluationItem(
                id=span.get("spanId", "plan_span"),
                input=user_goal or "No user goal provided.",
                output=output_content,
                expected_output="A sound, logical, and comprehensive project plan.",
                metadata={"step_type": "plan", "user_goal": user_goal}
            )
        
        elif step_type == "draft":
            step_name = get_attribute_value(span, "agento.step_name")
            criteria = get_attribute_value(span, "agento.criteria") or "No criteria provided."
            instructions = get_attribute_value(span, "agento.instructions") or "No instructions provided."
            yield EvaluationItem(
                id=span.get("spanId", f"draft_{step_name}"),
                input=instructions,
                output=output_content,
                expected_output=criteria,
                metadata={"step_type": "draft", "step_name": step_name, "user_goal": user_goal}
            )
            
        elif step_type == "critique":
            step_name = get_attribute_value(span, "agento.step_name")
            draft_content = get_attribute_value(span, "agento.draft_content") or "No draft content provided."
            yield EvaluationItem(
                id=span.get("spanId", f"critique_{step_name}"),
                input=draft_content,
                output=output_content,
                expected_output="Insightful, actionable feedback to improve the draft.",
                metadata={"step_type": "critique", "step_name": step_name, "user_goal": user_goal}
            )

        ### NEW INGESTER LOGIC ###
        elif step_type == "accepted_revision":
            step_name = get_attribute_value(span, "agento.step_name")
            revision_request = get_attribute_value(span, "agento.revision_request") or "No revision request found."
            final_content = get_attribute_value(span, "agento.final_content") or "No final content found."
            yield EvaluationItem(
                id=span.get("spanId", f"accepted_{step_name}"),
                input=revision_request,
                output=final_content,
                expected_output="The output should faithfully and completely implement the revision request.",
                metadata={"step_type": "accepted_revision", "step_name": step_name, "user_goal": user_goal}
            )

        elif step_type == "timed_out_revision":
            step_name = get_attribute_value(span, "agento.step_name")
            revision_request = get_attribute_value(span, "agento.revision_request") or "No revision request found."
            last_draft = get_attribute_value(span, "agento.last_attempted_draft") or "No last draft found."
            final_critique = get_attribute_value(span, "agento.final_critique") or "No final critique found."
            yield EvaluationItem(
                id=span.get("spanId", f"timeout_{step_name}"),
                input=revision_request,
                output=last_draft,
                expected_output=final_critique,
                metadata={"step_type": "timed_out_revision", "step_name": step_name, "user_goal": user_goal, "final_critique": final_critique}
            )

        elif step_type == "holistic_review":
            final_plan = get_attribute_value(span, "agento.final_plan_content") or "{}"
            yield EvaluationItem(
                id=span.get("spanId", "holistic_review"),
                input=user_goal or "No user goal provided.",
                output=final_plan,
                expected_output="The final plan should be coherent, comprehensive, and fully aligned with the user's goal.",
                metadata={"step_type": "holistic_review", "user_goal": user_goal}
            )