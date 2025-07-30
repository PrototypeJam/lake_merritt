# FILE: core/ingestion/agento_analytical_ingester.py
"""
Lake Merritt Enhanced Agento Ingester
-------------------------------------
Emits richer EvaluationItems and supports cross-span analysis.

Modes:
- default:            one item per span (superset of stock ingester fields)
- plan_delta:         one item per trace comparing first plan vs final plan
- revision_pairs:     one item per consecutive (draft, accepted_revision)
- context_aware_steps: one item per draft step incl. full outline
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from collections import defaultdict

# This import will work once the file is placed inside the Lake Merritt project structure.
from core.data_models import EvaluationItem

logger = logging.getLogger(__name__)

def _attr(span: Dict[str, Any], key: str) -> Optional[Any]:
    """Handle both OTLP attribute list format and flattened dict."""
    attrs = span.get("attributes")
    if isinstance(attrs, list):  # OTLP raw format
        for pair in attrs:
            if pair.get("key") == key:
                val = pair.get("value", {})
                return next(iter(val.values()), None)
        return None
    if isinstance(attrs, dict):  # Potentially flattened by prior tooling
        return attrs.get(key)
    return None

def _load_spans(trace_file: Any) -> List[Dict[str, Any]]:
    """Loads all spans from a file-like object or a file path."""
    spans: List[Dict[str, Any]] = []
    
    lines = []
    if hasattr(trace_file, 'getvalue'):
        content = trace_file.getvalue().decode('utf-8')
        lines = content.splitlines()
    else: # Fallback for file paths
        with open(trace_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    for raw in lines:
        if not raw.strip():
            continue
        try:
            j = json.loads(raw)
            for rs in j.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    spans.extend(ss.get("spans", []))
        except Exception as exc:
            logger.warning(f"[ingester] skipped corrupt line: {exc}")
    return spans

def _first(spans: List[Dict[str, Any]], step_type: str) -> Optional[Dict[str, Any]]:
    return next((s for s in spans if _attr(s, "agento.step_type") == step_type), None)

def _last(spans: List[Dict[str, Any]], step_type: str) -> Optional[Dict[str, Any]]:
    return next((s for s in reversed(spans) if _attr(s, "agento.step_type") == step_type), None)

# Note: The @register_ingester decorator is conceptual. We will call this function
# via the 'script_path' and 'entry_function' config in the Eval Pack.
def ingest_agento_analytical_trace(
    config: Dict[str, Any]
) -> Generator[EvaluationItem, None, None]:
    """
    This function is the entry point called by the PythonIngester.
    The `config` dict will contain `trace_file` and `mode`.
    """
    trace_file = config.get("trace_file")
    mode = config.get("mode", "default")
    
    spans = _load_spans(trace_file)
    if not spans:
        return

    # -------- global artefacts -------- #
    root_span = _first(spans, "plan")
    user_goal = _attr(root_span, "agento.user_goal") if root_span else "Unknown"
    first_plan_span = root_span
    final_plan_span = _last(spans, "holistic_review") or _last(spans, "accepted_revision")

    first_plan_txt = _attr(first_plan_span, "gen_ai.response.content") if first_plan_span else None
    final_plan_txt = _attr(final_plan_span, "agento.final_plan_content") or (_attr(final_plan_span, "agento.final_content") if final_plan_span else None)

    outline_json: Optional[str] = None
    if first_plan_txt:
        try:
            outline_candidate = json.loads(first_plan_txt)
            outline = outline_candidate.get("Detailed_Outline", outline_candidate)
            outline_json = json.dumps(outline, indent=2)
        except Exception:
            outline_json = None

    # -------- mode: plan_delta -------- #
    if mode == "plan_delta":
        if not (first_plan_txt and final_plan_txt):
            logger.warning("Missing initial or final plan for plan_delta mode.")
            return
        yield EvaluationItem(
            id="plan_delta",
            input=user_goal or "",
            output=final_plan_txt,
            expected_output=first_plan_txt,
            metadata={"analytical_type": "plan_delta", "user_goal": user_goal}
        )
        return

    # -------- mode: revision_pairs -------- #
    if mode == "revision_pairs":
        chains: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in spans:
            name = _attr(s, "agento.step_name")
            if name:
                chains[name].append(s)

        for step, chain in chains.items():
            draft_buffer: Optional[Dict[str, Any]] = None
            for s in chain:
                stype = _attr(s, "agento.step_type")
                if stype in ("draft", "revision_draft"):
                    draft_buffer = s
                if stype == "accepted_revision" and draft_buffer:
                    yield EvaluationItem(
                        id=f"revpair_{s.get('spanId')}",
                        input=_attr(draft_buffer, "gen_ai.response.content") or "",
                        output=_attr(s, "agento.final_content") or "",
                        expected_output=_attr(s, "agento.revision_request") or "",
                        metadata={"analytical_type": "revision_pair", "step_name": step, "user_goal": user_goal}
                    )
                    draft_buffer = None
        return

    # -------- mode: context_aware_steps -------- #
    if mode == "context_aware_steps":
        for s in spans:
            if _attr(s, "agento.step_type") != "draft":
                continue
            step_name = _attr(s, "agento.step_name") or "UNKNOWN"
            yield EvaluationItem(
                id=f"context_{s.get('spanId')}",
                input=user_goal or "",
                output=_attr(s, "gen_ai.response.content") or "",
                expected_output=_attr(s, "agento.instructions") or "",
                metadata={
                    "analytical_type": "step_context",
                    "step_name": step_name,
                    "user_goal": user_goal,
                    "full_plan_outline": outline_json
                }
            )
        return

    # -------- default: richer single-span emission -------- #
    for s in spans:
        attrs_dict = {}
        attrs = s.get("attributes", [])
        if isinstance(attrs, list):
            for pair in attrs:
                attrs_dict[pair.get("key")] = next(iter(pair.get("value", {}).values()), None)
        else:
            attrs_dict = attrs or {}

        yield EvaluationItem(
            id=s.get("spanId"),
            input=attrs_dict.get("agento.draft_content", user_goal or ""),
            output=attrs_dict.get("gen_ai.response.content", attrs_dict.get("agento.final_content", "")),
            expected_output=attrs_dict.get("agento.criteria", attrs_dict.get("agento.revision_request", "")),
            metadata={
                "user_goal": user_goal,
                "step_type": attrs_dict.get("agento.step_type", "UNKNOWN"),
                "step_name": attrs_dict.get("agento.step_name", ""),
            }
        )