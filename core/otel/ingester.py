import json, logging
from datetime import datetime
from typing import List, Dict, Any

from core.data_models import EvaluationItem

log = logging.getLogger(__name__)

class OTelTraceIngester:
    """Convert the custom manual_traces.json format into EvaluationItem objects."""

    def ingest_str(self, raw: str) -> List[EvaluationItem]:
        data = json.loads(raw)
        items: List[EvaluationItem] = []
        for tr in data.get("traces", []):
            meta = self._extract_fields(tr)
            items.append(
                EvaluationItem(
                    id=meta["id"],
                    input=meta["user_goal"] or "Unknown goal",
                    output=meta["final_output"] or "[]",
                    expected_output="Selected criteria should appropriately match the user goal",
                    metadata={**meta, "otel_trace": tr},
                )
            )
        log.info("Loaded %d traces", len(items))
        return items

    # ---------- helpers ---------- #
    def _extract_fields(self, tr: Dict[str, Any]) -> Dict[str, Any]:
        m: Dict[str, Any] = {"id": tr.get("id")}
        for s in tr.get("steps", []):
            stg, outp = s.get("stage"), s.get("outputs", {})
            if stg == "user_input":
                m["user_goal"] = outp.get("user_goal")
            elif stg == "search_complete":
                m["search_summary"] = outp.get("search_summary")
            elif stg == "criteria_generation_complete":
                m["generated_criteria"] = outp.get("generated_criteria", [])
            elif stg == "criteria_evaluation_complete":
                m["selected_criteria"] = outp.get("selected_criteria", [])

        # Extract just the criteria text if it's in the detailed format
        selected = m.get("selected_criteria", [])
        if selected and isinstance(selected[0], dict) and "criteria" in selected[0]:
            # Extract just the criteria field from each dict
            criteria_list = [{"criteria": item["criteria"]} for item in selected]
            m["final_output"] = json.dumps(criteria_list, indent=2)
        else:
            m["final_output"] = json.dumps(selected, indent=2)

        # --- fixed duration calc ---
        try:
            steps = tr["steps"]
            start = datetime.fromisoformat(steps[0]["timestamp"])
            end   = datetime.fromisoformat(steps[-1]["timestamp"])
            m["duration_ms"] = (end - start).total_seconds() * 1000.0
        except Exception:
            m["duration_ms"] = None

        m["agent_count"] = len(
            {s["stage"].replace("agent_start_", "")
             for s in tr.get("steps", [])
             if s["stage"].startswith("agent_start_")}
        )
        return m