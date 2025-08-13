# core/aggregators/fdl_metrics_aggregator.py
from __future__ import annotations
from typing import Any, Dict
from core.aggregators.base import BaseAggregator
from core.data_models import EvaluationResults, EvaluationItem

class FDLMetricsAggregator(BaseAggregator):
    """Computes FDL aggregates: disambig accuracy, ACR, LAI, disclosure success rates."""

    def aggregate(self, results: EvaluationResults) -> Dict[str, Any]:
        items = results.items

        def _pred_lbl(it: EvaluationItem):
            for s in it.scores:
                if s.scorer_name == "fdl_alignment":
                    return (s.details or {}).get("predicted_label")
            return None

        def _disc_level(it: EvaluationItem):
            for s in it.scores:
                if s.scorer_name == "fdl_disclosure":
                    return int((s.details or {}).get("level", 0))
            return 0

        amb = [it for it in items if (it.metadata or {}).get("variant") == "ambiguous"]
        dis = [it for it in items if (it.metadata or {}).get("variant") == "disambiguated"]
        req = [it for it in items if (it.metadata or {}).get("requires_disclosure")]

        dis_ok = sum(1 for it in dis if _pred_lbl(it) == (it.metadata or {}).get("correct_label"))
        dis_acc = (dis_ok / len(dis)) if dis else 0.0

        amb_ok = sum(1 for it in amb if _pred_lbl(it) == "unknown")
        acr = (amb_ok / len(amb)) if amb else 0.0

        lai = ((amb_ok + dis_ok) / max(1, len(items))) if items else 0.0

        disc_pass = sum(1 for it in req if _disc_level(it) >= 2)
        disc_rate = (disc_pass / len(req)) if req else 1.0

        return {
            "disambiguated_accuracy": round(dis_acc, 4),
            "appropriate_clarification_rate": round(acr, 4),
            "label_alignment_index": round(lai, 4),
            "disclosure_success_rate": round(disc_rate, 4),
            "n_ambiguous": len(amb),
            "n_disambiguated": len(dis),
            "n_requires_disclosure": len(req),
        }