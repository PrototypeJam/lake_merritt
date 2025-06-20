import json
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.otel.ingester import OTelTraceIngester
from core.scoring.otel.criteria_selection_judge import \
    CriteriaSelectionJudgeScorer


@pytest.mark.asyncio
async def test_prompt_building(monkeypatch):
    # --- mock LLM client ---
    fake_resp = json.dumps({"score": 0.9, "passed": True, "reasoning": "looks good"})
    monkeypatch.setattr(
        "core.scoring.otel.criteria_selection_judge.create_llm_client",
        lambda provider, key: MagicMock(generate=AsyncMock(return_value=fake_resp)),
    )

    raw = pathlib.Path("fixtures/manual_traces.json").read_text()
    item = OTelTraceIngester().ingest_str(raw)[0]
    scorer = CriteriaSelectionJudgeScorer({"provider": "openai", "api_key": "dummy"})
    result = await scorer.score(item)

    assert result.passed is True
    assert 0.0 <= result.score <= 1.0
