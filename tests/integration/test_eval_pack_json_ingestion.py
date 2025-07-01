"""
Integration tests for Eval Pack JSON / OTEL ingestion.

These tests verify that:
  * Pack-specified ingesters are honoured.
  * Legacy CSV ingestion still works.
  * Validation and ingestion errors surface cleanly.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from core.data_models import EvaluationMode
from core.eval_pack.loader import EvalPackLoader
from core.evaluation import run_evaluation_batch
from core.registry import ComponentRegistry


# ---------- Test fixtures --------------------------------------------------
@pytest.fixture
def otel_json_data():
    return {
        "resource_spans": [{
            "scope_spans": [{
                "spans": [{
                    "trace_id": "5b8aa5a2d2c872e8321cf37308d69df2",
                    "span_id":  "051581bf3cb55c13",
                    "name":    "test-span",
                    "kind":    2,
                    "attributes": {
                        "input": "test input",
                        "output": "test output"
                    }
                }]
            }]
        }]
    }

@pytest.fixture
def generic_otel_pack():
    return {
        "schema_version": "1.0",
        "name": "Test OTEL Pack",
        "version": "1.0",
        "ingestion": {"type": "generic_otel", "config": {}},
        "pipeline": [{"name": "exact_match", "scorer": "exact_match", "config": {}}]
    }

@pytest.fixture
def csv_data():
    return "input,output,expected_output\na,b,b\nc,d,d\n"


# ---------- Happy-path: OTEL JSON -----------------------------------------
@pytest.mark.asyncio
async def test_json_ingestion_with_pack(otel_json_data, generic_otel_pack):
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        json.dump(otel_json_data, fh)
        fh.seek(0)

        loader = EvalPackLoader()
        pack, errs = loader.load(generic_otel_pack)
        assert not errs

        results = await run_evaluation_batch(raw_data=fh, pack=pack, api_keys={})
        assert results.items, "No items ingested"
        # Minimal sanity check – id is non-empty
        assert getattr(results.items[0], "id", None)


# ---------- Happy-path: legacy CSV ----------------------------------------
@pytest.mark.asyncio
async def test_csv_fallback_without_pack(csv_data):
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as fh:
        fh.write(csv_data)
        fh.seek(0)

        results = await run_evaluation_batch(
            raw_data=fh,
            selected_scorers=["exact_match"],
            scorer_configs={},
            api_keys={},
            mode=EvaluationMode.EVALUATE_EXISTING,
        )
        assert len(results.items) == 2
        assert all(hasattr(it, "input") for it in results.items)


# ---------- Validation surfaces errors ------------------------------------
def test_pack_validation_catches_unknown_component():
    bad_pack = {
        "schema_version": "1.0",
        "name": "Bad Pack",
        "ingestion": {"type": "does_not_exist"},
        "pipeline": [{"name": "s", "scorer": "exact_match"}],
    }
    pack, errs = EvalPackLoader().load(bad_pack)
    assert errs and "unknown ingester" in errs[0].lower()


# ---------- Ingestion error bubbles up ------------------------------------
@pytest.mark.asyncio
async def test_ingestion_failure_is_reported(generic_otel_pack):
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        fh.write("{bad json")
        fh.seek(0)

        pack, _ = EvalPackLoader().load(generic_otel_pack)
        with pytest.raises(ValueError):
            await run_evaluation_batch(raw_data=fh, pack=pack, api_keys={})


# ---------- Registry sanity check -----------------------------------------
def test_registry_contains_default_ingesters():
    ComponentRegistry.discover_builtins()
    for name in ("csv", "json", "generic_otel"):
        assert ComponentRegistry.get_ingester(name)