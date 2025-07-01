# Lake Merritt Eval Pack JSON Ingestion Fix Plan

## Overview of Issues and Approach

### Current Problems
Lake Merritt's Eval Pack feature has two critical bugs preventing non-CSV data formats from being processed:

1. **Silent Pack Validation Failures**: When an Eval Pack fails to load (due to YAML errors, missing components, etc.), the system silently falls back to CSV ingestion without informing the user. This creates confusion when JSON/OTEL data is then parsed as CSV, resulting in cryptic pandas parsing errors.

2. **Hardcoded CSV Ingestion**: Even when an Eval Pack loads successfully and specifies a different ingester (e.g., `generic_otel`), the `run_evaluation_batch` function ignores the pack's ingestion configuration and always uses `CSVIngester`, causing JSON and other formats to fail.

### Fix Strategy
We will implement a three-phase solution:
- **Phase 1**: Surface pack validation errors in the UI to eliminate silent failures
- **Phase 2**: Fix the core ingestion logic to respect the pack's specified ingester
- **Phase 3**: Add comprehensive testing to verify the complete workflow

Each fix will be implemented independently to maintain system stability and allow for incremental deployment.

---

## Enumerated Fix Tasks

### Task 1: Surface Pack Validation Errors in UI

**File**: `app/pages/2_eval_setup.py`

**Location**: Lines 175-183 (in the eval pack upload section)

**Current Code**:
```python
try:
    pack_loader = EvalPackLoader()
    pack_content = uploaded_pack_file.getvalue().decode('utf-8')
    pack_dict = yaml.safe_load(pack_content)
    
    # --- THIS IS THE CORRECTED PART ---
    # Call the unified `load` method and handle the tuple it returns
    pack, validation_errors = pack_loader.load(source=pack_dict)
    
    if validation_errors:
        st.error("Eval Pack validation failed:")
        for error in validation_errors:
            st.code(error, language='text')
        st.stop() # Stop if the pack is invalid
    # --- END OF CORRECTION ---
```

**Required Changes**:
1. Add comprehensive error handling for YAML parsing failures
2. Add error handling for pack loading exceptions
3. Display clear, actionable error messages to users
4. Prevent setting invalid packs in session state

**New Code** (COMPLETE REPLACEMENT for the try-except block):
```python
try:
    pack_loader = EvalPackLoader()
    pack_content = uploaded_pack_file.getvalue().decode('utf-8')
    
    # First validate YAML syntax
    try:
        pack_dict = yaml.safe_load(pack_content)
    except yaml.YAMLError as e:
        st.error("❌ Invalid YAML syntax in Eval Pack:")
        st.code(str(e), language='text')
        st.info("💡 Check for incorrect indentation, missing colons, or invalid characters")
        st.stop()
    
    # Then validate pack structure and components
    pack, validation_errors = pack_loader.load(source=pack_dict)
    
    if validation_errors:
        st.error("❌ Eval Pack validation failed:")
        for error in validation_errors:
            st.error(f"• {error}")
        
        # Provide helpful hints based on common errors
        error_text = " ".join(validation_errors).lower()
        if "unknown ingester" in error_text:
            st.info("💡 Available ingesters: csv, json, generic_otel, otel, openinference")
        elif "unknown scorer" in error_text:
            st.info("💡 Available scorers: exact_match, fuzzy_match, llm_judge, tool_usage, criteria_selection_judge")
        
        st.stop()
    
    # Only set pack in session state if fully valid
    st.session_state.pack = pack
    st.success(f"✅ Loaded and validated Eval Pack: **{pack.name}** (v{pack.version})")
    
    # Show pack details for confirmation
    with st.expander("Pack Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ingestion Type", pack.ingestion.type)
            st.metric("Pipeline Stages", len(pack.pipeline))
        with col2:
            st.metric("Schema Version", pack.schema_version)
            if pack.description:
                st.info(pack.description)
                
except Exception as e:
    st.error(f"❌ Unexpected error loading Eval Pack: {str(e)}")
    st.info("💡 Please check the pack file format and try again")
    logger.exception("Failed to load eval pack")
    st.stop()
```

**Success Criteria**:
- Invalid YAML shows specific syntax errors
- Missing components show which component is missing and available options
- Valid packs show success message with pack details
- Invalid packs never get set in session state
- All error messages are user-friendly and actionable

---

### Task 2: Fix Hardcoded CSV Ingestion in Core (FINAL REVISION)

**File**: `core/evaluation.py`

**Location**: Replace the entire `run_evaluation_batch` function

**New Code** (COMPLETE REPLACEMENT):
```python
async def run_evaluation_batch(
    raw_data: Optional[Union[IO, pd.DataFrame]] = None,
    items: Optional[List[EvaluationItem]] = None,
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pack: Optional[Union[EvalPackV1, Dict[str, Any]]] = None,
    mode: EvaluationMode = EvaluationMode.EVALUATE_EXISTING,
) -> EvaluationResults:
    """
    Ingest data (if needed) and run a pack-based evaluation, preserving the
    legacy scorer path when no pack is supplied.
    """

    # ------------------------------------------------------------------
    # 1. Resolve which evaluation pack we are using **before** ingestion
    # ------------------------------------------------------------------
    eval_pack: EvalPackV1
    if pack is not None:
        if isinstance(pack, dict):
            from core.eval_pack.loader import EvalPackLoader
            loader = EvalPackLoader()
            eval_pack, errors = loader.load(pack)
            if errors:
                logger.error(f"Pack validation errors: {errors}")
                raise ValueError(f"Invalid pack configuration: {errors}")
        else:
            eval_pack = pack
    else:
        # Fallback to legacy one-off pack built from UI scorer choices
        if not selected_scorers:
            raise ValueError("Either 'pack' or 'selected_scorers' must be provided")
        
        eval_pack = create_legacy_pack(
            selected_scorers=selected_scorers,
            scorer_configs=scorer_configs or {},
            api_keys=api_keys or {},
            items=items or [],  # Pass empty list to maintain compatibility
        )
        
        # CRITICAL: Force CSV ingestion for manual mode
        # The legacy UI path is always CSV-based, so we override any guessing
        eval_pack.ingestion.type = "csv"
        eval_pack.ingestion.config = {"mode": mode.value}

    # ------------------------------------------------------------------
    # 2. Ingest raw data if the caller passed a file / DataFrame
    # ------------------------------------------------------------------
    if raw_data is not None:
        logger.info("Ingesting raw data before evaluation")

        ingester_type = eval_pack.ingestion.type
        logger.info(f'Using ingester "{ingester_type}" from pack')

        try:
            ingester_cls = ComponentRegistry.get_ingester(ingester_type)
            ingester = ingester_cls()
        except ValueError as e:
            raise ValueError(
                f"Pack specifies unknown ingester '{ingester_type}'. "
                f"Available ingesters: {list(ComponentRegistry._ingesters.keys())}"
            ) from e

        try:
            ingestion_cfg = eval_pack.ingestion.config or {}
            items = ingester.ingest(raw_data, ingestion_cfg)
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise ValueError(f"Failed to ingest data using {ingester_type}: {e}") from e

        logger.info("Ingested %d items", len(items))

    elif items is None:
        raise ValueError("Either 'raw_data' or 'items' must be provided")

    # ------------------------------------------------------------------
    # 3. Delegate to the existing implementation
    # ------------------------------------------------------------------
    return await _run_evaluation_batch_impl(
        items=items,
        selected_scorers=selected_scorers,
        scorer_configs=scorer_configs,
        api_keys=api_keys,
        batch_size=batch_size,
        progress_callback=progress_callback,
        pack=eval_pack,
    )
```

**Success Criteria**:
- When pack is provided, its ingester type is used
- CSV ingester is only used when no pack is provided (manual mode)
- Clear error messages when ingester is not found
- Clear error messages when ingestion fails
- Ingestion config from pack is properly passed to ingester
---

### Task 3: Add Import for ComponentRegistry

**File**: `core/evaluation.py`

**Location**: Top of file with other imports (around line 10)

**Current Imports Section**:
```python
from core.data_models import (EvaluationItem, EvaluationResults, RunMetadata,
                              ScorerConfig, EvaluationMode)
from core.eval_pack import (EvalPackV1, PipelineExecutor, create_legacy_pack,
                            extract_scorer_configs, extract_selected_scorers)
from core.scoring import create_scorer, get_available_scorers
from core.ingestion import CSVIngester
```

**Required Changes**:
Add ComponentRegistry import to access registered ingesters

**New Code** (ADD this import line):
```python
from core.registry import ComponentRegistry
```

**Complete Updated Imports Section**:
```python
from core.data_models import (EvaluationItem, EvaluationResults, RunMetadata,
                              ScorerConfig, EvaluationMode)
from core.eval_pack import (EvalPackV1, PipelineExecutor, create_legacy_pack,
                            extract_scorer_configs, extract_selected_scorers)
from core.scoring import create_scorer, get_available_scorers
from core.ingestion import CSVIngester
from core.registry import ComponentRegistry
```

**Success Criteria**:
- ComponentRegistry is imported and available for use
- No import errors occur
- Existing imports remain unchanged

---

### Task 4: Comprehensive End-to-End Testing (UPDATED)

**File**: `tests/integration/test_eval_pack_json_ingestion.py` (NEW FILE)

**New Code** (COMPLETE FILE):
```python
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
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": [{
                    "traceId": "5b8aa5a2d2c872e8321cf37308d69df2",
                    "spanId":  "051581bf3cb55c13",
                    "name":    "test-span",
                    "kind":    2,
                    "attributes": {}
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
```

**Success Criteria**:
- All tests pass
- JSON data is successfully ingested with generic_otel pack
- CSV data still works in manual mode
- Invalid packs show clear validation errors
- Ingestion failures show clear error messages
- All expected ingesters are registered

---

### Task 5: Run Complete Test Suite

**Command**: Run from project root directory

**Test Execution**:
```bash
# Run the new integration test specifically
pytest tests/integration/test_eval_pack_json_ingestion.py -v

# Run all tests to ensure no regressions
pytest -v -m "not requires_api"

# If manual testing needed, create test files:
# 1. Save the example pack as test_otel_pack.yaml
# 2. Save the OTEL JSON as test_otel_data.json
# 3. Run: streamlit run streamlit_app.py
# 4. Upload pack, then JSON, verify no parsing errors
```

**Success Criteria**:
- New integration tests pass
- Existing tests continue to pass
- Manual testing shows:
  - Pack validation errors are clearly displayed
  - JSON data uploads successfully with OTEL pack
  - CSV data still works in manual mode
  - No pandas parsing errors occur