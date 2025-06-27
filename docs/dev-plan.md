Unique Document ID: 2025-06-25-E-DevPlan

# **Lake Merritt Eval Packs: Final Development Plan**

## **a) Overview/Introduction**

# Lake Merritt Eval Packs: Extending Evaluation Beyond Code

## Why Eval Packs?

Lake Merritt has proven its value as an open-source AI evaluation workbench. However, creating custom evaluations currently requires forking the codebase or extensive modifications. This limits both adoption and our ability to provide consulting services while maintaining an open-source core.

**Eval Packs solve this by introducing a declarative, configuration-driven approach to evaluations.** Each pack is a YAML file that defines:

*   How to ingest data (CSV, JSON, OpenInference traces)
*   What scorers to run and in what order
*   How to generate custom reports
*   What visualizations to display

This separation enables several critical business and technical goals:

*   **Consulting Revenue**: Deliver custom evaluation logic as billable artifacts without touching core code
*   **Client IP Protection**: Evaluation logic stays in configuration files, not in the open-source codebase
*   **Community Growth**: Share and remix evaluation patterns without programming
*   **Rapid Development**: Create new evaluations in hours, not days

## Development Approach

After evaluating multiple architectural options, we've chosen a phased approach that prioritizes:

1.  **Immediate Value**: Working packs within 2-3 weeks
2.  **Backward Compatibility**: Existing workflows continue unchanged
3.  **Progressive Enhancement**: Start simple, add power over time

**Workflow note (June 2025)**  
  The branch **`eval-pack-architecture`** already exists and was
  created manually by a maintainer with write permission.  
  • Codex will commit and push only to that branch.  
  • After local testing in VS Code, open a PR from  
    `eval-pack-architecture` → `main` for review.  
  • No other branches will be created by Codex.


### Considered Alternatives

We evaluated three main approaches:

*   **UI-Only Configuration**: Too limiting for complex evaluations
*   **Full Plugin System**: Too complex for initial release
*   **Template-Driven Packs**: ✓ Best balance of power and simplicity

The template-driven approach was unanimously recommended across all architectural reviews as the optimal starting point.

## Development Philosophy

*   **Schema First**: Define contracts before code
*   **Test Everything**: Each pack includes test data and expected outputs
*   **Preserve Simplicity**: The "upload CSV and go" flow remains intact
*   **Enable Complexity**: Support multi-stage pipelines and conditional logic when needed

## **b) Full Development Plan**

# Lake Merritt Eval Packs: Development Plan

## Phase 0: Foundation & Schema Definition (Small - 3-5 days)

### Context

Establish the fundamental structure and contracts that all subsequent work will build upon. This phase is critical for ensuring consistency and preventing rework. We will align with the OpenInference standard from day one.

### Tasks

#### 0.1 Project Structure Setup

**Technical Specifications:**

```bash
# Ensure you are working in eval-pack-architecture branch and directories
git switch eval-pack-architecture   # branch was created manually
git pull 
mkdir -p eval_packs/{examples,schemas,templates}
mkdir -p workspaces/{default,_shared}
mkdir -p core/eval_pack
mkdir -p core/ingestion
mkdir -p core/utils
mkdir -p tests/{unit,integration}/eval_packs
mkdir -p docs/eval-packs
touch core/__init__.py
```

**Success Criteria:**

*   Directory structure exists and is documented.
*   `.gitignore` updated to exclude workspace-specific files.
*   Top-level README updated with new directory explanations.
*   A root-level `.gitattributes` file contains `*.pb binary` to keep OpenTelemetry fixtures from polluting diffs and reduce accidental merge conflicts.

#### 0.2 Schema Definition

**Technical Specifications:**
Create `core/eval_pack/schema.py` with Pydantic models. We will include the `SpanKind` enum from OpenInference to enable precise filtering within our pipelines.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

# NEW: Official OpenInference span kinds to be used for filtering.
class SpanKind(str, Enum):
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    TOOL = "TOOL"
    AGENT = "AGENT"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"

class SchemaVersion(str, Enum):
    V1_0 = "1.0"

class IngestionConfig(BaseModel):
    type: str  # e.g., "csv", "json", "openinference", "generic_otel"
    parser: Optional[str] = None  # e.g., "openinference_json", "openinference_proto"
    config: Dict[str, Any] = Field(default_factory=dict)

class PipelineStage(BaseModel):
    name: str
    scorer: str
    config: Dict[str, Any] = Field(default_factory=dict)
    on_fail: str = "continue"  # "continue" or "stop"
    run_if: Optional[str] = None  # Future: conditional execution
    # NEW: Allows a scorer to run only on items representing a specific span kind.
    span_kind: Optional[SpanKind] = None

class ReportingConfig(BaseModel):
    template: Optional[str] = None
    format: str = "markdown"  # "markdown", "html", "pdf"

class EvalPackV1(BaseModel):
    schema_version: SchemaVersion = SchemaVersion.V1_0
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    author: Optional[str] = None
    
    ingestion: IngestionConfig
    pipeline: List[PipelineStage]
    reporting: Optional[ReportingConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

Create formal documentation in `docs/eval-packs/schema-v1.md`.

**Success Criteria:**

*   Valid YAML can be loaded into the `EvalPackV1` Pydantic model.
*   Invalid YAML (e.g., incorrect `span_kind`) raises a clear `ValidationError`.
*   Schema documentation is complete and includes examples.

#### 0.2b Core Data Model Definition

**Technical Specifications:**
Define the core `ScorerResult` in `core/data_models.py`. This model is used throughout the application by the executor and scorers. `EvaluationItem` and `EvaluationResults` are assumed to already exist in this file.

```python
# core/data_models.py
from pydantic import BaseModel, Field
from typing import Any, Dict

class ScorerResult(BaseModel):
    scorer_name: str
    score: Any
    score_type: str = "float"
    numeric_score: float | None = None  # higher = better
    passed: bool
    reasoning: str | None = None
    error: str | None = None
    details: Dict[str, Any] = Field(default_factory=dict)
```

**Success Criteria:**

*   The `ScorerResult` model is available for import in other modules.
*   The model structure supports various scoring outcomes, including errors.

#### 0.3 Component Registry Design

**Technical Specifications:**
Create `core/ingestion/base.py` to define the abstract base class for all ingesters.
```python
# core/ingestion/base.py
from abc import ABC, abstractmethod
from typing import List, Any
from core.data_models import EvaluationItem

class BaseIngester(ABC):
    @abstractmethod
    def ingest(self, data: Any, config: dict) -> List[EvaluationItem]:
        """Parses raw data into a list of evaluation items."""
        pass
```

Update `core/scoring/base.py` to add the `requires_api_key` flag:
```python
# core/scoring/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.data_models import EvaluationItem, ScorerResult

class BaseScorer(ABC):
    """Abstract base class for all scorers."""
    
    requires_api_key: bool = False  # NEW: Flag for API key requirement
    
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def score(self, item: EvaluationItem) -> ScorerResult:
        """Score an evaluation item and return the result."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human friendly name for the scorer."""

    @property
    def description(self) -> str:
        return f"{self.name} scorer"
```

Create `core/registry.py`. This central registry will manage all pluggable components.

```python
# core/registry.py
from typing import Dict, Type
from core.scoring.base import BaseScorer
from core.ingestion.base import BaseIngester

class ComponentRegistry:
    """Central registry for discoverable components"""
    _scorers: Dict[str, Type[BaseScorer]] = {}
    _ingesters: Dict[str, Type[BaseIngester]] = {}
    
    @classmethod
    def register_scorer(cls, name: str, scorer_class: Type[BaseScorer]):
        cls._scorers[name] = scorer_class
    
    @classmethod
    def get_scorer(cls, name: str) -> Type[BaseScorer]:
        if name not in cls._scorers:
            raise ValueError(f"Unknown scorer: {name}")
        return cls._scorers[name]
    
    @classmethod
    def register_ingester(cls, name: str, ingester_class: Type[BaseIngester]):
        cls._ingesters[name] = ingester_class
        
    @classmethod
    def get_ingester(cls, name: str) -> Type[BaseIngester]:
        if name not in cls._ingesters:
            raise ValueError(f"Unknown ingester: {name}")
        return cls._ingesters[name]

    @classmethod
    def discover_builtins(cls):
        """Auto-register all built-in components, tolerating those not yet implemented."""
        # Register scorers
        try:
            from core.scoring import (
                ExactMatchScorer, FuzzyMatchScorer,
                LLMJudgeScorer, CriteriaSelectionJudgeScorer
            )
            cls.register_scorer("exact_match", ExactMatchScorer)
            cls.register_scorer("fuzzy_match", FuzzyMatchScorer)
            cls.register_scorer("llm_judge", LLMJudgeScorer)
            cls.register_scorer("criteria_selection_judge", CriteriaSelectionJudgeScorer)
        except ImportError:
            print("INFO: Core scorers not found, skipping registration.")

        try:
            from core.scoring.tool_usage_scorer import ToolUsageScorer
            cls.register_scorer("tool_usage_scorer", ToolUsageScorer)
        except ImportError:
            print("INFO: ToolUsageScorer not implemented yet, skipping.")

        # Register ingesters
        try:
            from core.ingestion.csv_ingester import CSVIngester
            cls.register_ingester("csv", CSVIngester)
        except ImportError:
            print("INFO: CSVIngester not implemented yet, skipping.")

        try:
            from core.ingestion.json_ingester import JSONIngester
            cls.register_ingester("json", JSONIngester)
        except ImportError:
            print("INFO: JSONIngester not implemented yet, skipping.")

        try:
            from core.ingestion.openinference_ingester import OpenInferenceIngester
            cls.register_ingester("openinference", OpenInferenceIngester)
        except ImportError:
            print("INFO: OpenInferenceIngester not implemented yet, skipping.")

        try:
            from core.ingestion.generic_otel_ingester import GenericOtelIngester
            cls.register_ingester("generic_otel", GenericOtelIngester)
        except ImportError:
            print("INFO: GenericOtelIngester not implemented yet, skipping.")
```
**Success Criteria:**
- `BaseIngester` is defined.
- `BaseScorer` includes the `requires_api_key` flag.
- Registry can store and retrieve both scorers and ingesters.
- Built-in components are auto-discovered upon application startup, gracefully handling missing modules during development.
- Requesting an unknown component name raises a clear `ValueError`.

#### 0.4 Tracing Utility Stub
**Technical Specifications:**
Create `core/utils/tracing.py` to provide a no-op tracing utility. This avoids code churn when real tracing is added later, as call sites will not need to change.

```python
# core/utils/tracing.py

class NoOpSpan:
    def set_attribute(self, key, value):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()

_tracer = NoOpTracer()

def get_tracer(name: str):
    """
    Returns a tracer instance. In this basic implementation, it's always a no-op tracer.
    A real implementation would check for OpenTelemetry SDK configuration.
    """
    return _tracer
```
**Success Criteria:**
- The no-op tracer can be imported and used without causing errors.
- Provides a stable interface for future OpenTelemetry integration.

#### 0.5 Application Bootstrap
**Technical Specifications:**
Create `core/__init__.py` to ensure that the component registry is populated as soon as the `core` package is imported. This is a critical step for making components available to the rest of the application (UI, tests, etc.).
```python
# core/__init__.py
from core.registry import ComponentRegistry

# This ensures that all built-in components are registered and ready
# for use the moment the application starts.
ComponentRegistry.discover_builtins()
```
**Success Criteria:**
- The `discover_builtins` method is called automatically on application startup.
- All existing and future components are available in the registry without needing manual calls in tests or UI code.

## Phase 1: Core Engine Implementation (Medium - 1-2 weeks)

### Context
Build the execution engine that interprets and runs Eval Packs while maintaining full backward compatibility with the existing simple UI workflow.

### Tasks

#### Done- 1.0 Dependencies

**Technical Specifications:**
Add the necessary libraries to `pyproject.toml`. Use `~=` for pre-1.0 libraries to prevent breaking changes. Add `pandas` here because the CSV ingester introduced later depends on it.

```toml
[project.dependencies]
# ... existing dependencies (openai and python-dotenv already present)
openinference-semantic-conventions ~= "0.9"
opentelemetry-proto ~= "1.5"
protobuf ~= "5.0"
pandas ~= "2.2"
```

**IMPLEMENTATION NOTE:** During task 1.0, the version for `openinference-semantic-conventions` had to be adjusted from `~=0.9` to `~=0.1.21` because version 0.9 doesn't exist yet. The current stable version is 0.1.21. The actual dependencies installed are:
- `openinference-semantic-conventions~=0.1.21`
- `opentelemetry-proto>=1.0`
- `protobuf>=4.0`
- `pandas>=2.0.0` (already present)

Also ensure `load_dotenv()` is called in `streamlit_app.py`:
```python
# In streamlit_app.py, near the top after imports
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file for local dev
```

**Success Criteria:**

*   Dependencies are installed correctly and available for import in subsequent tasks.

#### DONE- 1.1 Pack Loader Implementation
**Technical Specifications:**
Create `core/eval_pack/loader.py`. This class is responsible for reading and validating pack files.
```python
import yaml
from pathlib import Path
from typing import Union, List
from core.eval_pack.schema import EvalPackV1
from core.registry import ComponentRegistry

class EvalPackLoader:
    """Loads and validates Eval Pack files"""
    
    def load_from_file(self, path: Union[str, Path]) -> EvalPackV1:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return self.load_from_dict(data)
    
    def load_from_dict(self, data: dict) -> EvalPackV1:
        return EvalPackV1.model_validate(data)
    
    def validate(self, pack: EvalPackV1) -> List[str]:
        """Validate pack configuration"""
        errors = []
        # Check all referenced components exist
        for stage in pack.pipeline:
            try:
                ComponentRegistry.get_scorer(stage.scorer)
            except ValueError as e:
                errors.append(f"Stage '{stage.name}': {str(e)}")
        try:
            ComponentRegistry.get_ingester(pack.ingestion.type)
        except ValueError as e:
            errors.append(f"Ingestion: {str(e)}")
        return errors
```
**Success Criteria:**
- Successfully loads valid YAML files.
- Provides clear errors for invalid files (e.g., non-existent scorer or ingester).
- Validates component references against the ComponentRegistry.

#### DONE- 1.2 Backward Compatibility Layer
**Technical Specifications:**
Create `core/eval_pack/compatibility.py` to ensure the existing UI workflow continues to function seamlessly by translating its inputs into an in-memory Eval Pack.
```python
from typing import List, Dict, Any
from core.eval_pack.schema import EvalPackV1, IngestionConfig, PipelineStage

def create_legacy_pack(
    selected_scorers: List[str],
    scorer_configs: Dict[str, Dict[str, Any]],
    mode: str = "evaluate_existing"
) -> EvalPackV1:
    """Create an in-memory pack from legacy UI selections"""
    
    pipeline = []
    for scorer_name in selected_scorers:
        config = scorer_configs.get(scorer_name, {})
        stage = PipelineStage(
            name=f"{scorer_name}_stage",
            scorer=scorer_name,
            config=config
        )
        pipeline.append(stage)
    
    pack = EvalPackV1(
        name="Legacy UI Evaluation",
        description="Auto-generated from UI selections",
        ingestion=IngestionConfig(type="csv", config={"mode": mode}),
        pipeline=pipeline
    )
    
    return pack
```
Update the main evaluation entry point in `core/evaluation.py`. The key change is that ingestion logic is now handled *before* the executor is called.
```python
from typing import List, Dict, Any, Optional
from core.data_models import EvaluationItem, EvaluationResults
from core.eval_pack.schema import EvalPackV1
from core.registry import ComponentRegistry

async def run_evaluation_batch(
    raw_data: Any,
    selected_scorers: Optional[List[str]] = None,
    scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    pack: Optional[EvalPackV1] = None,
    api_keys: Optional[Dict[str, str]] = None,
    **kwargs
) -> EvaluationResults:
    """Main evaluation entry point with backward compatibility"""
    
    # 1. Determine the pack to use
    if pack is None:
        if selected_scorers is None:
            raise ValueError("Either pack or selected_scorers must be provided")
        from core.eval_pack.compatibility import create_legacy_pack
        pack = create_legacy_pack(selected_scorers, scorer_configs or {})

    # 2. Ingest data based on the pack's configuration
    ingester_cls = ComponentRegistry.get_ingester(pack.ingestion.type)
    ingester = ingester_cls()
    items = ingester.ingest(raw_data, pack.ingestion.config)

    # 3. Execute the pipeline on the prepared items
    from core.eval_pack.executor import PipelineExecutor
    executor = PipelineExecutor()
    return await executor.execute(pack, items, api_keys, **kwargs)
```
**Success Criteria:**
- All existing integration tests pass unchanged.
- The simple UI workflow functions identically to the user.
- The executor's responsibility is now cleanly separated from ingestion.

#### DONE- 1.3 Pipeline Executor
**Technical Specifications:**
Create `core/eval_pack/executor.py`. This class's sole responsibility is to execute a scoring pipeline on an *already ingested* list of evaluation items.
```python
import os
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from core.data_models import EvaluationItem, EvaluationResults, ScorerResult
from core.eval_pack.schema import EvalPackV1
from core.registry import ComponentRegistry
from core.utils.tracing import get_tracer
from openinference_semantic_conventions.trace import SpanAttributes as OI

tracer = get_tracer(__name__)

# Define all sensitive attributes based on the OpenInference spec
SENSITIVE_ATTRIBUTES = {
    "inputs": [OI.INPUT_VALUE, "input.value"],
    "outputs": [OI.OUTPUT_VALUE, "output.value"],
    "input_messages": [OI.LLM_INPUT_MESSAGES, "llm.input_messages"],
    "output_messages": [OI.LLM_OUTPUT_MESSAGES, "llm.output_messages"],
    "input_text": [OI.EMBEDDING_TEXT, "embedding.text"],
    "output_text": [OI.LLM_OUTPUT_MESSAGES, "llm.output_messages"],
    "embedding_vectors": [OI.EMBEDDING_EMBEDDINGS, "embedding.embeddings"],
}

class PipelineExecutor:
    """Executes evaluation pipelines defined in packs"""
    
    def __init__(self):
        self.registry = ComponentRegistry
    
    async def execute(
        self,
        pack: EvalPackV1,
        items: List[EvaluationItem],
        api_keys: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationResults:
        """Execute the pipeline defined in the pack"""
        with tracer.start_as_current_span("eval_pack_execution"):
            # Check for all OpenInference privacy environment variables at runtime
            hide_inputs = os.getenv("OPENINFERENCE_HIDE_INPUTS", "false").lower() == "true"
            hide_outputs = os.getenv("OPENINFERENCE_HIDE_OUTPUTS", "false").lower() == "true"
            hide_input_messages = os.getenv("OPENINFERENCE_HIDE_INPUT_MESSAGES", "false").lower() == "true"
            hide_output_messages = os.getenv("OPENINFERENCE_HIDE_OUTPUT_MESSAGES", "false").lower() == "true"
            hide_input_text = os.getenv("OPENINFERENCE_HIDE_INPUT_TEXT", "false").lower() == "true"
            hide_output_text = os.getenv("OPENINFERENCE_HIDE_OUTPUT_TEXT", "false").lower() == "true"
            hide_embedding_vectors = os.getenv("OPENINFERENCE_HIDE_EMBEDDING_VECTORS", "false").lower() == "true"

            attributes_to_mask = set()
            if hide_inputs: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["inputs"])
            if hide_outputs: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["outputs"])
            if hide_input_messages: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["input_messages"])
            if hide_output_messages: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["output_messages"])
            if hide_input_text: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["input_text"])
            if hide_output_text: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["output_text"])
            if hide_embedding_vectors: attributes_to_mask.update(SENSITIVE_ATTRIBUTES["embedding_vectors"])
            
            if attributes_to_mask:
                for item in items:
                    if "otel_trace" in item.metadata and "spans" in item.metadata["otel_trace"]:
                        for span in item.metadata["otel_trace"]["spans"]:
                            span_attrs = span.get("attributes", {})
                            for attr_to_mask in attributes_to_mask:
                                span_attrs.pop(attr_to_mask, None)

            results = EvaluationResults(
                items=items,
                config=pack.model_dump(),
                metadata={
                    "pack_name": pack.name,
                    "pack_version": pack.version,
                    "execution_time": datetime.now().isoformat()
                }
            )
            
            for stage_idx, stage in enumerate(pack.pipeline):
                scorer_class = self.registry.get_scorer(stage.scorer)
                
                # Securely prepare scorer configuration with API keys
                scorer_config = (
                    stage.config.model_dump(mode="python")  # Pydantic model → dict
                    if hasattr(stage.config, "model_dump")
                    else stage.config.copy()
                )
                
                if api_keys and scorer_class.requires_api_key:
                    provider = scorer_config.get("provider", "openai")
                    if provider in api_keys:
                        scorer_config["api_key"] = api_keys[provider]

                scorer = scorer_class(scorer_config)
                
                for item in items:
                    item_span_kind = item.metadata.get("span_kind")
                    if stage.span_kind and item_span_kind != stage.span_kind.value:
                        continue

                    try:
                        if asyncio.iscoroutinefunction(scorer.score):
                            result = await scorer.score(item)
                        else:
                            result = scorer.score(item)
                        item.scores.append(result)
                    except Exception as e:
                        item.scores.append(ScorerResult(scorer_name=stage.scorer, score="error", score_type="string", passed=False, error=str(e), reasoning=f"Scoring failed: {e}"))
                
                if progress_callback:
                    progress_callback(stage_idx + 1, len(pack.pipeline))
            
            results.calculate_summary_stats()
            return results
```
**Success Criteria:**
- Executes a simple two-stage pipeline successfully.
- Correctly applies expanded privacy masking based on environment variables.
- Securely provides API keys to scorers that require them.
- Respects the `span_kind` filter in pipeline stages.

#### DONE- 1.4 Refactor Built-in Scorers

**Technical Specifications:**
Refactor all existing scorers to return the new `ScorerResult` model defined in Task 0.2b and to set the `requires_api_key` flag where necessary. This is a critical step to prevent `ValidationError` exceptions at runtime. This task involves replacing the old `ScorerResult` implementation with the new one from Task 0.2b in `core/data_models.py` and then updating **every single scorer** in `core/scoring/` to conform to the new model's contract. Below is the representative implementation for `ExactMatchScorer` and `LLMJudgeScorer`. This pattern must be applied to all other built-in scorers.

```python
# In core/scoring/exact_match.py
from core.data_models import EvaluationItem, ScorerResult
from core.scoring.base import BaseScorer

class ExactMatchScorer(BaseScorer):
    @property
    def name(self) -> str: return "Exact Match"
    
    def score(self, item: EvaluationItem) -> ScorerResult:
        if item.output is None:
            return ScorerResult(
                scorer_name="exact_match", score="error", score_type="string",
                passed=False, reasoning="No output provided",
                error="Output field was null or missing."
            )

        is_match = item.output.strip() == item.expected_output.strip()
        numeric_score = 1.0 if is_match else 0.0
        return ScorerResult(
            scorer_name="exact_match", score=numeric_score, score_type="float",
            numeric_score=numeric_score, passed=is_match,
            reasoning="Exact match" if is_match else "No exact match",
            details={"output_len": len(item.output), "expected_len": len(item.expected_output)}
        )

# In core/scoring/llm_judge.py
class LLMJudgeScorer(BaseScorer):
    requires_api_key: bool = True
    # ... rest of the class ...
```

**Success Criteria:**

*   All built-in scorers (`ExactMatchScorer`, `FuzzyMatchScorer`, `LLMJudgeScorer`, etc.) are updated to return the new `ScorerResult` object.
*   All LLM-based scorers have `requires_api_key = True`.
*   The application can run an evaluation using any built-in scorer without runtime validation errors.

#### 1.5 Basic Example Packs & Ingesters

**Technical Specifications:**
Create `eval_packs/examples/basic_csv_eval.yaml`, `eval_packs/examples/generic_otel_eval.yaml`, and **implement the missing CSV and JSON ingesters** so the backward-compatibility layer and manual CSV uploads work.

**CRITICAL REFACTORING NOTE:** This task must also migrate the logic from the old `core/ingestion.py` file into the new `core/ingestion/csv_ingester.py` and `core/ingestion/json_ingester.py` classes. The `core/ingestion.py` file must be deleted as part of this task. Update all legacy imports in `app/pages/2_eval_setup.py` that used the old `load_evaluation_data` function.

**IMPORTANT:** During task 0.3, the `load_evaluation_data` import in `core/__init__.py` (lines 10 and 20) was temporarily commented out to prevent import errors. This import needs to be removed or updated as part of this refactoring task.

```python
# core/ingestion/csv_ingester.py
import pandas as pd
from typing import List, Dict, Any, Union, IO
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class CSVIngester(BaseIngester):
    """Ingests rows from a CSV whose header contains at least: input, expected_output[, output]."""
    REQUIRED = {"input", "expected_output"}

    def ingest(self, data: Union[str, IO, pd.DataFrame], config: Dict) -> List[EvaluationItem]:
        df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data
        missing = self.REQUIRED.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")
        mode = config.get("mode", "evaluate_existing")
        items: List[EvaluationItem] = []
        for idx, row in df.iterrows():
            items.append(
                EvaluationItem(
                    id=str(row.get("id", idx + 1)),
                    input=str(row["input"]),
                    output=str(row.get("output", "")) if mode == "evaluate_existing" else None,
                    expected_output=str(row["expected_output"]),
                    metadata={c: row[c] for c in df.columns if c not in {"id","input","output","expected_output"}}
                )
            )
        return items
```

```python
# core/ingestion/json_ingester.py
import json
from typing import List, Dict, Any, IO, Union
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class JSONIngester(BaseIngester):
    """Accepts a list of objects each containing input / expected_output [ / output ]."""
    def ingest(self, data: Union[str, IO, Dict, List], config: Dict) -> List[EvaluationItem]:
        payload = json.load(data) if hasattr(data, "read") else json.loads(data) if isinstance(data, str) else data
        records = payload if isinstance(payload, list) else payload.get("items", [payload])
        items: List[EvaluationItem] = [
            EvaluationItem(
                id=str(rec.get("id", i + 1)),
                input=rec["input"],
                output=rec.get("output"),
                expected_output=rec["expected_output"],
                metadata=rec.get("metadata", {})
            )
            for i, rec in enumerate(records)
        ]
        return items
```

```python
# core/scoring/tool_usage_scorer.py
from core.scoring.base import BaseScorer
from core.data_models import EvaluationItem, ScorerResult

class ToolUsageScorer(BaseScorer):
    """Scorer that evaluates tool usage patterns in traces."""
    
    @property
    def name(self) -> str: 
        return "Tool Usage Scorer"
    
    def score(self, item: EvaluationItem) -> ScorerResult:
        trace = item.metadata.get("otel_trace", {})
        spans = trace.get("spans", [])
        
        # FIX: Add fallback for generic traces
        tool_spans = [
            s for s in spans 
            if s.get("attributes", {}).get("openinference.span.kind") == "TOOL" or 
            ("tool" in s.get("name", "").lower() and not s.get("attributes", {}).get("openinference.span.kind"))
        ]
        expected_tools = self.config.get("expected_tools", [])
        
        if not expected_tools:
            score = 1.0 if tool_spans else 0.0
            passed = bool(tool_spans)
            reasoning = f"Found {len(tool_spans)} tool usage(s)" if tool_spans else "No tool usage found"
        else:
            used_tools = [s.get("attributes", {}).get("tool.name", "") for s in tool_spans]
            found_tools = [t for t in expected_tools if t in used_tools]
            score = len(found_tools) / len(expected_tools) if expected_tools else 0.0
            passed = score >= 0.5
            reasoning = f"Found {len(found_tools)}/{len(expected_tools)} expected tools: {found_tools}"
        
        return ScorerResult(
            scorer_name="tool_usage_scorer", 
            score=score, 
            score_type="float",
            numeric_score=score,
            passed=passed, 
            reasoning=reasoning,
            details={"tool_spans": len(tool_spans), "used_tools": used_tools if 'used_tools' in locals() else []}
        )
```

```python
# core/ingestion/generic_otel_ingester.py
import json
from typing import List, Any, Dict
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class GenericOtelIngester(BaseIngester):
    """
    Ingests *standard* OTLP traces that do NOT use OpenInference span-kind
    attributes. Groups spans by trace_id and returns one EvaluationItem per trace.
    """
    def ingest(self, data: bytes | str, config: Dict) -> List[EvaluationItem]:
        # 1) Load JSON or Protobuf
        if isinstance(data, bytes):
            rs = ResourceSpans()
            rs.ParseFromString(data)
            scope_lists = getattr(rs, 'scope_spans', []) or getattr(rs, 'instrumentation_library_spans', [])
            spans = [MessageToDict(s) for ils in scope_lists for s in ils.spans]
        else:
            loaded = json.loads(data)
            spans = loaded["spans"] if "spans" in loaded else loaded

        # 2) Group by traceId
        traces: Dict[str, List[Dict]] = {}
        for span in spans:
            tid = span.get("traceId") or span.get("trace_id")
            if tid:
                traces.setdefault(tid, []).append(span)

        # 3) Build evaluation items
        items: List[EvaluationItem] = []
        for trace_id, trace_spans in traces.items():
            root_span = next((s for s in trace_spans if not s.get("parentId")), trace_spans[0])
            input_ = root_span.get("attributes", {}).get("input.value", "")
            out    = root_span.get("attributes", {}).get("output.value", "")
            item = EvaluationItem(
                id=trace_id,
                input=input_,
                expected_output="Varies by trace",
                output=out,
                metadata={"otel_trace": {"spans": trace_spans}}
            )
            root_kind = root_span.get("kind") or root_span.get("attributes", {}).get("openinference.span.kind")
            if root_kind:
                item.metadata["span_kind"] = root_kind
            
            items.append(item)
        return items
```

```yaml
# eval_packs/examples/generic_otel_eval.yaml
schema_version: "1.0"
name: "Plain OTel Trace Evaluation"
version: "1.0"
description: "Checks that every span produced some output"
author: "Lake Merritt Team"

ingestion:
  type: "generic_otel"

pipeline:
  - name: "tool_usage_check"
    scorer: "tool_usage_scorer"
```

**Success Criteria:**

*   All example packs validate successfully against the schema.
*   The `GenericOtelIngester` is fully functional for standard OTLP traces.
*   The `CSVIngester` and `JSONIngester` are fully implemented.
*   A generic OTel trace fixture exists at `fixtures/plain_otel_trace.json`.

#### 1.6 Update UI Binding to New `run_evaluation_batch`

**Technical Specifications:**
Update `app/pages/2_eval_setup.py` so it passes **raw uploaded data** to `run_evaluation_batch` instead of a pre-ingested list of `EvaluationItem` objects. This keeps the UI aligned with the new architecture where ingestion happens inside the evaluation entrypoint.

*In `app/pages/2_eval_setup.py`, find the `st.button("🔬 Start Evaluation")` block and ensure the call is updated:*
```python
# In the button's "if" block:
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# FIX: Define progress widgets before the callback that uses them
progress_bar = st.progress(0.0, "Starting evaluation...")
status_text = st.empty()

results = loop.run_until_complete(
    run_evaluation_batch(
        raw_data=uploaded_file,  # Pass the raw file object from the uploader
        selected_scorers=selected_scorers,
        scorer_configs=scorer_configs,
        api_keys=st.session_state.api_keys,
        progress_callback=lambda i, total: (
            progress_bar.progress(i / total),
            status_text.text(f"Evaluating {i}/{total} items...")
        ),
    )
)
```

**Success Criteria:**

*   Running a CSV-based evaluation through the "manual configuration" UI flow continues to work without error.
*   The application no longer pre-processes the uploaded file into `EvaluationItem` objects in the UI; this logic is now fully delegated to the `core` module.

#### UPDATE on Task 1.6

#### 1.6 Update UI Binding to New `run_evaluation_batch`

> **IMPLEMENTATION NOTE (Discovered during testing):**
>
> The initial refactoring of this page to call the new `run_evaluation_batch` function inadvertently broke the backward-compatible manual workflow. The root cause was that obsolete data validation and ingestion logic (e.g., calls to the deleted `validate_csv_columns`) were not fully removed from the UI page, leading to an `ImportError`.
>
> The final, correct implementation of this task involved a more significant rewrite of `app/pages/2_eval_setup.py` to:
>
> 1.  Completely remove all data validation and ingestion logic from the UI layer.
> 2.  Preserve the full UI for both "Mode A" (Evaluate Existing) and "Mode B" (Generate then Evaluate).
> 3.  For "Mode A", the UI now passes the raw, uploaded file object directly to the `run_evaluation_batch` function.
> 4.  For "Mode B", the UI first ingests the data to run the `generate_outputs` function, and then passes the resulting list of generated items to `run_evaluation_batch`.
>
> This corrected approach successfully restores full backward compatibility for the manual UI while properly delegating responsibilities to the new core engine.
---

## Phase 1a: OpenInference Migration (Small - 5-7 days)

### Context

Validate the architecture with the most complex current use case—OpenInference traces—and provide a reference implementation for advanced evaluations.

### Tasks

#### 1a.1 Register OpenInference Components
**Technical Specifications:**
This step was completed in Phase 0.3 when `discover_builtins` was updated.

**Success Criteria:**
- The `ComponentRegistry` can successfully retrieve the `OpenInferenceIngester`.

#### 1a.2 OpenInference Ingester
**Technical Specifications:**
Create `core/ingestion/openinference_ingester.py`. This ingester MUST group spans by `trace_id` to create one `EvaluationItem` per trace and handle multiple Protobuf formats. **This task must migrate the relevant logic from the existing `core/otel/ingester.py` before deleting the old `core/otel/` directory and its contents.**
```python
# core/ingestion/openinference_ingester.py
import json
from typing import List, Any, Dict
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from openinference_semantic_conventions.trace import SpanAttributes as OI
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from core.ingestion.base import BaseIngester
from core.data_models import EvaluationItem

class OpenInferenceIngester(BaseIngester):
    def ingest(self, data: bytes | str, cfg: Dict) -> List[EvaluationItem]:
        if isinstance(data, bytes):
            # Handle multiple protobuf formats
            try:
                # First, try to parse as the common ResourceSpans message
                rs = ResourceSpans()
                rs.ParseFromString(data)
                scope_lists = getattr(rs, 'scope_spans', []) or getattr(rs, 'instrumentation_library_spans', [])
            except DecodeError:
                # Fallback for gRPC-style ExportTraceServiceRequest envelope
                req = ExportTraceServiceRequest()
                req.ParseFromString(data)
                scope_lists = []
                for resource_spans in req.resource_spans:
                    scope_lists.extend(getattr(resource_spans, 'scope_spans', []) or 
                                     getattr(resource_spans, 'instrumentation_library_spans', []))
            
            spans = [MessageToDict(s) for ils in scope_lists for s in ils.spans]
        else:
            loaded_json = json.loads(data)
            # Handle both a list of spans and a trace-formatted JSON
            if isinstance(loaded_json, list):
                spans = loaded_json
            elif "spans" in loaded_json and isinstance(loaded_json["spans"], list):
                spans = loaded_json["spans"]
            else:
                spans = []

        traces: Dict[str, List[Dict]] = {}
        for span in spans:
            context = span.get("context", {})
            trace_id = context.get("traceId") or context.get("trace_id")
            if trace_id:
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(span)

        evaluation_items = []
        for trace_id, trace_spans in traces.items():
            root_span = next((s for s in trace_spans if not s.get("parentId")), trace_spans[0])
            item = EvaluationItem(
                id=trace_id,
                input=root_span.get("attributes", {}).get(OI.INPUT_VALUE, ""),
                expected_output="Varies by trace",
                output=root_span.get("attributes", {}).get(OI.OUTPUT_VALUE, ""),
                metadata={"otel_trace": {"spans": trace_spans}}
            )
            root_kind = root_span.get("kind") or root_span.get("attributes", {}).get("openinference.span.kind")
            if root_kind:
                item.metadata["span_kind"] = root_kind
            
            evaluation_items.append(item)
        
        return evaluation_items
```
**Success Criteria:**
- The ingester successfully loads both JSON and binary Protobuf trace files.
- It correctly handles both `ResourceSpans` and `ExportTraceServiceRequest` formats.
- It correctly groups spans into a single `EvaluationItem` per trace.

#### 1a.3 Create OTel Eval Pack & Binary Fixture
**Technical Specifications:**
Create `eval_packs/examples/otel_agent_eval.yaml` and create small, compliant `.json` and `.pb` files in `fixtures/` for testing. Ensure the binary fixture file (`.pb`) is committed to the repository.
```yaml
# eval_packs/examples/otel_agent_eval.yaml
schema_version: "1.0"
name: "OpenTelemetry Agent Trace Evaluation"
version: "1.0"
description: "Evaluates AI agent decision-making from OpenInference traces"
author: "Lake Merritt Team"

ingestion:
  type: "openinference"

pipeline:
  - name: "criteria_selection_analysis"
    scorer: "criteria_selection_judge"
    config:
      provider: "openai"
      model: "gpt-4o-mini"
      threshold: 0.7
  - name: "tool_usage_check"
    scorer: "tool_usage_scorer"
    config:
      expected_tools: ["web_search"]
    span_kind: "TOOL" # Example of using the span_kind filter
```
**Success Criteria:**
- Pack loads successfully and references correct components.
- Both a JSON and a binary `.pb` fixture file exist in the repository and are tracked by Git.

#### 1a.4 Trace Ingestion Integration Tests
**Technical Specifications:**
Create `tests/integration/eval_packs/test_trace_packs.py` that tests both Generic OTel and OpenInference ingestion paths, using both JSON and Protobuf formats where applicable.
```python
# tests/integration/eval_packs/test_trace_packs.py
import pytest
from pytest import mark
import core  # This import triggers the bootstrap in core/__init__.py
from core.eval_pack.loader import EvalPackLoader
from core.eval_pack.executor import PipelineExecutor
from core.registry import ComponentRegistry

@mark.asyncio
async def test_generic_otel_pack():
    # ... test logic for generic_otel_eval.yaml and plain_otel_trace.json ...
    loader  = EvalPackLoader()
    pack    = loader.load_from_file("eval_packs/examples/generic_otel_eval.yaml")
    with open("fixtures/plain_otel_trace.json") as f:
        raw = f.read()
    ingester_cls = ComponentRegistry.get_ingester("generic_otel")
    ingester = ingester_cls()
    items    = ingester.ingest(raw, {})
    assert items            # at least one EvaluationItem
    executor = PipelineExecutor()
    results  = await executor.execute(pack, items)
    assert results.items    # scoring ran

@mark.asyncio
async def test_openinference_pack_evaluation_json():
    # ... test logic for JSON fixture ...

@mark.asyncio
async def test_openinference_pack_evaluation_protobuf():
    # ... test logic for .pb fixture ...
```
**Success Criteria:**
- All tests pass, proving the end-to-end architecture is sound for both standard and OpenInference trace formats.

## Phase 2: UI Integration & Workspaces (Medium - 2-3 weeks)

### Context
Expose pack functionality to users and implement workspace isolation for the consulting model.

### Tasks

#### 2.1 Pack Upload UI
**Technical Specifications:**
Update `app/pages/2_eval_setup.py`. Add a `st.radio` button to let users choose between "Upload Eval Pack" and "Configure Manually". If a pack is uploaded, display its details and hide the manual configuration UI. Also include the privacy options.

```python
# In app/pages/2_eval_setup.py
st.header("Evaluation Method")
eval_method = st.radio(
    "Choose evaluation method:",
    ["Upload Eval Pack", "Configure Manually"],
    horizontal=True
)

if eval_method == "Upload Eval Pack":
    uploaded_pack = st.file_uploader("Upload Eval Pack (.yaml)", type=['yaml', 'yml'])
    if uploaded_pack:
        # ... logic to load and display pack info ...
        with st.expander("Privacy Options (for OpenInference traces)"):
            hide_inputs = st.checkbox("Hide inputs from trace attributes")
            hide_outputs = st.checkbox("Hide outputs from trace attributes")
            hide_input_messages = st.checkbox("Hide LLM input messages")
            hide_output_messages = st.checkbox("Hide LLM output messages")
            hide_input_text = st.checkbox("Hide embedding input text")
            hide_output_text = st.checkbox("Hide output text")
            hide_embedding_vectors = st.checkbox("Hide embedding vectors")
            
            # Set environment variables based on selections
            os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true" if hide_inputs else "false"
            os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true" if hide_outputs else "false"
            os.environ["OPENINFERENCE_HIDE_INPUT_MESSAGES"] = "true" if hide_input_messages else "false"
            os.environ["OPENINFERENCE_HIDE_OUTPUT_MESSAGES"] = "true" if hide_output_messages else "false"
            os.environ["OPENINFERENCE_HIDE_INPUT_TEXT"] = "true" if hide_input_text else "false"
            os.environ["OPENINFERENCE_HIDE_OUTPUT_TEXT"] = "true" if hide_output_text else "false"
            os.environ["OPENINFERENCE_HIDE_EMBEDDING_VECTORS"] = "true" if hide_embedding_vectors else "false"
else:
    # Existing manual configuration UI
    # ...
```

**Success Criteria:**
- Users can upload and run evaluations using a pack file.
- All privacy options are exposed in the UI and function correctly.
- Manual configuration still works as before.

#### 2.2 Workspace Isolation
**Technical Specifications:**
Update `streamlit_app.py` to check for the `LAKE_MERRITT_WORKSPACE` environment variable. If set, add the corresponding `workspaces/<name>` directory to `sys.path` so custom Python modules can be imported. The `ComponentRegistry` must be updated to scan this path for custom components.
```python
# In streamlit_app.py
import os
import sys
from pathlib import Path

workspace = os.getenv('LAKE_MERRITT_WORKSPACE', 'default')
workspace_path = Path(f"workspaces/{workspace}")
if workspace_path.exists():
    sys.path.insert(0, str(workspace_path))
    st.session_state["workspace_path"] = workspace_path
    st.session_state.workspace = workspace
    from core.registry import ComponentRegistry
    ComponentRegistry.scan_workspace(workspace_path)

# In core/registry.py
@classmethod
def scan_workspace(cls, workspace_path: Path):
    """Scan workspace for custom components and register them."""
    import importlib.util
    import inspect
    from core.scoring.base import BaseScorer
    from core.ingestion.base import BaseIngester
    
    for py_file in workspace_path.rglob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                # Determine component's registration name
                component_name = getattr(obj, "__component_name__", name).lower()
                
                if issubclass(obj, BaseScorer) and obj is not BaseScorer:
                    cls.register_scorer(component_name, obj)
                if issubclass(obj, BaseIngester) and obj is not BaseIngester:
                    cls.register_ingester(component_name, obj)
        except Exception as e:
            print(f"Warning: Could not load components from {py_file}: {e}")
```
**Success Criteria:**
- Setting the environment variable correctly isolates workspace components.
- A custom scorer placed in a workspace directory can be successfully executed by a pack.

#### 2.3 Reporting Integration
**Technical Specifications:**
Create `core/reporting/jinja_renderer.py` and integrate it into `app/pages/4_downloads.py`. A "Download Custom Report" button will appear if the active pack specifies a reporting template. The renderer should have access to both built-in and workspace-specific template directories.
```python
# In core/reporting/jinja_renderer.py
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import List

class JinjaReportRenderer:
    def __init__(self, template_dirs: List[Path]):
        self.env = Environment(loader=FileSystemLoader(template_dirs))
    
    def render(self, template_name: str, context: dict) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)
        
# In app/pages/4_downloads.py
from datetime import datetime
# Add custom report download if pack specifies template
if st.session_state.get("eval_pack") and st.session_state.eval_pack.reporting and st.session_state.eval_pack.reporting.template:
    st.header("Custom Report")
    
    if st.button("Generate Custom Report"):
        # Logic to determine template search path, including workspace
        template_dirs = [Path("eval_packs/templates"), st.session_state.get("workspace_path", Path("workspaces/default")) / "templates"]
        renderer = JinjaReportRenderer(template_dirs)
        report_content = renderer.render(
            st.session_state.eval_pack.reporting.template,
            {
                "results": st.session_state.eval_results,
                "pack": st.session_state.eval_pack,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "workspace": st.session_state.get("workspace", "default")
                }
            }
        )
        
        st.download_button(
            "Download Custom Report",
            report_content,
            file_name=f"{st.session_state.eval_pack.name}_report.md",
            mime="text/markdown"
        )
```
**Success Criteria:**
- Custom Markdown reports can be generated and downloaded from the UI.
- Templates stored in a workspace directory are correctly found and used.

## Phase 3: Testing & Documentation (Small - 1 week)

### Context
Ensure robustness and usability through comprehensive testing and documentation.

### Tasks

#### 3.1 Test Framework
**Technical Specifications:**
`tests/__init__.py`, `tests/unit/__init__.py`, and `tests/integration/__init__.py` must be created (empty files) so that pytest's default discovery picks up sub-packages on every machine, including CI containers that run with `--pyargs`.

Create `tests/eval_packs/conftest.py` with pytest fixtures to simplify testing of packs.
```python
import pytest
from core.eval_pack.schema import EvalPackV1, IngestionConfig, PipelineStage
from core.evaluation import EvaluationResults

@pytest.fixture
def sample_pack():
    """Provide a simple test pack"""
    return EvalPackV1(
        name="Test Pack",
        version="1.0",
        ingestion=IngestionConfig(type="csv"),
        pipeline=[
            PipelineStage(name="test", scorer="exact_match")
        ]
    )

@pytest.fixture
def pack_test_kit():
    """Pack testing utilities"""
    class PackTestKit:
        def validate_pack(self, pack_path: str) -> bool:
            # Validation logic
            pass
            
        def run_pack(self, pack_path: str, test_data: str) -> EvaluationResults:
            # Test execution logic
            pass
    
    return PackTestKit()
```
**Success Criteria:**
- Test utilities are reusable and simplify writing new pack-based tests.
- CI/CD pipeline is updated to run all new pack-related tests.
- `pytest` passes on a clean clone using only the instructions in this document.

#### 3.2 Documentation
**Technical Specifications:**
Create comprehensive documentation in the `docs/eval-packs/` directory.
1. `docs/eval-packs/quickstart.md` – Getting started guide for users.
2. `docs/eval-packs/authoring.md` – Detailed guide on how to create new packs (and custom components).
3. `docs/eval-packs/components.md` – An auto-generated reference of all built-in scorers and ingesters.
4. `docs/eval-packs/examples.md` – Walkthroughs of the example packs with explanations.
5. Document custom component naming conventions in the authoring guide (e.g., describing the optional `__component_name__` class attribute for custom scorers and ingesters).
6. Add an "OpenInference Compliance" section to the documentation that states which version of the OpenInference specification Lake Merritt v1.1 complies with.

**Success Criteria:**
- Documentation is clear, complete, and covers all features through Phase 2.
- Includes working, copy-pasteable examples.
- Has a clear troubleshooting section for common pack authoring errors.
