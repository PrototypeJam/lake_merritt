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