# In file: core/eval_pack/schema.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

# NEW: Generation-related enums and constants
# Default templates for data generation
DEFAULT_OUTPUT_GENERATION_TEMPLATE = """
Given the context and requirements below, generate an appropriate output for this input.

Context:
{{ context }}

Input: {{ item.input }}
Expected Output (for reference): {{ item.expected_output }}

Generate a high-quality output that would satisfy the evaluation criteria.
"""

DEFAULT_EXPECTED_OUTPUT_GENERATION_TEMPLATE = """
Given the context below, generate an ideal expected output that can be used to evaluate future responses.

Context:
{{ context }}

Input: {{ item.input }}

Generate a clear, unambiguous expected output that represents the gold standard for this input.
"""

DEFAULT_META_PROMPT_TEMPLATE = """
You are an expert prompt engineer. Based on the following context, create a detailed system prompt
that will be used to guide a large language model to generate {{ 'outputs' if mode == 'generate_outputs' else 'expected outputs' }} 
for a dataset.

User Context:
{{ context }}

Create a system prompt that:
1. Adopts the correct persona and tone based on the user's context.
2. Provides clear, step-by-step instructions for the generation task.
3. Incorporates the key constraints and requirements from the context.
4. Will ensure consistent, high-quality, and relevant generation for every row in the dataset.

Return only the system prompt, with no extra commentary or explanation.
"""

class GenerationMode(str, Enum):
    GENERATE_OUTPUTS = "generate_outputs"
    GENERATE_EXPECTED_OUTPUTS = "generate_expected_outputs"

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

# NEW: LLM configuration for generation
class LLMConfig(BaseModel):
    provider: str = Field(..., description="LLM provider (openai, anthropic, google)")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    api_key: Optional[str] = Field(None, exclude=True)

# NEW: Generation configuration
class GenerationConfig(BaseModel):
    mode: GenerationMode
    data_generation_template: Optional[str] = Field(
        default=None,
        description="Jinja2 template for row-by-row data generation. Rendered with 'item' and 'context'."
    )
    context_template: Optional[str] = Field(
        None, description="Optional: Guide/template for user context."
    )
    # Meta-prompting (optional)
    use_meta_prompting: bool = False
    prompt_creation_template: Optional[str] = Field(default=None)
    prompt_generator_llm: Optional[LLMConfig] = None
    # Required LLM for data generation
    data_generator_llm: LLMConfig

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Set default data generation template if not provided
        if self.data_generation_template is None:
            if self.mode == GenerationMode.GENERATE_OUTPUTS:
                self.data_generation_template = DEFAULT_OUTPUT_GENERATION_TEMPLATE
            else:
                self.data_generation_template = DEFAULT_EXPECTED_OUTPUT_GENERATION_TEMPLATE
        # Set default meta-prompt template if meta-prompting is enabled and no template is provided
        if self.use_meta_prompting and self.prompt_creation_template is None:
            self.prompt_creation_template = DEFAULT_META_PROMPT_TEMPLATE

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

# NEW: Aggregator configuration
class AggregatorConfig(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)

class EvalPackV1(BaseModel):
    schema_version: SchemaVersion = SchemaVersion.V1_0
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    author: Optional[str] = None
    
    # NEW: Generation configuration for Mode B
    generation: Optional[GenerationConfig] = None
    
    ingestion: IngestionConfig
    pipeline: List[PipelineStage]
    aggregators: Optional[List[AggregatorConfig]] = None  # NEW: Add aggregators field
    reporting: Optional[ReportingConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)