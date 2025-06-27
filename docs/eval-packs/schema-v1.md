# Eval Pack Schema v1.0 Documentation

## Overview

The Eval Pack Schema defines the structure and format for evaluation pack configurations in Lake Merritt. These YAML files allow users to declaratively specify data ingestion, scoring pipelines, and reporting configurations without modifying the core application code.

## Schema Structure

### Root Level

```yaml
schema_version: "1.0"
name: "My Evaluation Pack"
version: "1.0"
description: "Optional description of what this pack evaluates"
author: "Author Name"

ingestion:
  # ... ingestion configuration

pipeline:
  # ... list of pipeline stages

reporting:
  # ... optional reporting configuration

metadata:
  # ... optional metadata
```

### Field Descriptions

#### `schema_version` (required)
- Type: `string`
- Allowed values: `"1.0"`
- Specifies the schema version for compatibility checking

#### `name` (required)
- Type: `string`
- A human-readable name for the evaluation pack

#### `version` (optional)
- Type: `string`
- Default: `"1.0"`
- Version of the evaluation pack (not the schema)

#### `description` (optional)
- Type: `string`
- Detailed description of the evaluation pack's purpose

#### `author` (optional)
- Type: `string`
- Name or organization that created the pack

### Ingestion Configuration

The `ingestion` section defines how data should be loaded and parsed.

```yaml
ingestion:
  type: "csv"  # or "json", "openinference", "generic_otel"
  parser: "optional_parser_name"
  config:
    key: value
    # Additional configuration specific to the ingestion type
```

#### Fields:
- `type` (required): The ingestion method to use
- `parser` (optional): Specific parser variant (e.g., "openinference_json", "openinference_proto")
- `config` (optional): Dictionary of configuration options passed to the ingester

### Pipeline Configuration

The `pipeline` is a list of stages that process evaluation items sequentially.

```yaml
pipeline:
  - name: "exact_match_stage"
    scorer: "exact_match"
    config:
      case_sensitive: false
    on_fail: "continue"
    span_kind: "LLM"
```

#### Stage Fields:
- `name` (required): Unique identifier for this stage
- `scorer` (required): The scorer to use (must be registered)
- `config` (optional): Configuration passed to the scorer
- `on_fail` (optional): Action on failure - "continue" (default) or "stop"
- `run_if` (optional): Conditional execution expression (future feature)
- `span_kind` (optional): Filter to run only on specific OpenInference span types

#### Available Span Kinds:
- `CHAIN`: Chain/workflow spans
- `RETRIEVER`: Document retrieval operations
- `RERANKER`: Result reranking operations
- `LLM`: Language model invocations
- `EMBEDDING`: Embedding generation
- `TOOL`: Tool/function calls
- `AGENT`: Agent decision spans
- `GUARDRAIL`: Safety/filtering checks
- `EVALUATOR`: Evaluation operations

### Reporting Configuration

Optional configuration for custom report generation.

```yaml
reporting:
  template: "my_custom_template.jinja2"
  format: "markdown"  # or "html", "pdf"
```

#### Fields:
- `template` (optional): Path to Jinja2 template file
- `format` (optional): Output format (default: "markdown")

### Metadata

Optional dictionary for additional pack information.

```yaml
metadata:
  tags: ["production", "llm-evaluation"]
  created_date: "2024-01-15"
  custom_field: "any value"
```

## Complete Examples

### Example 1: Basic CSV Evaluation

```yaml
schema_version: "1.0"
name: "Basic Accuracy Test"
version: "1.0"
description: "Evaluates model outputs for exact and fuzzy matching"

ingestion:
  type: "csv"
  config:
    mode: "evaluate_existing"

pipeline:
  - name: "exact_match"
    scorer: "exact_match"
    config:
      case_sensitive: false
  
  - name: "fuzzy_match"
    scorer: "fuzzy_match"
    config:
      threshold: 0.8
```

### Example 2: OpenInference Trace Evaluation

```yaml
schema_version: "1.0"
name: "Agent Trace Analysis"
version: "1.0"
description: "Evaluates AI agent behavior from OpenInference traces"
author: "Lake Merritt Team"

ingestion:
  type: "openinference"
  parser: "openinference_proto"

pipeline:
  - name: "llm_quality_check"
    scorer: "llm_judge"
    config:
      provider: "openai"
      model: "gpt-4"
      temperature: 0.1
    span_kind: "LLM"
    
  - name: "tool_usage_validation"
    scorer: "tool_usage_scorer"
    config:
      expected_tools: ["search", "calculator"]
    span_kind: "TOOL"

reporting:
  template: "agent_trace_report.jinja2"
  format: "html"

metadata:
  trace_version: "1.0"
  evaluation_focus: "tool_selection"
```

### Example 3: Multi-Stage Evaluation with Conditional Logic

```yaml
schema_version: "1.0"
name: "Complex Evaluation Pipeline"
version: "2.0"

ingestion:
  type: "json"

pipeline:
  - name: "initial_check"
    scorer: "exact_match"
    on_fail: "stop"  # Stop pipeline if exact match fails
    
  - name: "quality_assessment"
    scorer: "criteria_selection_judge"
    config:
      criteria:
        - "relevance"
        - "completeness"
        - "accuracy"
      threshold: 0.7
    
  - name: "final_validation"
    scorer: "llm_judge"
    config:
      provider: "anthropic"
      model: "claude-3"

metadata:
  pipeline_type: "progressive"
  early_stopping: true
```

## Validation Rules

1. **Schema Version**: Must be a supported version (currently only "1.0")
2. **Required Fields**: `schema_version`, `name`, `ingestion`, and `pipeline` must be present
3. **Pipeline**: Must contain at least one stage
4. **Scorer Names**: All referenced scorers must be registered in the system
5. **Span Kinds**: Must be valid OpenInference span types
6. **Ingestion Types**: Must be supported by the system

## Error Examples

### Invalid Span Kind
```yaml
pipeline:
  - name: "test"
    scorer: "exact_match"
    span_kind: "INVALID_KIND"  # Will raise ValidationError
```

### Missing Required Field
```yaml
schema_version: "1.0"
# Missing 'name' field - will raise ValidationError
ingestion:
  type: "csv"
pipeline:
  - name: "test"
    scorer: "exact_match"
```

### Invalid Schema Version
```yaml
schema_version: "2.0"  # Unsupported version - will raise ValidationError
name: "Test Pack"
# ...
```

## Best Practices

1. **Use Descriptive Names**: Give stages meaningful names that describe their purpose
2. **Document Complex Configs**: Use the description field to explain complex evaluations
3. **Version Your Packs**: Update the version field when modifying packs
4. **Test Incrementally**: Start with simple pipelines and add complexity gradually
5. **Use Span Filtering**: When evaluating traces, use span_kind to target specific operations
6. **Handle Failures Gracefully**: Consider using "continue" for non-critical stages

## Future Extensions

The schema is designed to support future enhancements:
- `run_if` expressions for conditional stage execution
- Additional output formats for reporting
- Custom validation rules
- Pipeline branching and merging