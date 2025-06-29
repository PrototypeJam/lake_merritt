# OpenTelemetry Ingestion Guide

This guide covers how to ingest OpenTelemetry (OTel) traces into Lake Merritt for evaluation.

## Available OTel Ingesters

Lake Merritt provides multiple ingesters for different OTel trace formats:

### 1. `otel` - Agent Trace Ingester
- **Purpose**: Ingests traces from agent prototypes (like those from agento.py)
- **File format**: JSON traces with specific agent-related spans
- **Extracts**: 
  - `input`: User's goal from the trace
  - `output`: Selected criteria as JSON
  - `metadata`: Additional context like search summaries

### 2. `generic_otel` / `otel_generic` - Generic OTel Ingester
- **Purpose**: Ingests standard OpenTelemetry JSON or protobuf traces
- **File format**: Canonical OTel JSON/protobuf format
- **Use case**: General-purpose OTel trace evaluation

### 3. `openinference` - OpenInference Format
- **Purpose**: Ingests traces using OpenInference semantic conventions
- **File format**: OpenInference-formatted JSON
- **Use case**: ML/AI-specific trace evaluation

## Example Eval Pack for OTel Traces

```yaml
schema_version: "1.0"
name: "My OTel Evaluation"
version: "1.0"
description: "Evaluate OpenTelemetry traces"

ingestion:
  type: "otel"  # or "generic_otel" for standard traces
  config: {}    # Optional configuration

pipeline:
  - name: "my_scorer"
    scorer: "criteria_selection_judge"  # or any other scorer
    config:
      provider: "openai"
      model: "gpt-4o"
      threshold: 0.7
```

## Implications for Pack Authors

After the recent fixes, pack authors benefit from:

### Zero Mental Overhead
- **No "core" vs "optional" confusion**: If a module exists in the repo, it's automatically available
- **Consistent naming**: Use simple names like `csv`, `json`, `otel`, `otel_generic`, or `openinference`
- **No deployment worries**: The `discover_builtins()` pattern ensures all ingesters work across environments

### Clear Trace Format Selection
- `otel` → Best for agent traces with goals and criteria (e.g., manual_traces.json)
- `otel_generic` → Best for standard OTel JSON or binary protobuf
- `openinference` → Best for traces using OpenInference semantic conventions

### Custom Ingester Support
To add a custom ingester:
1. Create a new file under `core/ingestion/`
2. Ensure it subclasses `BaseIngester`
3. Register it in `ComponentRegistry.discover_builtins()`

## Forward-Looking Enhancements

### 1. Self-Registering Plugins
Future versions could support pip-installed packages that expose an entry-point `lake_merritt.ingester`, allowing the registry to discover them automatically at import time.

### 2. Eval-Pack-Scoped Ingesters
The schema could be extended to allow packs to:
- Embed small Python snippets
- Reference Git URLs for custom ingesters
- Load ingesters in sandboxed sub-interpreters at runtime

### 3. In-UI Ingestion Wizard
The Streamlit app (or future desktop app) could:
- Let users drop a file and auto-detect the best ingester
- Preview extracted `EvaluationItem`s before evaluation
- Generate the appropriate YAML fragment automatically

These enhancements would make Lake Merritt more accessible to newcomers while maintaining support for advanced, custom evaluation pipelines.

## Troubleshooting

### "Unknown ingester type" Error
If you see this error, ensure:
1. The ingester type is correctly spelled in your YAML
2. The ingester is registered in `core/registry.py`
3. You're using one of: `csv`, `json`, `otel`, `otel_generic`, `generic_otel`, `openinference`

### Empty Evaluation Items
If ingestion produces no items:
1. Check your trace file format matches the expected format for the ingester
2. For `otel` ingester: Ensure traces contain user goals and criteria
3. For `generic_otel`: Ensure valid OTel JSON/protobuf format

### Custom Fields Not Appearing
The `otel` ingester extracts specific fields. For custom fields:
1. Use `generic_otel` and process in your scorer, or
2. Create a custom ingester for your specific format