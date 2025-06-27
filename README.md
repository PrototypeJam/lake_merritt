# [Lake Merritt](https://lakemerritt.streamlit.app/): AI Evaluation Workbench

A general-purpose, modular, and extensible platform for custom evaluations of AI models and applications.

**NOTE:** Lake Merritt is currently in the midst of a [major upgrade to support Dev Packs](https://github.com/PrototypeJam/lake_merritt/blob/main/docs/dev-plan.md) as part of a [larger evolution roadmap](https://github.com/PrototypeJam/lake_merritt/wiki/Dev-Plan-for-Eval-Packs-Major-Uplift).

## Overview

Lake Merritt provides a standardized yet flexible environment for evaluating AI systems. With its new **Eval Pack** architecture, you can run everything from quick, simple comparisons using a spreadsheet to complex, multi-stage evaluation pipelines defined in a single configuration file.

The platform is designed for:
- **Rapid Prototyping**: Get feedback on your model with a simple CSV upload and a few clicks.
- **Customizable Evaluation**: Define bespoke evaluation logic using YAML "Eval Packs" to test for specific behaviors, tool usage, and more.
- **Repeatable & Shareable Workflows**: Codify your evaluation strategy in a version-controllable file that can be shared and reused.
- **Deep Analysis**: Analyze results through intuitive visualizations and detailed data exports.

## Getting Started: Two Paths to Evaluation

Lake Merritt now offers two distinct workflows, allowing you to choose the level of complexity that fits your needs.

### Path 1: The Manual Workflow (For Quick Tests)

This is the fastest way to get started. If you have a simple CSV file, you can upload it and configure scorers directly in the user interface. It's perfect for quick checks and initial exploration.

1.  **Prepare Your Data**: Create a CSV file with `input`, `output`, and `expected_output` columns.
2.  **Navigate to "Evaluation Setup"**: Select the **"Configure Manually"** option.
3.  **Upload & Select**: Upload your CSV and choose from a list of built-in scorers (Exact Match, Fuzzy Match, LLM Judge).
4.  **Run**: Click "Start Evaluation" to see your results.

### Path 2: The Eval Pack Workflow (For Power and Repeatability)

This is the new, powerful way to use Lake Merritt. An **Eval Pack** is a YAML file where you declaratively define the entire evaluation process: from data ingestion to a multi-stage scoring pipeline.

This is the recommended path for any serious or recurring evaluation task.

1.  **Create an Eval Pack**: Define your data source, scorers, and configurations in a `.yaml` file.
2.  **Navigate to "Evaluation Setup"**: Select the **"Upload Eval Pack"** option.
3.  **Upload Pack & Data**: Upload your Eval Pack, then upload the corresponding data file (e.g., a CSV, a JSON trace file, etc.).
4.  **Run**: Click "Start Pack Evaluation" to execute your custom workflow.

## Features

### Current Features (v1.0)
- **Dual-Mode UI**: Choose between a simple manual setup or the powerful Eval Pack workflow.
- **Eval Pack Engine**: Define and run custom, multi-stage evaluation pipelines from a single YAML file.
- **Flexible Data Ingestion**: Built-in support for CSVs, JSON, and standard **OpenTelemetry / OpenInference** trace formats. The architecture is extensible for new formats.
- **Rich Scorer Library**:
    - **Exact & Fuzzy Match**: For deterministic and near-match checks.
    - **LLM-as-a-Judge**: Use powerful models (GPT, Claude, Gemini) to score for quality and nuance.
    - **Trace-Specific Scorers**: Evaluate AI agent behavior with scorers like `CriteriaSelectionJudge` and `ToolUsageScorer`.
- **Mode B Generation**: For datasets without outputs, you can configure an "Actor LLM" to generate them before the evaluation pipeline runs.
- **Modular & Extensible**: A registry-based architecture allows for easy addition of custom scorers and ingesters.

### Planned
- Cross-run analysis and comparison
- Live system monitoring via OpenTelemetry integration
- Custom report generation from templates
- UI-driven Eval Pack creation and editing

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/PrototypeJam/lake_merritt.git
    cd lake_merritt
    ```

2.  Create a virtual environment:
    ```bash
    # Using uv (recommended)
    uv venv
    source .venv/bin/activate

    # Using standard venv
    # python -m venv venv
    # source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    # For development (includes testing and linting tools)
    uv pip install -e ".[test,dev]"

    # For standard installation
    # uv pip install .
    ```

4.  Copy `.env.template` to `.env` and add your API keys:
    ```bash
    cp .env.template .env
    ```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
Navigate to the "Evaluation Setup" page to begin.

## Live Web-Based Version

A free deployed version of Lake Merritt is currently available for low-volume live testing on Streamlit Community Cloud (subject to Streamlit's terms and restrictions), at: [https://lakemerritt.streamlit.app](https://lakemerritt.streamlit.app)

## Using Eval Packs

The Eval Pack system is the most powerful feature of Lake Merritt. It turns your evaluation logic into a shareable, version-controllable artifact.

### What is an Eval Pack?

It's a YAML file that defines your entire evaluation. Here is a simple example that ingests a CSV and runs two scorers:

**`basic_csv_eval.yaml`**
```yaml
schema_version: "1.0"
name: "Basic CSV Two-Scorer Test"
version: "1.0"
description: "A simple pack to test the CSV ingester and a multi-stage pipeline."

ingestion:
  type: "csv"
  config:
    mode: "evaluate_existing"

pipeline:
  - name: "strict_check"
    scorer: "exact_match"
    
  - name: "flexible_check"
    scorer: "fuzzy_match"
    config:
      threshold: 0.8
```

### Data Ingestion with Eval Packs

The pack defines how to interpret your data. The `ingestion` block specifies the `type` of ingester to use, such as:
- **`csv`**: For standard CSV files.
- **`json`**: For simple JSON data.
- **`generic_otel`**: For standard OpenTelemetry trace files (JSON or Protobuf).
- **`openinference`**: For traces adhering to the OpenInference semantic conventions.

This makes the system incredibly flexible—to support a new data format, you simply need to add a new ingester and reference it in your pack.

## Architecture

The project follows a modular, extensible architecture designed around Eval Packs.

- `app/`: Streamlit UI layer.
- `core/`: Business logic and evaluation orchestration.
  - **`eval_pack/`**: The core Eval Pack engine, loader, and schema definitions.
  - **`ingestion/`**: Data ingestion modules for various formats (CSV, JSON, OTel, etc.).
  - **`scoring/`**: Implementations for all scorers (`ExactMatch`, `LLMJudge`, etc.).
  - `utils/`: Core utility functions.
- `services/`: External API integrations (e.g., LLM providers).
- **`eval_packs/`**: A directory for storing reusable Eval Pack definitions.
  - `examples/`: Example pack configurations to get you started.
- `workspaces/`: (Coming Soon) Isolated environments for custom packs and components.
- `tests/`: A comprehensive test suite for all components.

## Advanced Use Case: Evaluating OpenTelemetry Traces

Evaluating the complex behavior of AI agents is a primary use case for Lake Merritt. Instead of simple input/output pairs, you can evaluate an agent's entire decision-making process captured in an OpenTelemetry trace.

#### How it Works with Eval Packs

1.  **Capture Traces**: Instrument your AI agent to produce standard OpenTelemetry traces, preferably using the [OpenInference](https://openinference.io/) semantic conventions.
2.  **Create an Eval Pack**: Write a pack that specifies an OTel-compatible ingester (`openinference` or `generic_otel`).
3.  **Define a Pipeline**: Add stages that use trace-aware scorers like `ToolUsageScorer` or `LLMJudgeScorer` to evaluate aspects like:
    - Did the agent use the correct tool?
    - Was the agent's final response consistent with its retrieved context?
    - Did the agent follow its instructions?
4.  **Run in the UI**: Upload your pack and your trace file (e.g., `traces.json`) to run the evaluation.

This workflow provides a powerful, repeatable method for ensuring your AI agents behave as expected.

## Contributing

This project emphasizes deep modularity. When adding new features:
1.  Scorers go in `core/scoring/` and inherit from `BaseScorer`.
2.  Ingesters go in `core/ingestion/` and inherit from `BaseIngester`.
3.  Register new components in `core/registry.py` to make them available to the Eval Pack engine.
4.  All data structures should be defined as Pydantic models in `core/data_models.py`.

## License

MIT License - see LICENSE file for details

## Working With Streamlit Community Cloud

This section documents critical learnings from deploying Lake Merritt to Streamlit Community Cloud, including common issues, their root causes, and proven solutions.

### Key Discovery: Streamlit Uses Poetry, Not Setuptools

**Issue**: "Oh no. Error running app" with logs showing:
```
Error: The current project could not be installed: No file/folder found for package ai-eval-workbench
```

**Root Cause**: Streamlit Community Cloud uses Poetry for dependency management, not setuptools. Even though pyproject.toml specified `build-backend = "setuptools.build_meta"`, Streamlit ignores this and uses Poetry.

**Solution**: Add Poetry configuration to disable package mode:
```toml
[tool.poetry]
package-mode = false
```

This tells Poetry to only install dependencies, not attempt to install the project as a package.

### Python Version Consistency

**Issue**: Deployment failures or unexpected behavior.

**Root Cause**: Mismatch between Python version configured in Streamlit Cloud settings and `runtime.txt` file.

**Solution**: 
1. Check Python version in Streamlit Cloud dashboard (e.g., 3.13)
2. Ensure `runtime.txt` matches exactly:
   ```
   python-3.13
   ```

### Package Discovery Issues

**Issue**: Import errors for new subdirectories (e.g., `ModuleNotFoundError: No module named 'core.otel'`)

**Root Cause**: When using setuptools locally, listing parent packages (e.g., "core") automatically includes subdirectories. However, this may not work reliably in all deployment environments.

**Initial Wrong Approach**: 
```toml
[tool.setuptools]
packages = ["app", "core", "core.otel", "core.scoring", "core.scoring.otel", "services", "utils"]
```
This caused conflicts because subdirectories were listed explicitly when the parent was already included.

**Correct Approach**:
```toml
[tool.setuptools]
packages = ["app", "core", "services", "utils"]
```
List only top-level packages; subdirectories are automatically included.

### Deployment Cache Issues

**Issue**: Changes pushed to GitHub but deployment still shows old errors.

**Symptoms**:
- Logs show timestamps from hours ago
- Error messages reference issues already fixed
- "Updating the app files has failed: exit status 1" repeatedly

**Root Cause**: Streamlit Community Cloud aggressively caches deployments. Once a deployment fails, it may get stuck in a bad state.

**Solutions** (in order of effectiveness):
1. **Nuclear Option - Delete and Redeploy** (Most Reliable):
   - Go to share.streamlit.io
   - Find your app
   - Click three dots menu → Delete
   - Create new app with same settings
   
2. **Force Fresh Pull**:
   - Deploy from a different branch temporarily
   - Then switch back to main
   
3. **Reboot App**:
   - From app page, click "Manage app" 
   - Click three dots → "Reboot app"
   - Note: This may not clear all cache

### Debugging Deployment Failures

**Strategy**: Add temporary debug logging to identify exact failure point.

**Example**:
```python
# In streamlit_app.py
try:
    from core.logging_config import setup_logging
    print("✓ Core imports successful")
except ImportError as e:
    st.error(f"Import error: {e}")
    raise
```

This makes import errors visible in the deployment logs instead of just showing "Oh no."

### Project Structure Requirements

**Critical Files**:
- `pyproject.toml` - Dependencies and project metadata
- `runtime.txt` - Python version specification
- All directories must have `__init__.py` files to be recognized as packages

**Do NOT Use**:
- `requirements.txt` alongside `pyproject.toml` - This can cause conflicts
- Mixed dependency management systems

### Common Error Messages and Solutions

1. **"installer returned a non-zero exit code"**
   - Check pyproject.toml syntax
   - Verify all dependencies can be installed
   - Look for package name conflicts

2. **"Oh no. Error running app"**
   - Check deployment logs for specific errors
   - Verify Python version consistency
   - Ensure Poetry configuration is correct

3. **"This app has gone over its resource limits"**
   - Implement proper caching with TTL
   - Avoid loading large models repeatedly
   - Monitor memory usage in compute-heavy operations

### Best Practices for Streamlit Deployment

1. **Always test locally first** with the exact Python version
2. **Use pyproject.toml exclusively** - don't mix with requirements.txt
3. **Include Poetry configuration** even if using setuptools locally
4. **Monitor deployment logs** immediately after pushing changes
5. **When stuck, delete and redeploy** rather than fighting cache issues
6. **Document non-obvious dependencies** in comments

### Deployment Checklist

Before deploying to Streamlit Community Cloud:
- [ ] Verify `runtime.txt` matches Streamlit's Python version
- [ ] Add `[tool.poetry]` section with `package-mode = false`
- [ ] Ensure all packages have `__init__.py` files
- [ ] Test imports locally
- [ ] Commit and push all changes
- [ ] If updating existing deployment, consider delete/redeploy for clean state

### Getting Help

When deployment fails:
1. Check logs at share.streamlit.io → Your App → Logs
2. Add debug logging to identify exact failure point
3. Search Streamlit forums for similar errors
4. Consider the nuclear option: delete and redeploy

Remember: Streamlit Community Cloud's deployment environment differs from local development. What works locally may need adjustments for cloud deployment.
