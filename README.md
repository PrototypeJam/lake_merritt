# [Lake Merritt](https://lakemerritt.streamlit.app/): AI Evaluation Workbench

A general-purpose, modular, and extensible platform for evaluating AI models and applications, primarily Large Language Models (LLMs).

## Overview

This platform provides a standardized yet flexible environment for:
- Configuring evaluation parameters and models
- Uploading evaluation datasets or generating model outputs
- Applying multiple scoring methods (exact match, fuzzy match, LLM-as-a-Judge)
- Analyzing results through intuitive visualizations
- Comparing performance across different models and configurations

## Features

### Current (v0)
- **Mode A**: Evaluate existing model outputs against expected outputs
- **Mode B**: Generate outputs from an Actor LLM, then evaluate
- Multiple scoring methods with configurable parameters
- Streamlit-based UI with session state management
- Support for CSV data import/export
- Modular architecture for easy extension

### Planned
- Cross-run analysis and comparison
- Live system monitoring via OpenTelemetry
- Enhanced LLM-as-a-Judge configuration
- Prompt versioning and management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-eval-workbench.git
cd ai-eval-workbench
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# For development (includes testing and linting tools):
pip install -e ".[test,dev]"

# For standard installation:
pip install .
```

4. Copy `.env.template` to `.env` and add your API keys:
```bash
cp .env.template .env
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Navigate through the pages:
1. **System Configuration**: Set up API keys and default model parameters
2. **Evaluation Setup**: Upload data, select scoring methods, and run evaluations
3. **View Results**: Analyze evaluation outcomes and detailed scores
4. **Download Center**: Export results in various formats

## Data Format

### CSV Format
For Mode A (evaluate existing outputs), your CSV should have:
- `input`: The prompt/input given to the model
- `output`: The model's actual output
- `expected_output`: The ideal/correct output
- `id` (optional): Unique identifier for each row

For Mode B (generate then evaluate), your CSV needs only:
- `input`: The prompt/input for the model
- `expected_output`: The ideal/correct output

### JSON Format (OpenTelemetry Traces)
Upload JSON files containing OpenTelemetry trace data. The system automatically:
- Extracts user goals and search summaries
- Identifies generated and selected criteria
- Creates evaluation items with metadata for scoring

## Selecting and Configuring Scorers

Lake Merritt provides multiple scoring methods that can be used individually or in combination:

### Available Scorers

1. **Exact Match** - Simple string comparison
   - Basic: Exact string match (with whitespace normalization)
   - Case Insensitive: Ignores case differences
   - Normalized: Handles smart quotes, apostrophes, and optional trailing punctuation

2. **Fuzzy Match** - Flexible string similarity
   - Uses Levenshtein distance for similarity scoring
   - Configurable threshold (default: 0.8)
   - Good for outputs that may have minor variations

3. **LLM Judge** - AI-powered evaluation
   - Uses an LLM to evaluate output quality
   - Configurable prompts and criteria
   - Supports OpenAI, Anthropic, and Google models
   - Provides reasoning for scores

4. **Criteria Selection Judge** - Specialized for OTel traces
   - Evaluates if selected criteria match user goals
   - Analyzes search context and criteria quality
   - Designed specifically for agent trace evaluation

### How to Select Scorers

1. Navigate to **Evaluation Setup**
2. After uploading your data, you'll see the "Select Scoring Methods" section
3. Check the boxes next to the scorers you want to use
4. Click the gear icon ⚙️ next to each scorer to configure settings:
   - **Threshold**: Minimum score to pass (0.0 to 1.0)
   - **Model**: For LLM-based scorers, choose the AI model
   - **Temperature**: Control randomness in LLM scoring
   - **Custom prompts**: For LLM Judge, customize evaluation criteria

### Scorer Recommendations

- **For deterministic outputs**: Use Exact Match or Normalized Exact Match
- **For creative outputs**: Use Fuzzy Match or LLM Judge
- **For multiple valid answers**: Use LLM Judge with custom criteria
- **For agent traces**: Use Criteria Selection Judge

## Architecture

The project follows a modular architecture:
- `app/`: Streamlit UI layer
- `core/`: Business logic and evaluation orchestration
- `services/`: External API integrations (LLM providers)
- `utils/`: Helper utilities

### Evaluating OpenTelemetry Traces

OpenTelemetry (OTel) trace evaluation is a unique feature that analyzes AI agent decision-making:

#### What are OTel Traces?
OpenTelemetry traces capture the execution flow of AI agents, including:
- User goals and inputs
- Search queries and results
- Generated success criteria
- Selected criteria for evaluation
- Timing and metadata for each step

#### How OTel Evaluation Works

1. **Upload**: Go to **Evaluation Setup** and upload a JSON file containing OTel traces
2. **Automatic Processing**: The OTel ingester:
   - Extracts the user's goal from the trace
   - Captures search summaries and context
   - Identifies all generated criteria
   - Records which criteria were selected
   - Preserves timing and agent metadata

3. **Scoring**: The **Criteria Selection Judge** scorer:
   - Analyzes if selected criteria align with the user goal
   - Considers the search context when evaluating
   - Uses an LLM to provide nuanced scoring
   - Returns a score (0-1) with detailed reasoning

4. **Results**: View results includes:
   - Standard scoring metrics
   - Expandable trace timeline showing each step
   - Detailed metadata for debugging
   - Visual indicators for trace quality

#### OTel Trace Format
Your JSON should contain a `traces` array with objects following this structure:
```json
{
  "traces": [{
    "id": "trace_id",
    "steps": [
      {
        "stage": "user_input",
        "outputs": {"user_goal": "..."}
      },
      {
        "stage": "search_complete", 
        "outputs": {"search_summary": "..."}
      },
      {
        "stage": "criteria_generation_complete",
        "outputs": {"generated_criteria": [...]}
      },
      {
        "stage": "criteria_evaluation_complete",
        "outputs": {"selected_criteria": [...]}
      }
    ]
  }]
}
```

This feature is particularly useful for:
- Evaluating AI agent decision quality
- Understanding criteria selection reasoning
- Debugging agent behavior
- Ensuring agents stay aligned with user goals

## Contributing

This project emphasizes deep modularity. When adding new features:
1. Scorers go in `core/scoring/`
2. LLM providers extend `services/llm_clients.py`
3. UI pages go in `app/pages/`
4. All data structures should be defined as Pydantic models in `core/data_models.py`

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
