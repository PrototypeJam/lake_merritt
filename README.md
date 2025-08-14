# [Lake Merritt](https://lakemerritt.streamlit.app/): AI Evaluation Workbench

*A general-purpose, modular, and extensible platform for custom evaluations of AI models and applications.*

<video src="/media/Lake_Merritt_HelloWorld.mp4" controls autoplay muted style="max-width:100%;">
  Your browser does not support HTML5 video.
</video>


## Overview

Lake Merritt provides a standardized yet flexible environment for evaluating AI systems. With its **Eval Pack** architecture, you can run everything from quick, simple comparisons using a spreadsheet to complex, multi-stage evaluation pipelines defined in a single configuration file.

The platform is designed for:
- **Rapid Prototyping**: Get feedback on your model with a simple CSV upload and a few clicks.
- **Customizable Evaluation**: Define bespoke evaluation logic using YAML "Eval Packs" to test for specific behaviors, tool usage, and more.
- **Repeatable & Shareable Workflows**: Codify your evaluation strategy in a version-controllable file that can be shared and reused across your team.
- **Deep Analysis**: Analyze results through intuitive visualizations and detailed data exports.

## Understanding the Two Evaluation Modes

Before starting, it's important to understand the two fundamental workflows Lake Merritt supports. Your choice of mode determines what kind of data you need to provide.

### Mode A: Evaluate Existing Outputs
This is the most common use case. You provide a dataset that already contains the model's performance data.
- **Required Data**: `input`, `output`, and `expected_output`.
- **Use Case**: You have already run your model and have its outputs, and you want to score them against a ground truth.

### Mode B: Generate & Evaluate
This mode is for when you have test inputs but need to generate the outputs to be evaluated.
- **Required Data**: `input` and `expected_output` (the `output` column is not needed).
- **Use Case**: You want to test a new prompt or model on a set of inputs. Lake Merritt will first use a configurable "Actor LLM" to generate the `output` for each row, and *then* run the scoring pipeline on the newly generated data.

### Mode B "Hold-My-Beer" Workflow: From Idea to Insight in Minutes

Lake Merritt introduces a uniquely powerful workflow that dramatically accelerates the evaluation process, making it accessible to everyone on your team, regardless of their technical background. This approach allows you to bootstrap an entire evaluation lifecycle starting with nothing more than a list of inputs and a plain-text description of your goals.

This isn't for generating production-grade, statistically perfect evaluation data. Instead, it's an incredibly handy feature for a quick start and rapid iteration, allowing you to see how an entire evaluation would run *before* you invest heavily in manual data annotation.

Here’s how you can go from an idea to a full evaluation run in five steps:

#### **Step 1: Start with Only Inputs and an Idea**

Begin with the bare minimum: a simple CSV file containing only an `input` column. Your "idea" is a natural language explanation of what you're trying to achieve—your success criteria, the persona you want the AI to adopt, and the business, legal, or risk rules it must follow. You can write this directly in the UI or in a markdown file.

#### **Step 2: Generate Your "Gold Standard" (`expected_output`)**

Using **Mode B: Generate New Data**, you'll run the "Generate Expected Outputs" sub-mode. The system will use your context to guide a powerful LLM, which will read each of your inputs and generate a high-quality, correctly formatted `expected_output` for every row. At the end of this step, you can download a brand new dataset, ready for evaluation.

#### **Step 3: Generate the Model's Response (`output`)**

With your new dataset in hand, you immediately run a second Mode B pass. This time, you'll use the "Generate Outputs" sub-mode. You provide context for the model you want to *test* (the "Actor LLM"), and the system generates its `output` for each input, creating a complete three-column CSV.

#### **Step 4: Run a Full, End-to-End Evaluation**

Now, with a complete dataset of synthetically generated data, you can immediately run a **Mode A: Evaluate Existing Outputs** workflow. You can select scorers like `LLM-as-a-Judge` to see how the generated `outputs` stack up against the generated `expected_outputs`, getting a full report with scores and analysis.

#### **Step 5: Iterate with Human Insight**

This is the most crucial step. Having seen a full evaluation lifecycle, you and your team can now intelligently refine the process. You can go back and manually revise the generated `expected_outputs` to better reflect reality-based context, edit the inputs to cover more edge cases, or adjust your initial context to improve the success criteria.

#### Why This is a Game-Changer for AI Evaluation

This workflow is a significant step forward for accessible, open-source AI evaluation tools, offering several unique advantages:

*   **Rapid Prototyping & Iteration**: Go from a concept to a full evaluation baseline in minutes, not days or weeks. This allows you to test hypotheses and iterate on your models and prompts at an unprecedented speed.
*   **Democratizing Evaluation**: This feature is designed for non-technical experts. A product manager, lawyer, or risk officer can directly provide the context that matters in a simple text file, ensuring that the evaluation's success criteria truly support and reflect their domain of authority. It brings essential business, legal, and safety expertise directly into the evaluation setup process.
*   **Evaluate Your Evals First**: Before spending dozens of hours meticulously hand-crafting a "perfect" dataset, you can run a quick, synthetic version through the entire lifecycle. This helps you validate whether your evaluation criteria and prompts are even correct in the first place.
*   **From Zero to Baseline Instantly**: For new projects without existing test data, this workflow instantly generates a starter set of correctly formatted synthetic data, providing a tangible starting point for more rigorous, reality-based annotation later on.

By transforming the tedious task of initial dataset creation into a creative and iterative process, Lake Merritt empowers teams to build better, safer, and more aligned AI systems faster than ever before.

## Getting Started: Two Paths to Evaluation

Lake Merritt offers two UIs for running evaluations, catering to different needs.

### Path 1: The Manual Workflow (For Quick Tests)

This is the fastest way to get started. If you have a simple CSV file, you can upload it and configure scorers directly in the user interface. It's perfect for quick checks and initial exploration.

1.  **Prepare Your Data**: Create a CSV file with the required columns for your chosen mode (e.g., `input`, `output`, `expected_output` for Mode A).
2.  **Navigate to "Evaluation Setup"**: Select the **"Configure Manually"** option.
3.  **Upload & Select**: Upload your CSV and choose from a list of built-in scorers (Exact Match, Fuzzy Match, LLM Judge).
4.  **Run**: Click "Start Evaluation" to see your results.

### Path 2: The Eval Pack Workflow (For Power and Repeatability)

This is the new, powerful way to use Lake Merritt. An **Eval Pack** is a YAML file where you declaratively define the entire evaluation process. This is the recommended path for any serious or recurring evaluation task.

1.  **Create an Eval Pack**: Define your data source, scorers, and configurations in a `.yaml` file.
2.  **Navigate to "Evaluation Setup"**: Select the **"Upload Eval Pack"** option.
3.  **Upload Pack & Data**: Upload your Eval Pack, then upload the corresponding data file (e.g., a CSV, a JSON trace file, etc.).
4.  **Run**: Click "Start Pack Evaluation" to execute your custom workflow.

## Core Features

-   **Dual Evaluation Workflows**:
    -   **Mode A (Evaluate Existing Data)**: Upload a dataset with inputs, outputs, and expected outputs to score pre-existing results.
    -   **Mode B (Generate & Evaluate)**: Provide a dataset with inputs and expected outcomes, and use a configurable "Actor LLM" to generate the outputs before the evaluation pipeline runs.
-   **Powerful Eval Pack Engine**: Define and run custom, multi-stage evaluation pipelines from a single, version-controllable YAML file.
-   **Flexible Data Ingestion**:
    -   **`csv`**: For standard CSV files, supporting both Mode A and Mode B.
    -   **`json`**: For evaluating records in simple JSON or list-of-JSON formats.
    -   **`generic_otel`**: A powerful ingester for standard OpenTelemetry JSON traces, allowing evaluation of complex agent behavior by extracting fields from across an entire trace.
    -   **`python`**: For ultimate flexibility, run a custom Python script to ingest any data format and yield `EvaluationItem` objects.
-   **Rich Scorer Library**:
    -   **Deterministic Scorers**: `exact_match` and `fuzzy_match` for clear, repeatable checks.
    -   **LLM-as-a-Judge**: Use powerful models (GPT, Claude, Gemini) with fully customizable Jinja2-based prompts to score for quality, nuance, and correctness.
    -   **Trace-Aware Scorers**: Evaluate AI agent behavior directly from OTEL traces using scorers like `CriteriaSelectionJudge` and `ToolUsageScorer`.
- **Advanced Generation with Meta-Prompting**: In Mode B, leverage an LLM to generate a sophisticated, context-aware prompt for another LLM, perfect for creating high-quality, structured test data.
-   **Modular & Extensible Architecture**: A registry-based system allows for the easy addition of custom scorers and ingesters to meet any evaluation need.

## Using Eval Packs

The Eval Pack system is the most powerful feature of Lake Merritt. It turns your evaluation logic into a shareable, version-controllable artifact that is essential for rigorous testing.

### Where to Find Existing Eval Packs
A great way to start is by exploring the examples provided in the repository. The `examples/eval_packs/` directory contains ready-to-use packs for a variety of tasks, including:
-   Evaluating legal citation formatting.
-   Assessing AI agent decision quality from OTEL traces.
-   Multi-stage pipelines combining fuzzy matching and LLM judgment.

Use these as templates for your own custom evaluations.

### The Power of Eval Packs: Routine & Ad-Hoc Evals
Eval Packs are designed for both systematic and exploratory analysis.

-   **For Routine & Ongoing Evals**: Codify your team's quality bar in an Eval Pack. Run the same pack against every new model version or prompt update to track regressions and improvements over time. This makes your evaluation process a reliable, repeatable part of your development lifecycle, suitable for integration into CI/CD pipelines.

-   **For Ad-Hoc & Exploratory Evals**: Quickly prototype a new evaluation idea without changing the core application code. Have a novel data format? Write a small Python ingester and reference it in your pack. Want to test a new scoring idea? Define a new pipeline stage with a custom prompt. An Eval Pack lets you experiment rapidly and share the entire evaluation strategy in a single file.

### Creative Configurations & Versioning

Eval Packs enable sophisticated evaluation designs:
-   **Multi-Stage Pipelines**: Combine scorers for efficiency. Use a cheap `exact_match` scorer to filter easy passes, then run an expensive `llm_judge` only on the items that failed the initial check.
-   **Targeted Trace Analysis**: When evaluating an OTEL trace, use different scorers for different parts of the agent's process. Use the `ToolUsageScorer` to validate tool calls and a separate `LLMJudgeScorer` to assess the quality of the agent's final answer.

**Versioning is crucial**. The `version` field in your Eval Pack (e.g., `version: "1.1"`) is more than just metadata. When you change a prompt, adjust a threshold, or add a scorer, you should increment the version. This ensures that you can always reproduce past results and clearly track how your evaluation standards evolve over time.

## Advanced Use Case: Evaluating OpenTelemetry Traces

Evaluating the complex behavior of AI agents is a primary use case for Lake Merritt. Instead of simple input/output pairs, you can evaluate an agent's entire decision-making process captured in an OpenTelemetry trace.

#### How it Works with Eval Packs

1.  **Capture Traces**: Instrument your AI agent to produce standard OpenTelemetry traces, preferably using the [OpenInference](https://openinference.io/) semantic conventions.
2.  **Create an Eval Pack**: Write a pack that specifies an OTel-compatible ingester (`generic_otel` or `python`).
3.  **Define a Pipeline**: Add stages that use trace-aware scorers like `ToolUsageScorer` or `LLMJudge` to evaluate aspects like:
    - Did the agent use the correct tool?
    - Was the agent's final response consistent with its retrieved context?
    - Did the agent follow its instructions?
4.  **Run in the UI**: Upload your pack and your trace file (e.g., `traces.json`) to run the evaluation.

## Key Gotcha: Jinja2 Prompts for LLM Judge

A common source of errors, especially in the manual workflow, is an incorrectly formatted `LLM-as-a-Judge` prompt.

-   **Problem**: The LLM Judge scorer uses the **Jinja2** templating engine to insert data into your prompt. Jinja2 requires variables to be enclosed in **double curly braces**.
-   **Symptom**: If your prompt uses single braces (e.g., `{input}`), the LLM will not see your data and will return an error message instead of a JSON score. This will cause the item to be marked as "failed to score."
-   **Solution**: Always use double curly braces for placeholders in your prompts, like `{{ input }}`, `{{ output }}`, and `{{ expected_output }}`. The default prompt in the manual UI has been corrected, but be sure to use this syntax if you write a custom prompt.

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
    ```

3.  Install dependencies:
    ```bash
    # This installs the app, Streamlit, and all core dependencies
    # including python-dotenv for local .env file handling
    uv pip install -e ".[test,dev]"
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

## Contributing

This project emphasizes deep modularity. When adding new features:
1.  Scorers go in `core/scoring/` and inherit from `BaseScorer`.
2.  Ingesters go in `core/ingestion/` and inherit from `BaseIngester`.
3.  Register new components in `core/registry.py` to make them available to the Eval Pack engine.
4.  All data structures should be defined as Pydantic models in `core/data_models.py`.

## License

MIT License - see LICENSE file for details.

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

_______________________

# Planned

Highlights:

- Cross-run analysis and comparison
- Live system monitoring via OpenTelemetry integration
- Custom report generation from templates
- UI-driven Eval Pack creation and editing

Deeper Dives into Roadmap Items:

* [IMPORTANT ENHANCEMENTS (Feature Backlog)](https://github.com/PrototypeJam/lake_merritt/issues/38)
* [NON-URGENT FIXES [Backlog for Future Sprints](https://github.com/PrototypeJam/lake_merritt/issues/37)
* Also see: [Add Backend Database](https://github.com/PrototypeJam/lake_merritt/issues/45), [Custom Scorers Without Adding to Registry](https://github.com/PrototypeJam/lake_merritt/issues/59), [Less awful frontend](https://github.com/PrototypeJam/lake_merritt/issues/64), and [Concept for refactoring OpenAI Evals (via data, config, and sometimes grading logic) into simple Eval Packs](https://github.com/PrototypeJam/lake_merritt/issues/63)
