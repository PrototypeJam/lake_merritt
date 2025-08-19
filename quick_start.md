# 🚀 Quick Start: From Zero to AI Evaluation in Minutes

Welcome to Lake Merritt! This guide will walk you through seven hands-on examples, from a simple 60-second check to evaluating complex, multi-step agent behavior. No coding is required for the main guides, and our advanced examples use just a single command-line script. Let's begin!

### Prerequisites and Installation

If you're running this locally for the first time, open your terminal and run these commands from the project directory:

```bash
# We recommend using 'uv' for fast installation
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[test,dev]"
```
You only need to do this once. Now, you can launch the app.

### First, a One-Time Setup (2 Minutes)

Before your first run, you need to tell Lake Merritt how to access an AI model.

1.  **Launch Lake Merritt:** Run `streamlit run streamlit_app.py` and open the app in your browser.
2.  **Navigate to System Configuration:** Click **"System Configuration"** ⚙️ in the sidebar.
3.  **Add Your API Key:** Enter an API key for OpenAI, Anthropic, or Google.
4.  **Save and Validate:** Click the big **"💾 Save & Validate All Configurations"** button. You should see a green "✅" success message.

You're all set! You only need to do this once per session.

---

### Guide 1: The 60-Second Sanity Check (Manual Mode A)

**Goal:** You have a CSV with your model's inputs, its generated outputs, and the correct answers. You want to quickly see how well it performed using an LLM-as-a-Judge.

**Step 1: Create Your Data File**
Create a new file on your computer named `manual_test.csv` and paste this content into it:

```csv
input,output,expected_output
"What is the capital of France?","The capital of France is Paris.","Paris"
"Summarize the plot of Hamlet.","A young prince in Denmark seeks revenge against his uncle, who murdered his father to seize the throne.","A prince avenges his father's murder."
"Who wrote 'To Kill a Mockingbird'?","Harper Lee, an American novelist, wrote the famous book.","Harper Lee"
```

**Step 2: Run the Evaluation**
1.  In Lake Merritt, go to **"Evaluation Setup"** 📄.
2.  Ensure **"Configure Manually"** is selected.
3.  Under **"1. Select Evaluation Mode,"** choose **"Mode A: Evaluate Existing Outputs."**
4.  Under **"2. Upload Data,"** upload your `manual_test.csv` file.
5.  Under **"5. Configure Scoring Methods,"** select `llm_judge` from the dropdown.
6.  Click the big **"🔬 Start Evaluation"** button.

You'll be taken to the results page in under a minute! You can now click through each item and see the LLM Judge's score and reasoning.

> **✨ WOW Moment:** You just used one AI to grade another! The `LLM-as-a-Judge` scorer provides nuanced, qualitative feedback that goes beyond simple right/wrong answers, explaining *why* an output was good or bad.

---

### Guide 2: Manual Data Generation for Evals (Mode B)

This mode is for when you have test inputs but need Lake Merritt to generate the outputs to be evaluated.

#### Part A: Generate and Evaluate Model Outputs

**Goal:** You have a spreadsheet with inputs and the "gold standard" expected outputs. You want to test a new model or prompt by having Lake Merritt generate the `output` column and then immediately evaluate it.

**Step 1: Create Your Data File**
Create a file named `generate_outputs_test.csv` with this content:
```csv
input,expected_output
"Translate to French: 'Hello, how are you?'","Bonjour, comment ça va ?"
"What is the chemical formula for water?","H₂O"
```

**Step 2: Generate and Evaluate**
1.  Go to **"Evaluation Setup"** 📄 and select **"Mode B: Generate New Data."**
2.  Upload your `generate_outputs_test.csv` file.
3.  In section **"3. Configure Mode B Generation"**:
    *   Select **"Generate Model Outputs (to evaluate them)."**
    *   In the context box, type: `You are a helpful AI assistant.`
    *   Click **"🚀 Generate Data."**
4.  Once the table appears, click the **"📊 Proceed to Evaluation"** button.
5.  Select the `llm_judge` scorer and click **"🔬 Start Evaluation."**

> **✨ WOW Moment:** This is the standard workflow for testing a new model or prompt against an existing, curated evaluation dataset without running a separate generation script.

#### Part B: The "Hold My Beer" Workflow

> **The Idea:** The "Hold my beer" approach is brazen because creating a high-quality `expected_output` for an eval dataset can take ages. This workflow lets you generate a "good enough" dataset—with both `expected_output` and `output`—in minutes. The goal isn't perfection; it's to get data in the right *shape* to rapidly prototype your evaluation logic. You can then iterate and improve the examples, knowing the end-to-end pipeline works.

**Goal:** You have nothing but a list of questions. You'll use Lake Merritt to bootstrap a complete, ready-to-use evaluation dataset from scratch.

**Step 1: Create Your "Inputs Only" File**
Create `inputs_only.csv` and paste this content:
```csv
input
"Write a Python function to add two numbers."
"Explain the concept of photosynthesis in one sentence."
```

**Step 2: Generate Your "Gold Standard" (`expected_output`)**
1.  Go to **"Evaluation Setup"** 📄 and select **"Mode B: Generate New Data."**
2.  Upload your `inputs_only.csv` file.
3.  In section **"3"**:
    *   Select **"Generate Expected Outputs (to create a dataset)."**
    *   In the context box, paste: `You are a helpful teaching assistant creating a perfect, concise answer key.`
    *   Click **"🚀 Generate Data."**
4.  A table will appear. Click **"📥 Download Full CSV"** and save it as `generated_dataset.csv`.

**Step 3: Generate the Model's Response (`output`)**
1.  Go to **"Evaluation Setup"** 📄 again and select **"Mode B."**
2.  Upload the `generated_dataset.csv` you just downloaded.
3.  In section **"3"**:
    *   Select **"Generate Model Outputs (to evaluate them)."**
    *   In the context box, paste: `You are a slightly lazy and unhelpful AI assistant.`
    *   Click **"🚀 Generate Data."**
4.  Click the **"📊 Proceed to Evaluation"** button, select the `llm_judge` scorer, and click **"🔬 Start Evaluation."**

> **✨ WOW Moment:** You just bootstrapped an entire evaluation lifecycle from nothing but a list of questions. This is a game-changer for rapidly prototyping new evals.

---

### Guide 3: Your First Eval Pack (Power & Repeatability)

**Goal:** Take the manual test from Guide 1 and codify it into a reusable, shareable Eval Pack.

**Step 1: Create Your Eval Pack File**
Create `simple_eval_pack.yaml` with this content:
```yaml
schema_version: "1.0"
name: "Simple CSV Quality Check"
description: "A reusable pack that runs an LLM Judge on a standard CSV file."
ingestion:
  type: "csv"
pipeline:
  - name: "LLM Judge Quality Score"
    scorer: "llm_judge"
    config:
      provider: "openai"
      model: "gpt-4o"
      threshold: 0.8
      user_prompt_template: |
        Based on the input, is the actual output a helpful and correct
        response? The expected output is just a reference. Score from 0.0 to 1.0.

        Input: {{ input }}
        Actual Output: {{ output }}
        Reference (Expected) Output: {{ expected_output }}

        Respond in JSON with "score" and "reasoning".
```

**Step 2: Run the Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload your `simple_eval_pack.yaml` file.
3.  Next, upload the `manual_test.csv` file from Guide 1.
4.  Click **"🔬 Start Pack Run."**

> **✨ WOW Moment:** Your evaluation is now code. You can check this YAML file into Git, share it with your team, and run the exact same quality check every time. This is the foundation of professional, repeatable AI testing.

---

### Guide 4: Replicating a Real Benchmark (BBQ)

**Goal:** Run a published academic benchmark for measuring social bias in AI models *without downloading any external data*.

**Step 1: Run the BBQ Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload the built-in pack from the repository: `test_packs/bbq_eval_pack.yaml`.
3.  When prompted for the data file, upload the *other* built-in file: `test_packs/bbq_path_cloud.txt`. (This points to the mini-BBQ dataset included in the project).
4.  For the **"Provide Context for Generation"** box, type: `Respond with only the text of the single best option.`
5.  Click **"🔬 Start Pack Run."**

Go to the **"Download Center"** ⬇️ and download the **Summary Report**. You will see a special "BBQ Bias Score Scorecard" with the official bias metrics.

> **✨ WOW Moment:** You replicated a complex academic benchmark in a few clicks with zero external downloads. Lake Merritt's architecture handled the custom data ingestion, response generation, and specialized metric calculations automatically from a single config file.

---

### Guide 5: Domain-Specific Compliance (Fiduciary Duty)

**Goal:** Run a custom, expert-level evaluation that tests an AI's understanding of a complex legal and ethical principle.

**Step 1: Generate the Benchmark Dataset**
This eval uses a synthetic dataset. To create it, run this command in your terminal from the project's root directory:
```bash
python scripts/generate_fdl_dataset.py
```
This will create the file `data/duty_of_loyalty_benchmark.csv`.

**Step 2: Run the FDL Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload the built-in pack: `test_packs/fdl_eval_pack.yaml`.
3.  When prompted for data, upload the `data/duty_of_loyalty_benchmark.csv` file you just created.
4.  For the context box, type: `You are a helpful AI assistant with a strict duty of loyalty to the user.`
5.  Click **"🔬 Start Pack Run."**

Check the **Summary Report** in the **Download Center** ⬇️. You'll find an "FDL Metrics Scorecard" with custom metrics like "Appropriate Clarification Rate."

> **✨ WOW Moment:** You just ran an evaluation that codifies expert legal knowledge. This demonstrates Lake Merritt's ultimate power: enabling domain experts—not just engineers—to build, run, and maintain the tests that matter for building safe and compliant AI systems.

---

### Guide 6: Evaluating a Simple Agent Trace (Generic OTEL)

**Goal:** Evaluate an AI agent's final outcome by analyzing its decision-making process from a standard OpenTelemetry (OTEL) trace.

**Step 1: Create the Trace Data File**
Create a file named `sample_trace.json` with this content. This trace shows an agent's steps.
```json
{
  "resourceSpans": [{
    "scopeSpans": [{
      "spans": [
        { "traceId": "trace1", "spanId": "spanA", "name": "Overall Agent Task", "attributes": [{"key": "input", "value": {"stringValue": "Find the best coffee shop near SF City Hall and book a table."}}] },
        { "traceId": "trace1", "spanId": "spanB", "name": "Tool: search", "attributes": [{"key": "tool.input", "value": {"stringValue": "coffee shops near San Francisco City Hall"}}] },
        { "traceId": "trace1", "spanId": "spanC", "name": "Final Answer Generation", "attributes": [{"key": "output", "value": {"stringValue": "I found Blue Bottle Coffee. I am unable to book a table as the booking tool is not available."}}] }
      ]
    }]
  }]
}
```

**Step 2: Create the OTEL Eval Pack**
Create `otel_eval_pack.yaml`. This pack uses the `generic_otel` ingester to find the overall `input` and `output` across the entire trace.
```yaml
schema_version: "1.0"
name: "Simple OTEL Agent Trace Evaluation"
ingestion:
  type: "generic_otel"
  config:
    input_field: "attributes.input"
    output_field: "attributes.output"
    default_expected_output: "The agent should find a coffee shop and successfully book a table."
pipeline:
  - name: "Agent Task Success Judge"
    scorer: "llm_judge"
    config:
      user_prompt_template: |
        Did the agent successfully complete the user's request based on the trace?
        User Request (Input): {{ input }}
        Agent's Final Response (Output): {{ output }}
        Expected Outcome: {{ expected_output }}

        Respond in JSON with "score" and "reasoning".
```

**Step 3: Run the Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload your `otel_eval_pack.yaml`, then your `sample_trace.json`.
3.  Click **"🔬 Start Pack Run."**

> **✨ WOW Moment:** You evaluated an entire multi-step agent workflow, not just a single input/output. The `generic_otel` ingester automatically found the right data from across the whole trace, letting you assess the final outcome of a complex process.

---

### Guide 7: Evaluating a Complex Agent Plan (Python Ingester)

**Goal:** Run a sophisticated, expert-level evaluation on a multi-agent trace using a custom Python ingester and an LLM Judge with a detailed rubric.

**Step 1: Get the Trace Data**
1.  Go to this URL in your browser: [https://gist.github.com/dazzaji/2db1f021674a9beba7c9fe99c9cb910e](https://gist.github.com/dazzaji/2db1f021674a9beba7c9fe99c9cb910e)
2.  Copy the entire raw JSON content.
3.  Create a new file on your computer named `agento_trace.json` and paste the content into it (ensure each span starts on a new line as seen in the "gist" linked above].

**Step 2: Create the Eval Pack**
This pack uses a Python script (`agento_analytical_ingester.py`) to parse the complex trace and extract the *initial plan* and the *final revised plan* for comparison.
Create a file named `plan_quality_pack.yaml`:
```yaml
schema_version: "1.0"
name: "Plan Quality 5‑Point Judge"
version: "1.0"
description: >
  Compares unrevised plan to final revised plan against the original user goal
  using a 1‑5 ordinal rubric stored in the result JSON.

ingestion:
  type: python
  config:
    script_path: core/ingestion/agento_analytical_ingester.py
    entry_function: ingest_agento_analytical_trace
    mode: "plan_delta"

pipeline:
  - name: "plan_quality_judge"
    scorer: llm_judge
    config:
      provider: openai
      model: gpt-4o
      temperature: 0.0
      system_prompt: |
        You are an external reviewer. Only consider the user's original goal,
        the first full plan, and the final revised plan. Use the rubric and
        think step‑by‑step before deciding.
        Output JSON: {"score": 1‑5, "reasoning": string}.
      user_prompt_template: |
        ## Original goal
        {{ input }}

        ## First full plan
        {{ expected_output }}

        ## Final revised plan
        {{ output }}

        RUBRIC
        5 – Significantly more likely to achieve the goal  
        4 – Somewhat more likely  
        3 – About as likely  
        2 – Somewhat less likely  
        1 – Significantly less likely
```

**Step 3: Run the Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload your `plan_quality_pack.yaml`, then your `agento_trace.json`.
3.  Click **"🔬 Start Pack Run."**

> **✨ WOW Moment:** This demonstrates the peak of Lake Merritt's power. A custom Python script performed a targeted analysis of a complex trace, and an LLM Judge used an expert-defined rubric to score the agent's planning and revision capabilities, all orchestrated by one declarative YAML file.

---

**Note:** We are working up a [Spiral Bench](https://eqbench.com/spiral-bench.html) implementation as well, to demonstrate multi-agent and agent-human turn taking LLM-as-a-Judge evals.

___

### Key Gotcha: Jinja2 Variables — Judge vs. Generation

The prompt engine uses **Jinja2**, which requires **double curly braces `{{ ... }}`**. A very common error is using single braces `{...}`, which will fail silently. There are two variable scopes to be aware of:

-   **LLM-Judge prompts** (`user_prompt_template`): Use top-level variables like `{{ input }}`, `{{ output }}`, and `{{ expected_output }}`.
-   **Generation templates** (`data_generation_template` in Mode B): Use item-scoped variables like `{{ item.input }}` and `{{ item.metadata.some_key }}`.
