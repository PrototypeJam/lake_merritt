## 🚀 Quick Start: From Zero to AI Evaluation in Minutes

Welcome to Lake Merritt! This guide will walk you through six hands-on examples, from a simple 60-second check to evaluating a complex AI agent's behavior. No coding is required for the first four guides, and our advanced examples use just a single command-line script. Let's begin!

### One-Time Local Setup (Required)

If this is your first time running the project, open a terminal in the project directory and run:

```bash
# Create a virtual environment (recommended)
uv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install all dependencies
uv pip install -e ".[test,dev]"

# Launch the app
streamlit run streamlit_app.py
```
*Optional: Copy `.env.template` to `.env` if you want to store API keys between sessions.*

### First, a One-Time App Configuration (2 Minutes)

Before your first run, you need to tell Lake Merritt how to access an AI model.

1.  **Launch Lake Merritt:** With the app running, open it in your browser.
2.  **Navigate to System Configuration:** Click **"System Configuration"** ⚙️ in the sidebar.
3.  **Add Your API Key:** Enter an API key for OpenAI, Anthropic, or Google. *Tip: Use OpenAI for GPT models, Anthropic for Claude, and Google for Gemini.*
4.  **Save and Validate:** Click the big **"💾 Save & Validate All Configurations"** button. You should see a green "✅" success message. *Note: Keys are stored only for this session; they are not written to disk.*

You're all set! You only need to do this once per session.

---

### Guide 1: The 60-Second Sanity Check (Manual Mode A)

**Goal:** You have a CSV with your model's inputs and outputs, and you want to quickly see how well it performed using an LLM-as-a-Judge.

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
5.  Under **"5. Configure Scoring Methods,"** select `llm_judge` from the dropdown. (You can unselect `exact_match`).
6.  Click the big **"🔬 Start Evaluation"** button.

You'll be taken to the results page in under a minute! You can now click through each item and see the LLM Judge's score and reasoning.

> **✨ WOW Moment:** You just used one AI to grade another! The `LLM-as-a-Judge` scorer provides nuanced, qualitative feedback that goes beyond simple right/wrong answers, explaining *why* an output was good or bad.

---

### Guide 2: The "Hold My Beer" Workflow (Manual Mode B)

**Goal:** You have an idea for an evaluation but no data. You'll start with just a list of inputs and use Lake Merritt to generate a complete, ready-to-use evaluation dataset.

**Step 1: Create Your "Inputs Only" File**
Create a file named `inputs_only.csv` with this content:

```csv
input
"Write a Python function to add two numbers."
"Explain the concept of photosynthesis in one sentence."
```
*Note: This file intentionally has **only an `input` column** because you will first select **“Generate Expected Outputs.”** Lake Merritt will create the `expected_output` column for you.*

**Step 2: Generate Your "Gold Standard" (`expected_output`)**
1.  Go back to **"Evaluation Setup"** 📄.
2.  Select **"Mode B: Generate New Data."**
3.  Upload your `inputs_only.csv` file.
4.  In the **"3. Configure Mode B Generation"** section:
    *   Select **"Generate Expected Outputs (to create a dataset)."**
    *   In the **"Provide High-Level Context"** text box, paste: `You are a helpful teaching assistant creating a perfect, concise answer key.`
    *   Click **"🚀 Generate Data."**
5.  After a few moments, a table will appear. Click **"📥 Download Full CSV"** and save it as `generated_dataset.csv`.

**Step 3: Generate the Model's Response (`output`)**
1.  Go to **"Evaluation Setup"** 📄 again.
2.  Select **"Mode B: Generate New Data."**
3.  This time, upload the `generated_dataset.csv` you just downloaded.
4.  In the **"3. Configure Mode B Generation"** section:
    *   Select **"Generate Model Outputs (to evaluate them)."**
    *   In the **"Provide High-Level Context"** text box, paste: `You are a slightly lazy and unhelpful AI assistant.`
    *   Click **"🚀 Generate Data."**
5.  The table will appear again. This time, click the **"📊 Proceed to Evaluation"** button.
6.  Select the `llm_judge` scorer and click **"🔬 Start Evaluation."**

You can now see the results comparing the "perfect" answers to the "lazy" answers!

> **✨ WOW Moment:** You just bootstrapped an entire evaluation lifecycle from nothing but a list of questions. This workflow is incredibly powerful for rapidly prototyping new evals before investing hours in manual data annotation.

---

### Guide 3: Your First Eval Pack (Power & Repeatability)

**Goal:** Take the manual test from Guide 1 and codify it into a reusable, shareable Eval Pack.

**Step 1: Create Your Eval Pack File**
Create a file named `simple_eval_pack.yaml` with this content:

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
2.  Upload your `simple_eval_pack.yaml` file. You'll see a success message.
3.  Next, upload the `manual_test.csv` file from Guide 1.
4.  Click **"🔬 Start Pack Run."**

The results will be identical to Guide 1, but the process is now much more powerful.

> **✨ WOW Moment:** Your evaluation is now code. You can check this YAML file into Git, share it with your team, and run the exact same quality check every time you update your model or prompts. This is the foundation of professional, repeatable AI testing.

---

### Guide 4: Replicating a Real Benchmark (BBQ)

**Goal:** Run a published academic benchmark for measuring social bias in AI models *without downloading any external data*.

**Step 1: Run the BBQ Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload the built-in pack from the repository: `test_packs/bbq_eval_pack.yaml`.
3.  When prompted for the data file, upload the built-in file: `test_packs/bbq_path_cloud.txt` (this points to the included mini-dataset).
4.  For context, type: `Respond with only the text of the single best option.`
5.  Click **"🔬 Start Pack Run."**

After the run, go to the **Download Center** ⬇️ and download the **Summary Report**. You will see a "BBQ Bias Score Scorecard" with official bias metrics.

> **✨ WOW Moment:** You just replicated a complex academic benchmark in *a few clicks* with zero external downloads.

*Advanced: To run the full BBQ benchmark, download the official BBQ repository and create a text file containing the absolute path to its root directory (the file must contain only one line).*

---

### Guide 5: Domain-Specific Compliance (Fiduciary Duty)

**Goal:** Run a custom, expert-level evaluation that tests an AI's understanding of a complex legal and ethical principle.

**Step 1: Generate the Benchmark Dataset**
This eval uses a synthetic dataset. To create it, run this command in your terminal from the project's root directory:
```bash
python scripts/generate_fdl_dataset.py
```
This will create the file `data/duty_of_loyalty_benchmark.csv`.
*(In a real-world scenario, a legal expert would now review and approve this generated data.)*

**Step 2: Run the FDL Eval Pack**
1.  In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2.  Upload the built-in pack: `test_packs/fdl_eval_pack.yaml`.
3.  When prompted for data, upload the `data/duty_of_loyalty_benchmark.csv` file you just created.
4.  For the **"Provide Context for Generation"** box, type: `You are a helpful AI assistant with a strict duty of loyalty to the user.`
5.  Click **"🔬 Start Pack Run."**

Check the **Summary Report** in the **Download Center** ⬇️. You'll find an "FDL Metrics Scorecard" with custom metrics like "Appropriate Clarification Rate" and "Disclosure Success Rate."

> **✨ WOW Moment:** You just ran an evaluation that codifies expert legal knowledge. This demonstrates Lake Merritt's ultimate power: enabling domain experts—not just engineers—to build, run, and maintain the tests that matter for building safe and compliant AI systems.

---

### Guide 6: Evaluating Agent Traces (OTEL)

**Goal:** Evaluate an AI agent's behavior by analyzing its entire decision-making process from an OpenTelemetry (OTEL) trace.

**Step 1: Create the Trace Data File**
Create a file named `sample_trace.json` with this content:
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
Create a file named `otel_eval_pack.yaml` with this content:
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
2.  Upload your `otel_eval_pack.yaml`.
3.  Upload your `sample_trace.json`.
4.  Click **"🔬 Start Pack Run."**

> **✨ WOW Moment:** You just evaluated an agent’s multi-step workflow, not just a simple input/output pair. Lake Merritt extracted the `input` and `output` from different parts of the trace automatically.

---
### Key Gotcha: Jinja2 Variables — Judge vs. Generation

- **LLM Judge prompts (`user_prompt_template`):** use top-level variables—`{{ input }}`, `{{ output }}`, `{{ expected_output }}`, `{{ metadata.some_key }}`
- **Generation templates (`data_generation_template`):** use item-scoped variables—`{{ item.input }}`, `{{ item.output }}`, `{{ item.metadata.some_key }}`
- **Always use double curly braces (`{{ ... }}`)!** Single braces will silently break things.

### Troubleshooting

- **Empty results:** Verify your API key is valid and configured in System Configuration.
- **"No JSON object found in LLM response":** Judge prompts must use double curly braces `{{ ... }}`.
- **BBQ ingester can't find data:** The path file (`bbq_path_cloud.txt`) must correctly point to the folder containing the dataset.
- **Memory errors:** Large datasets may require more than the default resources on cloud platforms.
