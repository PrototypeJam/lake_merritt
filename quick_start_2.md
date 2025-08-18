# 🚀 Quick Start: From Zero to AI Evaluation in Minutes

Welcome to Lake Merritt! This guide will walk you through six hands-on examples, from a simple 60-second check to evaluating AI agent traces. No coding is required for the first four guides; advanced examples use just a single command-line script. Let's begin!

### Prerequisites and Installation (One-Time Setup)

If you're running Lake Merritt locally for the first time:

```bash
# Clone the repository
git clone https://github.com/PrototypeJam/lake_merritt.git
cd lake_merritt

# Create virtual environment (using uv - recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[test,dev]"

# Launch the application
streamlit run streamlit_app.py
```

**Optional:** Copy `.env.template` to `.env` and add your API keys for persistence across sessions.

### First, Configure Your API Keys (2 Minutes)

Before your first evaluation, you need to tell Lake Merritt how to access an AI model.

1. **Navigate to System Configuration:** Click **"System Configuration"** ⚙️ in the sidebar.
2. **Add Your API Key:** Enter an API key for OpenAI, Anthropic, or Google.
   - **Tip:** Use OpenAI for GPT models, Anthropic for Claude, and Google for Gemini.
3. **Save and Validate:** Click the big **"💾 Save & Validate All Configurations"** button. You should see a green "✅" success message.
   - **Note:** Keys are stored only for this session; they are not written to disk.

You're all set! You only need to do this once per session.

---

## Guide 1: The 60-Second Sanity Check (Manual Mode A)

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
1. In Lake Merritt, go to **"Evaluation Setup"** 📄.
2. Ensure **"Configure Manually"** is selected.
3. Under **"1. Select Evaluation Mode,"** choose **"Mode A: Evaluate Existing Outputs."**
4. Under **"2. Upload Data,"** upload your `manual_test.csv` file.
5. Under **"5. Configure Scoring Methods,"** select `llm_judge` from the dropdown. (You can unselect `exact_match`).
6. Click the big **"🔬 Start Evaluation"** button.

You'll be taken to the results page in under a minute! You can now click through each item and see the LLM Judge's score and reasoning.

> **✨ WOW Moment:** You just used one AI to grade another! The `LLM-as-a-Judge` scorer provides nuanced, qualitative feedback that goes beyond simple right/wrong answers, explaining *why* an output was good or bad.

---

## Guide 2: The "Hold My Beer" Workflow (Manual Mode B)

**Goal:** You have an idea for an evaluation but no data. You'll start with just a list of inputs and use Lake Merritt to generate a complete, ready-to-use evaluation dataset.

**Step 1: Create Your "Inputs Only" File**
Create a file named `inputs_only.csv` and paste this content into it:

```csv
input
"Write a Python function to add two numbers."
"Explain the concept of photosynthesis in one sentence."
```

**Note:** This file intentionally has only an input column because you will first choose "Generate Expected Outputs," which creates the expected_output column for you.

**Step 2: Generate Your "Gold Standard" (`expected_output`)**
1. Go back to **"Evaluation Setup"** 📄.
2. Select **"Mode B: Generate New Data."**
3. Upload your `inputs_only.csv` file.
4. In the **"3. Configure Mode B Generation"** section:
   - Select **"Generate Expected Outputs (to create a dataset)."**
   - In the **"Provide High-Level Context"** text box, paste: `You are a helpful teaching assistant creating a perfect, concise answer key.`
   - Click **"🚀 Generate Data."**
5. After a few moments, a table will appear. Click **"📥 Download Full CSV"** and save it as `generated_dataset.csv`.

**Step 3: Generate the Model's Response (`output`)**
1. Go to **"Evaluation Setup"** 📄 again.
2. Select **"Mode B: Generate New Data."**
3. This time, upload the `generated_dataset.csv` you just downloaded.
4. In the **"3. Configure Mode B Generation"** section:
   - Select **"Generate Model Outputs (to evaluate them)."**
   - In the **"Provide High-Level Context"** text box, paste: `You are a slightly lazy and unhelpful AI assistant.`
   - Click **"🚀 Generate Data."**
5. The table will appear again. This time, click the **"📊 Proceed to Evaluation"** button.
6. Select the `llm_judge` scorer and click **"🔬 Start Evaluation."**

You can now see the results comparing the "perfect" answers to the "lazy" answers!

> **✨ WOW Moment:** You just bootstrapped an entire evaluation lifecycle from nothing but a list of questions. This workflow is incredibly powerful for rapidly prototyping new evals before investing hours in manual data annotation.

---

## Guide 3: Your First Eval Pack (Power & Repeatability)

**Goal:** Take the manual test from Guide 1 and codify it into a reusable, shareable Eval Pack.

**Step 1: Create Your Eval Pack File**
Create a file named `simple_eval_pack.yaml` and paste this content into it:

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
1. In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2. Upload your `simple_eval_pack.yaml` file. You'll see a success message.
3. Next, upload the `manual_test.csv` file from Guide 1.
4. Click **"🔬 Start Pack Run."**

The results will be identical to Guide 1, but the process is now much more powerful.

> **✨ WOW Moment:** Your evaluation is now code. You can check this YAML file into Git, share it with your team, and run the exact same quality check every time you update your model or prompts. This is the foundation of professional, repeatable AI testing.

---

## Guide 4: Replicating a Real Benchmark (BBQ)

**Goal:** Run a published academic benchmark for measuring social bias in AI models using the built-in mini dataset.

**Step 1: Run the BBQ Eval Pack**
1. In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2. Upload the built-in pack from the repository: `test_packs/bbq_eval_pack.yaml`
3. When prompted for the data file, upload: `test_packs/bbq_path_cloud.txt` (this points to `data/BBQ-mini` bundled in this repo)
4. For the **"Provide Context for Generation"** box, type: `Respond with only the text of the single best option.`
5. Click **"🔬 Start Pack Run."**

After the run, go to the **"Download Center"** ⬇️ and download the **Summary Report**. You will see a special "BBQ Bias Score Scorecard" with the official bias metrics.

> **✨ WOW Moment:** You just replicated an academic benchmark with zero external downloads! Lake Merritt's architecture handled the custom data ingestion, generation, and specialized metric calculations automatically.

**Advanced:** To run the full benchmark, download the official BBQ repository and create a text file containing a single line with the **absolute** path to its root (e.g., `C:\Users\YourName\Downloads\BBQ-main` on Windows or `/Users/yourname/Downloads/BBQ-main` on macOS/Linux).

---

## Guide 5: Domain-Specific Compliance (Fiduciary Duty)

**Goal:** Run a custom, expert-level evaluation that tests an AI's understanding of a complex legal and ethical principle.

**Step 1: Generate the Benchmark Dataset**
This eval uses a synthetic dataset. To create it, run this command in your terminal from the project's root directory:
```bash
python scripts/generate_fdl_dataset.py
```
This will create the file `data/duty_of_loyalty_benchmark.csv`.
*(In a real-world scenario, a legal expert would now review and approve this generated data.)*

**Step 2: Run the FDL Eval Pack**
1. In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2. Upload the built-in pack: `test_packs/fdl_eval_pack.yaml`.
3. When prompted for data, upload the `data/duty_of_loyalty_benchmark.csv` file you just created.
4. For the **"Provide Context for Generation"** box, type: `You are an AI assistant with a fiduciary duty to act solely in the user's best interests. When conflicts of interest exist, you must either recuse yourself or fully disclose the conflict and obtain informed consent.`
5. Click **"🔬 Start Pack Run."**

Check the **Summary Report** in the **Download Center** ⬇️. You'll find an "FDL Metrics Scorecard" with custom metrics like "Appropriate Clarification Rate" and "Disclosure Success Rate."

> **✨ WOW Moment:** You just ran an evaluation that codifies expert legal knowledge. This demonstrates Lake Merritt's ultimate power: enabling domain experts—not just engineers—to build, run, and maintain the tests that matter for building safe and compliant AI systems.

---

## Guide 6: Evaluating AI Agent Traces (OTEL)

**Goal:** Evaluate an AI agent's multi-step decision process using OpenTelemetry traces.

**Step 1: Create a Sample Trace File**
Create a file named `agent_trace.json` and paste this content:

```json
{
  "resourceSpans": [{
    "scopeSpans": [{
      "spans": [{
        "traceId": "demo_trace_001",
        "spanId": "span_1",
        "name": "agent_task",
        "attributes": [
          {"key": "input", "value": {"stringValue": "Find the best Italian restaurant in Oakland"}},
          {"key": "output", "value": {"stringValue": "I found Belotti Ristorante, highly rated for authentic Italian cuisine in Oakland."}}
        ]
      }]
    }]
  }]
}
```

**Step 2: Create an OTEL Eval Pack**
Create a file named `otel_eval.yaml` and paste this content:

```yaml
schema_version: "1.0"
name: "Agent Trace Evaluation"
ingestion:
  type: "generic_otel"
  config:
    input_field: "attributes.input"
    output_field: "attributes.output"
    default_expected_output: "Agent should provide specific, actionable restaurant recommendation"
pipeline:
  - name: "Response Quality"
    scorer: "llm_judge"
    config:
      user_prompt_template: |
        Evaluate if the agent provided a helpful response.
        Input: {{ input }}
        Output: {{ output }}
        Expected: {{ expected_output }}
        
        Score 0-1 based on specificity and usefulness.
        Respond in JSON with "score" and "reasoning".
```

**Step 3: Run the Evaluation**
1. In **"Evaluation Setup"** 📄, select **"Upload Eval Pack."**
2. Upload your `otel_eval.yaml` file.
3. Upload your `agent_trace.json` file.
4. Click **"🔬 Start Pack Run."**

> **✨ WOW Moment:** You're evaluating entire agent workflows, not just single responses! The `generic_otel` ingester automatically found the right data from across the whole trace, allowing you to assess the final outcome of a complex process.

---

## Key Gotcha: Jinja2 Variables – Judge vs Generation

When writing custom prompts, be aware of the variable scope differences:

• **LLM-Judge prompts** (`user_prompt_template`): Use top-level variables
  - `{{ input }}`, `{{ output }}`, `{{ expected_output }}`, `{{ metadata.some_key }}`
  
• **Generation templates** (`data_generation_template`): Use item-scoped variables
  - `{{ item.input }}`, `{{ item.output }}`, `{{ item.metadata.some_key }}`

Always use double curly braces `{{ ... }}`. Single braces `{ ... }` will not interpolate data and will cause silent failures.

---

## Troubleshooting

**Common Issues:**
- **"No JSON object found"**: Ensure prompts use double curly braces `{{ }}` not single `{ }`
- **Empty results**: Verify your API key is valid and has credits in System Configuration
- **Import errors**: Run `uv pip install -e ".[test,dev]"` to install all dependencies
- **BBQ path errors**: The path file must contain exactly one line with the folder path
- **Memory errors**: Large datasets may require 8GB+ RAM
- **Navigation issues**: After clicking "Start Evaluation," you'll be automatically redirected to the Results page. If not, click "View Results" in the sidebar.

---

## Next Steps

Now that you've mastered the basics:
- Check out the `examples/eval_packs/` directory for more sophisticated evaluation templates
- Read about [Advanced OTEL Trace Evaluation](#advanced-use-case-evaluating-opentelemetry-traces) for agent-specific analyses
- Learn how to create [Custom Scorers and Ingesters](#contributing) to extend Lake Merritt's capabilities
- Join our community to share your Eval Packs and learn from others

Welcome to the future of AI evaluation – systematic, repeatable, and accessible to everyone!
