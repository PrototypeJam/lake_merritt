# Evaluation Summary Report

Generated on: 2025-07-31 06:17:50

## Overview

- **Total Items Evaluated**: 23
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: 

## Summary Statistics

## Failure Analysis

No failures detected across all scorers.

## Configuration

```json
{
  "eval_pack": {
    "schema_version": "1.0",
    "name": "Timeout Failure Root Cause Analysis",
    "version": "1.0",
    "description": "Finds revision cycles that timed out and evaluates the agent's last attempt to determine the root cause of the failure.\n",
    "author": null,
    "generation": null,
    "ingestion": {
      "type": "python",
      "parser": null,
      "config": {
        "script_path": "core/ingestion/agento_generalized_ingester.py",
        "entry_function": "ingest_agento_trace"
      }
    },
    "pipeline": [
      {
        "name": "analyze_timeout_failure",
        "scorer": "llm_judge",
        "config": {
          "provider": "anthropic",
          "model": "claude-3-5-sonnet-20240620",
          "threshold": 0.5,
          "system_prompt": "You are a senior AI diagnostician. An AI agent failed to complete a revision within the maximum number of attempts. Your task is to analyze the final state to determine the root cause of the failure. Review the revision request, the agent's last attempted draft, and the final critique it received.\n\nRespond ONLY in valid JSON with a \"failure_category\" string and your detailed \"analysis\".\n",
          "user_prompt_template": "### Step Name:\n{{ metadata.step_name }}\n\n### Original Revision Request (What it was asked to do):\n{{ input }}\n\n### Agent's Last Attempted Draft (How far it got):\n{{ output }}\n\n### Final Critique (The last feedback it received):\n{{ expected_output }}\n\n### Diagnostic Task & Categories:\nBased on the provided data, categorize the primary reason for the failure. Select one of the following for `failure_category`:\n\n- \"SUBSTANTIVE_PROGRESS_MADE\": The agent was making good progress and likely would have succeeded with more attempts.\n- \"INSTRUCTION_MISUNDERSTANDING\": The agent consistently failed to grasp the core of the revision request.\n- \"REPETITIVE_LOOP\": The agent was stuck in a repetitive loop, making the same or similar mistakes.\n- \"HALLUCINATION_OR_CONFABULATION\": The agent introduced incorrect or irrelevant information, derailing the revision.\n- \"REFUSAL_OR_INCOMPLETE\": The agent refused to perform the task or produced incomplete, placeholder content.\n- \"CRITIQUE_FAILURE\": The revision request itself was ambiguous, contradictory, or unhelpful, making success impossible.\n\nProvide your detailed `analysis` explaining your choice."
        },
        "on_fail": "continue",
        "run_if": "metadata.get('step_type') == 'timed_out_revision'",
        "span_kind": null
      }
    ],
    "reporting": null,
    "metadata": {}
  },
  "batch_size": 10,
  "privacy_settings": {}
}
```

## Recommendations

- Overall accuracy is below 50%. Consider reviewing the model's training data or prompts.