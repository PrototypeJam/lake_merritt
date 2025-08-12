# Evaluation Summary Report

Generated on: 2025-07-31 06:29:54

## Overview

- **Total Items Evaluated**: 1
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: LLM Judge

## Summary Statistics

### Llm Judge

- **Accuracy**: 0.0%
- **Items Passed**: 0/1
- **Items Failed**: 1/1
- **Average Score**: 0.000
- **Score Range**: 0.000 - 0.000

## Failure Analysis

### Llm Judge Failures

Total failures: 1

- **Item plan_delta** (Score: 0.000)
  - Reason: The final revised plan is a clear and definite improvement over the initial plan draft in terms of s...

## Configuration

```json
{
  "eval_pack": {
    "schema_version": "1.0",
    "name": "Holistic Plan Improvement Score vs. Goal",
    "version": "1.0",
    "description": "Compares the initial vs. final plan strictly against the user's original goal to determine if the final version is more likely to succeed.\n",
    "author": null,
    "generation": null,
    "ingestion": {
      "type": "python",
      "parser": null,
      "config": {
        "script_path": "core/ingestion/agento_analytical_ingester.py",
        "entry_function": "ingest_agento_analytical_trace",
        "mode": "plan_delta"
      }
    },
    "pipeline": [
      {
        "name": "judge_holistic_plan_improvement",
        "scorer": "llm_judge",
        "config": {
          "provider": "openai",
          "model": "gpt-4o",
          "threshold": 0.75,
          "system_prompt": "You are a discerning executive who cares only about results. You have been given a user's goal, an initial project plan, and a final revised plan. You must ignore all other context. Your entire judgment must be based on one question: Which plan is more likely to fully and correctly achieve the user's stated goal?\n\nYou must return ONLY valid JSON with three fields: `rating` (an integer from 1-5), `rating_meaning` (the corresponding string description from the rubric), and your detailed `reasoning`.\n",
          "user_prompt_template": "### User's Original Goal:\n{{ input }}\n\n### Initial Plan Draft:\n{{ expected_output }}\n\n### Final Revised Plan:\n{{ output }}\n\n### Your Crystal Clear Task:\nUsing ONLY the User's Original Goal as your measure of success, evaluate if the Final Revised Plan is an improvement over the Initial Plan Draft. Use the following 5-point scale for your `rating`:\n\n- **5 (Significantly More Likely):** The final plan is a masterful improvement that is far more likely to achieve the user's goal. It addresses critical strategic flaws or adds essential components missing from the original.\n- **4 (Somewhat More Likely):** The final plan is a clear and definite improvement. Its structure, steps, or focus make it noticeably more likely to succeed.\n- **3 (About as Likely):** The changes are minor, cosmetic, or have little to no material impact on the probability of achieving the goal. Neither plan has a clear advantage.\n- **2 (Somewhat Less Likely):** The revision process was slightly detrimental. The final plan introduced some confusion, removed a good idea, or is less focused on the goal than the original.\n- **1 (Significantly Less Likely):** The final plan is a major regression. It is strategically flawed, incoherent, or has fundamentally misunderstood the user's goal compared to the initial draft.\n\nProvide your `rating`, the corresponding `rating_meaning` string, and your `reasoning`. Do not consider any other information besides the goal and the two plans."
        },
        "on_fail": "continue",
        "run_if": null,
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