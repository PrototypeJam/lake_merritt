# Evaluation Summary Report

Generated on: 2025-07-31 06:26:10

## Overview

- **Total Items Evaluated**: 1
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: LLM Judge

## Summary Statistics

### Llm Judge

- **Accuracy**: 100.0%
- **Items Passed**: 1/1
- **Items Failed**: 0/1
- **Average Score**: 1.000
- **Score Range**: 1.000 - 1.000

## Failure Analysis

No failures detected across all scorers.

## Configuration

```json
{
  "eval_pack": {
    "schema_version": "1.0",
    "name": "Plan Quality 5\u2011Point Judge",
    "version": "1.0",
    "description": "Compares unrevised plan to final revised plan against the original user goal using a 1\u20115 ordinal rubric stored in the result JSON.\n",
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
        "name": "plan_quality_judge",
        "scorer": "llm_judge",
        "config": {
          "provider": "openai",
          "model": "gpt-4o",
          "temperature": 0.0,
          "system_prompt": "You are an external reviewer. Only consider the user's original goal,\nthe first full plan, and the final revised plan. Use the rubric and\nthink step\u2011by\u2011step before deciding.\nOutput JSON: {\"score\": 1\u20115, \"reasoning\": string}.\n",
          "user_prompt_template": "## Original goal\n{{ input }}\n\n## First full plan\n{{ expected_output }}\n\n## Final revised plan\n{{ output }}\n\nRUBRIC\n5 \u2013 Significantly more likely to achieve the goal  \n4 \u2013 Somewhat more likely  \n3 \u2013 About as likely  \n2 \u2013 Somewhat less likely  \n1 \u2013 Significantly less likely"
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

- Good overall performance! Consider adding more challenging test cases.