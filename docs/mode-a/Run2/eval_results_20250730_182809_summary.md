# Evaluation Summary Report

Generated on: 2025-07-30 18:28:09

## Overview

- **Total Items Evaluated**: 12
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: LLM Judge

## Summary Statistics

### Llm Judge

- **Accuracy**: 83.3%
- **Items Passed**: 10/12
- **Items Failed**: 2/12
- **Average Score**: 0.708
- **Score Range**: 0.300 - 0.900

## Failure Analysis

### Llm Judge Failures

Total failures: 2

- **Item 6** (Score: 0.600)
  - Reason: The actual output captures the essence of promoting a 7-day trial for a fitness app with an emphasis...
- **Item 11** (Score: 0.300)
  - Reason: The actual output 'Peek-a-Boo AI Cam' significantly deviates from the expected output 'ObserveSafe A...

## Configuration

```json
{
  "eval_pack": {
    "schema_version": "1.0",
    "name": "Legacy UI Configuration",
    "version": "1.0",
    "description": "Automatically generated from legacy UI selections",
    "author": "Legacy UI",
    "generation": null,
    "ingestion": {
      "type": "json",
      "parser": null,
      "config": {}
    },
    "pipeline": [
      {
        "name": "llm_judge_stage",
        "scorer": "llm_judge",
        "config": {
          "provider": "openai",
          "model": "gpt-4o",
          "temperature": 0.3,
          "max_tokens": 1000,
          "system_prompt": "You are an expert evaluator. Compare the actual output to the expected output and provide:\n1. A score from 0.0 to 1.0 (where 1.0 is perfect match)\n2. A brief reasoning for your score\n3. Any specific errors or discrepancies noted\n\nRespond in JSON format:\n{\n    \"score\": 0.0-1.0,\n    \"reasoning\": \"explanation\",\n    \"errors\": [\"error1\", \"error2\"] or []\n}",
          "user_prompt_template": "Compare the actual output to the expected output for the given input.\n\nInput: {{ input }}\nExpected Output: {{ expected_output }}\nActual Output: {{ output }}\n\nRespond in JSON format with:\n- \"score\": 0.0 to 1.0\n- \"reasoning\": explanation of your evaluation",
          "api_key": "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"
        },
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      }
    ],
    "reporting": null,
    "metadata": {
      "source": "legacy_ui",
      "auto_generated": true,
      "selected_scorers": [
        "llm_judge"
      ]
    }
  },
  "batch_size": 10,
  "privacy_settings": {}
}
```

## Recommendations

- Good overall performance! Consider adding more challenging test cases.