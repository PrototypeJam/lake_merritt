# Evaluation Summary Report

Generated on: 2025-07-28 17:03:34

## Overview

- **Total Items Evaluated**: 12
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: fuzzy_match, LLM Judge

## Summary Statistics

### Fuzzy Match

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 12/12
- **Average Score**: 0.557
- **Score Range**: 0.289 - 0.721

**Score Distribution:**
  - 0.0-0.2: 0 items
  - 0.2-0.4: 1 items
  - 0.4-0.6: 6 items
  - 0.6-0.8: 5 items
  - 0.8-1.0: 0 items

### Llm Judge

- **Accuracy**: 91.7%
- **Items Passed**: 11/12
- **Items Failed**: 1/12
- **Average Score**: 0.783
- **Score Range**: 0.500 - 0.900

## Failure Analysis

### Fuzzy Match Failures

Total failures: 12

- **Item 1** (Score: 0.452)
  - Reason: Similarity score 0.45 below threshold 0.85...
- **Item 2** (Score: 0.534)
  - Reason: Similarity score 0.53 below threshold 0.85...
- **Item 3** (Score: 0.289)
  - Reason: Similarity score 0.29 below threshold 0.85...
- **Item 4** (Score: 0.668)
  - Reason: Similarity score 0.67 below threshold 0.85...
- **Item 5** (Score: 0.428)
  - Reason: Similarity score 0.43 below threshold 0.85...
- ... and 7 more failures

### Llm Judge Failures

Total failures: 1

- **Item 11** (Score: 0.500)
  - Reason: The actual output 'Peek-a-Boo AI Cam' introduces a playful and whimsical element to the product name...

## Configuration

```json
{
  "eval_pack": {
    "schema_version": "1.0",
    "name": "CSV quick check",
    "version": "1.0",
    "description": null,
    "author": null,
    "generation": null,
    "ingestion": {
      "type": "csv",
      "parser": null,
      "config": {
        "mode": "evaluate_existing"
      }
    },
    "pipeline": [
      {
        "name": "fuzzy",
        "scorer": "fuzzy_match",
        "config": {
          "threshold": 0.85
        },
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      },
      {
        "name": "judge",
        "scorer": "llm_judge",
        "config": {
          "provider": "openai",
          "model": "gpt-4o",
          "threshold": 0.7,
          "user_prompt_template": "Input: {{ input }}\nExpected: {{ expected_output }}\nActual: {{ output }}\nRespond in JSON with \"score\" and \"reasoning\".\n"
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
- Large variance in scorer results. Ensure all scorers are properly configured and aligned.