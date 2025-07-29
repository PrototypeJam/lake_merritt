# Evaluation Summary Report

Generated on: 2025-07-27 23:27:48

## Overview

- **Total Items Evaluated**: 12
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: exact_match, fuzzy_match, LLM Judge

## Summary Statistics

### Exact Match

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 12/12
- **Average Score**: 0.000
- **Score Range**: 0.000 - 0.000

### Fuzzy Match

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 12/12
- **Average Score**: 0.387
- **Score Range**: 0.338 - 0.440

**Score Distribution:**
  - 0.0-0.2: 0 items
  - 0.2-0.4: 7 items
  - 0.4-0.6: 5 items
  - 0.6-0.8: 0 items
  - 0.8-1.0: 0 items

### Llm Judge

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 0/12
- **Errors**: 12
- **Average Score**: 0.000
- **Score Range**: 0.000 - 0.000

## Failure Analysis

### Exact Match Failures

Total failures: 12

- **Item 1** (Score: 0.000)
  - Reason: Output does not exactly match expected...
- **Item 2** (Score: 0.000)
  - Reason: Output does not exactly match expected...
- **Item 3** (Score: 0.000)
  - Reason: Output does not exactly match expected...
- **Item 4** (Score: 0.000)
  - Reason: Output does not exactly match expected...
- **Item 5** (Score: 0.000)
  - Reason: Output does not exactly match expected...
- ... and 7 more failures

### Fuzzy Match Failures

Total failures: 12

- **Item 1** (Score: 0.420)
  - Reason: Similarity score 0.42 below threshold 0.8...
- **Item 2** (Score: 0.440)
  - Reason: Similarity score 0.44 below threshold 0.8...
- **Item 3** (Score: 0.381)
  - Reason: Similarity score 0.38 below threshold 0.8...
- **Item 4** (Score: 0.417)
  - Reason: Similarity score 0.42 below threshold 0.8...
- **Item 5** (Score: 0.376)
  - Reason: Similarity score 0.38 below threshold 0.8...
- ... and 7 more failures

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
        "name": "exact_match_stage",
        "scorer": "exact_match",
        "config": {},
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      },
      {
        "name": "fuzzy_match_stage",
        "scorer": "fuzzy_match",
        "config": {},
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      },
      {
        "name": "llm_judge_stage",
        "scorer": "llm_judge",
        "config": {
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
        "exact_match",
        "fuzzy_match",
        "llm_judge"
      ]
    }
  },
  "batch_size": 10,
  "privacy_settings": {}
}
```

## Recommendations

- Overall accuracy is below 50%. Consider reviewing the model's training data or prompts.
- Low exact match scores. Consider using fuzzy matching for more flexibility.
- The LLM Judge scorer encountered errors. Check API limits or configuration.