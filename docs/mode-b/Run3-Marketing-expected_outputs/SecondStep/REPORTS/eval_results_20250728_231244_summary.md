# Evaluation Summary Report

Generated on: 2025-07-28 23:12:44

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
- **Average Score**: 0.581
- **Score Range**: 0.397 - 0.694

**Score Distribution:**
  - 0.0-0.2: 0 items
  - 0.2-0.4: 1 items
  - 0.4-0.6: 4 items
  - 0.6-0.8: 7 items
  - 0.8-1.0: 0 items

### Llm Judge

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 0/12
- **Errors**: 12
- **Average Score**: 0.000
- **Score Range**: 0.000 - 0.000

## Failure Analysis

### Fuzzy Match Failures

Total failures: 12

- **Item 1** (Score: 0.397)
  - Reason: Similarity score 0.40 below threshold 0.8...
- **Item 2** (Score: 0.592)
  - Reason: Similarity score 0.59 below threshold 0.8...
- **Item 3** (Score: 0.462)
  - Reason: Similarity score 0.46 below threshold 0.8...
- **Item 4** (Score: 0.694)
  - Reason: Similarity score 0.69 below threshold 0.8...
- **Item 5** (Score: 0.612)
  - Reason: Similarity score 0.61 below threshold 0.8...
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
    "ingestion": {
      "type": "csv",
      "parser": null,
      "config": {
        "mode": "evaluate_existing"
      }
    },
    "pipeline": [
      {
        "name": "fuzzy_match_stage",
        "scorer": "fuzzy_match",
        "config": {
          "threshold": 0.8
        },
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      },
      {
        "name": "llm_judge_stage",
        "scorer": "llm_judge",
        "config": {
          "provider": "openai",
          "model": "gpt-4.1",
          "temperature": 0.3,
          "max_tokens": 1000,
          "system_prompt": "You are an expert evaluator. Compare the actual output to the expected output and provide:\n1. A score from 0.0 to 1.0 (where 1.0 is perfect match)\n2. A brief reasoning for your score\n3. Any specific errors or discrepancies noted\n\nRespond in JSON format:\n{\n    \"score\": 0.0-1.0,\n    \"reasoning\": \"explanation\",\n    \"errors\": [\"error1\", \"error2\"] or []\n}",
          "threshold": 0.7,
          "user_prompt_template": "Compare the actual output to the expected output for the given input.\n\nInput: {input}\nExpected Output: {expected_output}\nActual Output: {output}\n\nRespond in JSON format with:\n- \"score\": 0.0 to 1.0\n- \"reasoning\": explanation of your evaluation",
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
- The LLM Judge scorer encountered errors. Check API limits or configuration.