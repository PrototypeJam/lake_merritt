# Evaluation Summary Report

Generated on: 2025-07-30 21:26:16

## Overview

- **Total Items Evaluated**: 12
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: LLM Judge, structured_llm_judge

## Summary Statistics

### Llm Judge

- **Accuracy**: 83.3%
- **Items Passed**: 10/12
- **Items Failed**: 2/12
- **Average Score**: 0.775
- **Score Range**: 0.500 - 0.900

### Structured Llm Judge

- **Accuracy**: 0.0%
- **Items Passed**: 0/12
- **Items Failed**: 0/12
- **Errors**: 12
- **Average Score**: 0.000
- **Score Range**: 0.000 - 0.000

## Failure Analysis

### Llm Judge Failures

Total failures: 2

- **Item 8** (Score: 0.500)
  - Reason: The actual output and expected output both serve as testimonial-style ad copies for language learnin...
- **Item 9** (Score: 0.600)
  - Reason: The actual output captures the general structure and purpose of a 30-second radio ad for a car deale...

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
          "user_prompt_template": "Compare the actual output to the expected output for the given input.\n\nInput: {{ input }}\nExpected Output: {{ expected_output }}\nActual Output: {{ output }}\n\nRespond in JSON format with:\n- \"score\": 0.0 to 1.0\n- \"reasoning\": explanation of your evaluation"
        },
        "on_fail": "continue",
        "run_if": null,
        "span_kind": null
      },
      {
        "name": "structured_llm_judge_stage",
        "scorer": "structured_llm_judge",
        "config": {},
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
        "llm_judge",
        "structured_llm_judge"
      ]
    }
  },
  "batch_size": 10,
  "privacy_settings": {}
}
```

## Recommendations

- Overall accuracy is below 50%. Consider reviewing the model's training data or prompts.
- The structured_llm_judge scorer encountered errors. Check API limits or configuration.
- Large variance in scorer results. Ensure all scorers are properly configured and aligned.