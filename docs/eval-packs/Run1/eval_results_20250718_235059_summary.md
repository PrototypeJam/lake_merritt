# Evaluation Summary Report

Generated on: 2025-07-18 23:50:59

## Overview

- **Total Items Evaluated**: 14
- **Evaluation Mode**: Unknown
- **Duration**: 0.00 seconds
- **Scorers Used**: llm_judge

## Summary Statistics

### Llm Judge

- **Accuracy**: 90.5%
- **Items Passed**: 38/42
- **Items Failed**: 4/42
- **Average Score**: 0.846
- **Score Range**: 0.700 - 0.900

**Score Distribution:**
  - 0.0-0.2: 0 items
  - 0.2-0.4: 0 items
  - 0.4-0.6: 0 items
  - 0.6-0.8: 4 items
  - 0.8-1.0: 38 items

## Failure Analysis

### Llm Judge Failures

Total failures: 4

- **Item afb651ad67a8d03c** (Score: 0.750)
  - Reason: The project plan for the Hyperdomo Agent development is comprehensive in its coverage of the agent's...
- **Item cba2a41fd9129017** (Score: 0.700)
  - Reason: The project plan provides a structured approach to developing specialized worker agents for an auton...
- **Item cba2a41fd9129017** (Score: 0.700)
  - Reason: The project plan provides a solid foundation for developing specialized worker agents, each with a d...
- **Item cba2a41fd9129017** (Score: 0.700)
  - Reason: The project plan is generally sound and provides a comprehensive approach to developing specialized ...

## Configuration

```json
{
  "eval_pack": {
    "name": "Generalized Agento Lifecycle Evaluation",
    "version": "1.0",
    "description": "A reusable evaluation pack that judges the quality of each step in an Agento workflow by leveraging semantic OTEL attributes. Works on any Agento trace.\n",
    "author": null
  },
  "pipeline": [
    {
      "name": "judge_initial_plan",
      "scorer": "llm_judge",
      "config": {
        "provider": "openai",
        "model": "gpt-4o",
        "threshold": 0.8,
        "system_prompt": "You are an expert project manager. Evaluate if the provided project plan is a sound and comprehensive response to the user's goal. Return JSON with \"score\" (0-1) and \"reasoning\".\n",
        "user_prompt_template": "### User's Goal:\n{input}\n\n### Agent's Generated Plan:\n{output}\n\n### Your Task: Evaluate the plan's quality and alignment with the goal.\n"
      },
      "on_fail": "continue",
      "span_kind": null
    },
    {
      "name": "judge_step_draft",
      "scorer": "llm_judge",
      "config": {
        "provider": "openai",
        "model": "gpt-4o",
        "threshold": 0.7,
        "system_prompt": "You are an expert evaluator. Assess if the draft for step '{metadata[step_name]}' faithfully implements its instructions and criteria, considering the overall user goal. Return JSON with \"score\" and \"reasoning\".\n",
        "user_prompt_template": "### Overall User Goal:\n{metadata[user_goal]}\n\n### Instructions for '{metadata[step_name]}':\n{input}\n\n### Evaluation Criteria for this Step:\n{expected_output}\n\n### Agent's Draft for this Step:\n{output}\n\n### Your Task: Evaluate how well the draft fulfills its specific instructions.\n"
      },
      "on_fail": "continue",
      "span_kind": null
    },
    {
      "name": "judge_step_critique",
      "scorer": "llm_judge",
      "config": {
        "provider": "openai",
        "model": "gpt-4o",
        "threshold": 0.8,
        "system_prompt": "You are an expert at providing constructive feedback. Evaluate if the critique for '{metadata[step_name]}' is insightful, actionable, and likely to improve the draft to better meet the user's goal. Return JSON with \"score\" and \"reasoning\".\n",
        "user_prompt_template": "### Overall User Goal:\n{metadata[user_goal]}\n\n### Original Draft for '{metadata[step_name]}' (to be critiqued):\n{input}\n\n### Agent's Critique of the Draft:\n{output}\n\n### Your Task: Evaluate the quality of the critique. Is it valuable?"
      },
      "on_fail": "continue",
      "span_kind": null
    }
  ],
  "batch_size": 10,
  "privacy_settings": {}
}
```

## Recommendations

- Good overall performance! Consider adding more challenging test cases.