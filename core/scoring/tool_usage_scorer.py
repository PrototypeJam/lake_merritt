# core/scoring/tool_usage_scorer.py
from typing import Dict, Any, List, Optional
from core.scoring.base import BaseScorer
from core.data_models import EvaluationItem, ScorerResult

class ToolUsageScorer(BaseScorer):
    """
    Scores based on tool usage in OpenTelemetry traces.
    
    This scorer examines tool usage patterns in OTEL traces and scores based on:
    - Whether specific tools were used
    - The sequence of tool usage
    - Tool usage frequency
    - Tool success/failure rates
    """
    
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.required_tools = self.config.get("required_tools", [])
        self.forbidden_tools = self.config.get("forbidden_tools", [])
        self.check_sequence = self.config.get("check_sequence", False)
        self.expected_sequence = self.config.get("expected_sequence", [])
        self.min_usage_count = self.config.get("min_usage_count", {})
        self.max_usage_count = self.config.get("max_usage_count", {})
    
    @property
    def name(self) -> str:
        return "tool_usage"
    
    @property
    def description(self) -> str:
        return "Scores based on tool usage patterns in OpenTelemetry traces"
    
    def score(self, item: EvaluationItem) -> ScorerResult:
        """Score based on tool usage in the trace."""
        try:
            # Extract OTEL trace from metadata
            otel_trace = item.metadata.get("otel_trace")
            if not otel_trace:
                return ScorerResult(
                    scorer_name=self.name,
                    score=0.0,
                    numeric_score=0.0,
                    passed=False,
                    error="No OTEL trace found in metadata"
                )
            
            # Extract tool calls from the trace
            tool_calls = self._extract_tool_calls(otel_trace)
            tool_names = [tc["name"] for tc in tool_calls]
            
            # Initialize scoring components
            score_components = []
            issues = []
            
            # Check required tools
            if self.required_tools:
                missing_tools = set(self.required_tools) - set(tool_names)
                if missing_tools:
                    issues.append(f"Missing required tools: {', '.join(missing_tools)}")
                    score_components.append(0.0)
                else:
                    score_components.append(1.0)
            
            # Check forbidden tools
            if self.forbidden_tools:
                used_forbidden = set(self.forbidden_tools) & set(tool_names)
                if used_forbidden:
                    issues.append(f"Used forbidden tools: {', '.join(used_forbidden)}")
                    score_components.append(0.0)
                else:
                    score_components.append(1.0)
            
            # Check tool usage sequence
            if self.check_sequence and self.expected_sequence:
                sequence_score = self._check_sequence(tool_names, self.expected_sequence)
                score_components.append(sequence_score)
                if sequence_score < 1.0:
                    issues.append("Tool usage sequence does not match expected pattern")
            
            # Check usage counts
            tool_counts = {}
            for tool in tool_names:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # Min usage count
            for tool, min_count in self.min_usage_count.items():
                actual_count = tool_counts.get(tool, 0)
                if actual_count < min_count:
                    issues.append(f"Tool '{tool}' used {actual_count} times, expected at least {min_count}")
                    score_components.append(0.0)
                else:
                    score_components.append(1.0)
            
            # Max usage count
            for tool, max_count in self.max_usage_count.items():
                actual_count = tool_counts.get(tool, 0)
                if actual_count > max_count:
                    issues.append(f"Tool '{tool}' used {actual_count} times, expected at most {max_count}")
                    score_components.append(0.0)
                else:
                    score_components.append(1.0)
            
            # Calculate final score
            if score_components:
                final_score = sum(score_components) / len(score_components)
            else:
                final_score = 1.0  # No criteria specified, so pass by default
            
            passed = final_score >= self.config.get("threshold", 0.8)
            
            return ScorerResult(
                scorer_name=self.name,
                score=final_score,
                numeric_score=final_score,
                passed=passed,
                reasoning="; ".join(issues) if issues else "All tool usage criteria met",
                details={
                    "tool_calls": tool_calls,
                    "tool_counts": tool_counts,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                numeric_score=0.0,
                passed=False,
                error=f"Error scoring tool usage: {str(e)}"
            )
    
    def _extract_tool_calls(self, otel_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from OTEL trace."""
        tool_calls = []
        
        # Handle different OTEL trace formats
        if "spans" in otel_trace:
            # Standard OTEL format
            for span in otel_trace.get("spans", []):
                if span.get("kind") == "TOOL" or "tool" in span.get("name", "").lower():
                    tool_calls.append({
                        "name": span.get("name", "unknown"),
                        "status": span.get("status", {}).get("code", "OK"),
                        "duration_ms": span.get("duration_ms", 0),
                        "attributes": span.get("attributes", {})
                    })
        elif "steps" in otel_trace:
            # Custom format (like manual_traces.json)
            for step in otel_trace.get("steps", []):
                if "tool" in step.get("stage", "").lower():
                    tool_calls.append({
                        "name": step.get("tool_name", step.get("stage", "unknown")),
                        "status": "OK" if step.get("outputs") else "ERROR",
                        "duration_ms": 0,  # Not available in this format
                        "attributes": step.get("outputs", {})
                    })
        
        return tool_calls
    
    def _check_sequence(self, actual: List[str], expected: List[str]) -> float:
        """Check if actual tool sequence matches expected pattern."""
        if not actual or not expected:
            return 0.0
        
        # Simple subsequence matching
        j = 0
        matches = 0
        for tool in actual:
            if j < len(expected) and tool == expected[j]:
                matches += 1
                j += 1
        
        return matches / len(expected)