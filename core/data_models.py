"""
Pydantic models for structured data exchange throughout the application.
These models serve as the contract between all modules.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class EvalRecord(BaseModel):
    """Backward-compatible record for tests."""

    prompt: str
    expected: str
    output: Optional[str] = None


class Score(BaseModel):
    """Simple score wrapper used in legacy scorers."""

    value: float = Field(..., ge=0.0, le=1.0)


class EvaluationMode(str, Enum):
    """Evaluation modes supported by the system."""
    EVALUATE_EXISTING = "evaluate_existing"
    GENERATE_THEN_EVALUATE = "generate_then_evaluate"


class EvaluationItem(BaseModel):
    """Represents a single item to be evaluated."""
    id: Optional[str] = None
    input: str = Field(..., description="The input/prompt given to the model")
    output: Optional[str] = Field(None, description="The model's actual output")
    expected_output: str = Field(..., description="The ideal/correct output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    scores: List["ScorerResult"] = Field(default_factory=list, description="Scoring results")
    
    @validator("input", "expected_output")
    def non_empty_strings(cls, v):
        if not v or not v.strip():
            raise ValueError("Input and expected_output cannot be empty")
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ScorerResult(BaseModel):
    """Result from a single scorer for an evaluation item."""
    scorer_name: str = Field(..., description="Name of the scorer")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    passed: bool = Field(..., description="Whether the item passed this scorer's criteria")
    reasoning: Optional[str] = Field(None, description="Explanation for the score")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional scorer-specific details")
    error: Optional[str] = Field(None, description="Error message if scoring failed")
    
    @validator("score")
    def validate_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return round(v, 4)  # Round to 4 decimal places


class LLMConfig(BaseModel):
    """Configuration for an LLM client."""
    provider: str = Field(..., description="LLM provider (openai, anthropic, google)")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System prompt for the model")
    api_key: Optional[str] = Field(None, description="API key (if not using environment)")
    
    class Config:
        # Don't include api_key in serialization by default
        fields = {
            "api_key": {"exclude": True}
        }


class ScorerConfig(BaseModel):
    """Configuration for a scorer."""
    name: str = Field(..., description="Scorer name")
    enabled: bool = Field(True, description="Whether this scorer is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Scorer-specific configuration")


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    mode: EvaluationMode
    scorers: List[ScorerConfig]
    actor_config: Optional[LLMConfig] = None  # For generate mode
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class EvaluationResults(BaseModel):
    """Complete results from an evaluation run."""
    items: List[EvaluationItem] = Field(..., description="Evaluated items with scores")
    config: Dict[str, Any] = Field(..., description="Configuration used for this run")
    summary_stats: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Summary statistics per scorer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the run"
    )
    
    def calculate_summary_stats(self) -> None:
        """Calculate summary statistics for all scorers."""
        # Initialize stats dict
        scorer_stats = {}
        
        # Gather scores by scorer
        for item in self.items:
            for score in item.scores:
                if score.scorer_name not in scorer_stats:
                    scorer_stats[score.scorer_name] = {
                        "scores": [],
                        "passed": 0,
                        "failed": 0,
                        "errors": 0,
                    }
                
                if score.error:
                    scorer_stats[score.scorer_name]["errors"] += 1
                elif score.passed:
                    scorer_stats[score.scorer_name]["passed"] += 1
                else:
                    scorer_stats[score.scorer_name]["failed"] += 1
                
                scorer_stats[score.scorer_name]["scores"].append(score.score)
        
        # Calculate final statistics
        for scorer_name, stats in scorer_stats.items():
            total = stats["passed"] + stats["failed"] + stats["errors"]
            scores = stats["scores"]
            
            self.summary_stats[scorer_name] = {
                "total": total,
                "passed": stats["passed"],
                "failed": stats["failed"],
                "errors": stats["errors"],
                "accuracy": stats["passed"] / total if total > 0 else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            }
            
            # Add score distribution for certain scorers
            if scorer_name in ["fuzzy_match", "llm_judge"] and scores:
                import numpy as np
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                hist, _ = np.histogram(scores, bins=bins)
                self.summary_stats[scorer_name]["score_distribution"] = {
                    f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i])
                    for i in range(len(hist))
                }


class RunMetadata(BaseModel):
    """Metadata for an evaluation run."""
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_seconds: Optional[float] = None
    total_items: int = 0
    mode: EvaluationMode
    user_notes: Optional[str] = None
    
    class Config:
        use_enum_values = True


# Update forward references
EvaluationItem.model_rebuild()
