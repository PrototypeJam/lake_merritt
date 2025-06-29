"""
Unit tests for LLMJudgeScorer with pack-friendly features.
"""

import json
import logging
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.data_models import EvaluationItem, ScorerResult
from core.scoring.llm_judge import LLMJudgeScorer


class FakeLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, response: str):
        self.response = response
        self.last_messages = None
        self.call_count = 0
    
    async def generate(self, messages, model, temperature, max_tokens):
        self.last_messages = messages
        self.call_count += 1
        return self.response


def create_scorer_with_mock(monkeypatch, response: str, config: Dict[str, Any] = None):
    """Helper to create scorer with mocked LLM client."""
    fake_client = FakeLLMClient(response)
    monkeypatch.setattr(
        "core.scoring.llm_judge.create_llm_client",
        lambda provider, api_key: fake_client
    )
    scorer = LLMJudgeScorer(config or {})
    return scorer, fake_client


class TestLLMJudgeScorer:
    """Test cases for pack-friendly LLMJudgeScorer."""
    
    @pytest.mark.asyncio
    async def test_basic_scoring(self, monkeypatch):
        """Test basic scoring functionality."""
        response = json.dumps({
            "score": 0.9,
            "passed": True,
            "reasoning": "Excellent match",
            "errors": []
        })
        
        scorer, client = create_scorer_with_mock(monkeypatch, response)
        item = EvaluationItem(
            input="What is 2+2?",
            output="4",
            expected_output="4"
        )
        
        result = await scorer.score(item)
        
        assert result.score == 0.9
        assert result.passed is True
        assert result.reasoning == "Excellent match"
        assert result.scorer_name == "llm_judge"
    
    @pytest.mark.asyncio
    async def test_default_prompt_warns_on_missing_passed(self, monkeypatch, caplog):
        """Test that default prompt warns when 'passed' field is missing."""
        response = json.dumps({
            "score": 0.8,
            "reasoning": "Good answer"
        })
        
        scorer, client = create_scorer_with_mock(monkeypatch, response)
        item = EvaluationItem(
            input="Question",
            output="Answer",
            expected_output="Answer"
        )
        
        with caplog.at_level(logging.WARNING):
            result = await scorer.score(item)
        
        # Should warn about missing 'passed' field
        assert "missing 'passed'" in caplog.text
        assert result.passed is True  # Should use threshold (0.7 default)
        assert result.score == 0.8
        
        # Verify default prompt was used
        user_message = client.last_messages[1]["content"]
        assert "Please evaluate the following" in user_message
    
    @pytest.mark.asyncio
    async def test_custom_template_no_warning(self, monkeypatch, caplog):
        """Test that custom template suppresses missing 'passed' warning."""
        response = json.dumps({
            "score": 0.8,
            "reasoning": "Good answer"
        })
        
        template = "Evaluate this: Input={input}, Output={output}, Expected={expected_output}"
        config = {"user_prompt_template": template}
        
        scorer, client = create_scorer_with_mock(monkeypatch, response, config)
        item = EvaluationItem(
            input="Q1",
            output="A1",
            expected_output="A1"
        )
        
        with caplog.at_level(logging.WARNING):
            result = await scorer.score(item)
        
        # Should NOT warn about missing 'passed' when using custom template
        assert "missing 'passed'" not in caplog.text
        assert result.passed is True
        assert result.score == 0.8
        
        # Verify custom template was used
        expected_prompt = template.format(
            input="Q1",
            output="A1",
            expected_output="A1"
        )
        assert client.last_messages[1]["content"] == expected_prompt
    
    @pytest.mark.asyncio
    async def test_threshold_configuration(self, monkeypatch):
        """Test that threshold can be configured."""
        response = json.dumps({
            "score": 0.85,
            "reasoning": "Pretty good"
        })
        
        # Test with high threshold
        config = {"threshold": 0.9}
        scorer, _ = create_scorer_with_mock(monkeypatch, response, config)
        item = EvaluationItem(
            input="Question",
            output="Answer",
            expected_output="Different Answer"
        )
        
        result = await scorer.score(item)
        
        assert result.score == 0.85
        assert result.passed is False  # 0.85 < 0.9 threshold
        assert result.details["threshold"] == 0.9
    
    @pytest.mark.asyncio
    async def test_no_output_handling(self, monkeypatch):
        """Test handling when item has no output."""
        scorer, _ = create_scorer_with_mock(monkeypatch, "unused")
        item = EvaluationItem(
            input="Question",
            output=None,
            expected_output="Expected"
        )
        
        result = await scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
        assert result.reasoning == "No output provided"
    
    @pytest.mark.asyncio
    async def test_json_extraction_fallback(self, monkeypatch, caplog):
        """Test fallback when JSON extraction fails."""
        # Response with invalid JSON
        response = "The score is about 0.75 out of 1.0, which is pretty good."
        
        scorer, _ = create_scorer_with_mock(monkeypatch, response)
        item = EvaluationItem(
            input="Question",
            output="Answer",
            expected_output="Answer"
        )
        
        with caplog.at_level(logging.ERROR):
            result = await scorer.score(item)
        
        assert "Failed to parse LLM response as JSON" in caplog.text
        assert result.score == 0.75  # Should extract from text
        assert result.passed is True  # 0.75 >= 0.7 default threshold
        assert "Failed to parse structured response" in result.details["errors"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, monkeypatch):
        """Test error handling when LLM call fails."""
        # Mock client that raises exception
        async def failing_generate(*args, **kwargs):
            raise Exception("API connection failed")
        
        mock_client = MagicMock()
        mock_client.generate = failing_generate
        
        monkeypatch.setattr(
            "core.scoring.llm_judge.create_llm_client",
            lambda provider, api_key: mock_client
        )
        
        scorer = LLMJudgeScorer({})
        item = EvaluationItem(
            input="Question",
            output="Answer",
            expected_output="Answer"
        )
        
        result = await scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
        assert "API connection failed" in result.error
        assert "Failed to obtain LLM judgment" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_raw_response_inclusion(self, monkeypatch):
        """Test that raw response is included when configured."""
        response = json.dumps({
            "score": 0.9,
            "passed": True,
            "reasoning": "Good"
        })
        
        config = {"include_raw_response": True}
        scorer, _ = create_scorer_with_mock(monkeypatch, response, config)
        item = EvaluationItem(
            input="Q",
            output="A",
            expected_output="A"
        )
        
        result = await scorer.score(item)
        
        assert result.details["raw_response"] == response
    
    @pytest.mark.asyncio
    async def test_score_clamping(self, monkeypatch):
        """Test that scores are clamped to [0, 1] range."""
        # Test score > 1
        response = json.dumps({
            "score": 1.5,
            "passed": True,
            "reasoning": "Too high"
        })
        
        scorer, _ = create_scorer_with_mock(monkeypatch, response)
        item = EvaluationItem(
            input="Q",
            output="A",
            expected_output="A"
        )
        
        result = await scorer.score(item)
        assert result.score == 1.0  # Clamped to max
        
        # Test score < 0
        response = json.dumps({
            "score": -0.5,
            "passed": False,
            "reasoning": "Too low"
        })
        
        scorer, _ = create_scorer_with_mock(monkeypatch, response)
        result = await scorer.score(item)
        assert result.score == 0.0  # Clamped to min


@pytest.mark.parametrize("text,expected_score", [
    ("The score is 0.8", 0.8),
    ("I give it 7/10", 0.7),
    ("85% correct", 0.85),
    ("Score: 0.95", 0.95),
    ("9 out of 10", 0.9),
    ("50%", 0.5),
    ("No score found", 0.0),
])
def test_extract_score_from_text(text, expected_score):
    """Test score extraction from various text formats."""
    scorer = LLMJudgeScorer({})
    score = scorer._extract_score_from_text(text)
    assert score == expected_score