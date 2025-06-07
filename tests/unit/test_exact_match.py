"""
Unit tests for exact match scorer.
"""
import pytest
from core.data_models import EvaluationItem, ScorerResult
from core.scoring.exact_match import (
    ExactMatchScorer,
    CaseInsensitiveExactMatchScorer,
    NormalizedExactMatchScorer,
)


class TestExactMatchScorer:
    """Test cases for ExactMatchScorer."""
    
    def test_exact_match_success(self):
        """Test exact match when strings are identical."""
        scorer = ExactMatchScorer()
        item = EvaluationItem(
            input="What is 2+2?",
            output="4",
            expected_output="4"
        )
        
        result = scorer.score(item)
        
        assert isinstance(result, ScorerResult)
        assert result.score == 1.0
        assert result.passed is True
        assert "Exact match found" in result.reasoning
    
    def test_exact_match_failure(self):
        """Test exact match when strings differ."""
        scorer = ExactMatchScorer()
        item = EvaluationItem(
            input="What is 2+2?",
            output="Four",
            expected_output="4"
        )
        
        result = scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
        assert "does not exactly match" in result.reasoning
    
    def test_exact_match_whitespace_handling(self):
        """Test that whitespace is stripped before comparison."""
        scorer = ExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="  Hello World  ",
            expected_output="Hello World"
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_exact_match_no_output(self):
        """Test handling when output is None."""
        scorer = ExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output=None,
            expected_output="Expected"
        )
        
        result = scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
        assert "No output provided" in result.reasoning
    
    def test_exact_match_includes_details(self):
        """Test that result includes helpful details."""
        scorer = ExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="Hello",
            expected_output="Hello World"
        )
        
        result = scorer.score(item)
        
        assert result.details["output_length"] == 5
        assert result.details["expected_length"] == 11
        assert result.details["stripped_match"] is False


class TestCaseInsensitiveExactMatchScorer:
    """Test cases for CaseInsensitiveExactMatchScorer."""
    
    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        scorer = CaseInsensitiveExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="HELLO WORLD",
            expected_output="hello world"
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_case_insensitive_no_match(self):
        """Test case-insensitive non-matching."""
        scorer = CaseInsensitiveExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="HELLO",
            expected_output="hello world"
        )
        
        result = scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
    
    def test_details_include_case_sensitive_info(self):
        """Test that details show both case-sensitive and insensitive results."""
        scorer = CaseInsensitiveExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="Hello",
            expected_output="HELLO"
        )
        
        result = scorer.score(item)
        
        assert result.details["case_sensitive_match"] is False
        assert result.details["case_insensitive_match"] is True


class TestNormalizedExactMatchScorer:
    """Test cases for NormalizedExactMatchScorer."""
    
    def test_normalization_whitespace(self):
        """Test whitespace normalization."""
        scorer = NormalizedExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="Hello    World",  # Multiple spaces
            expected_output="hello world"      # Single space
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_normalization_smart_quotes(self):
        """Test smart quote normalization."""
        scorer = NormalizedExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="“Hello World”",  # Smart quotes
            expected_output='"Hello World"'    # Regular quotes
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_normalization_apostrophes(self):
        """Test smart apostrophe normalization."""
        scorer = NormalizedExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="It’s working",     # Smart apostrophe
            expected_output="It's working"  # Regular apostrophe
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_ignore_trailing_punctuation_config(self):
        """Test configuration to ignore trailing punctuation."""
        scorer = NormalizedExactMatchScorer({"ignore_trailing_punctuation": True})
        item = EvaluationItem(
            input="Test",
            output="Hello World!",
            expected_output="Hello World"
        )
        
        result = scorer.score(item)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_keep_trailing_punctuation_default(self):
        """Test that trailing punctuation is kept by default."""
        scorer = NormalizedExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="Hello World!",
            expected_output="Hello World"
        )
        
        result = scorer.score(item)
        
        assert result.score == 0.0
        assert result.passed is False
    
    def test_details_show_normalized_values(self):
        """Test that details include normalized values."""
        scorer = NormalizedExactMatchScorer()
        item = EvaluationItem(
            input="Test",
            output="HELLO   WORLD",
            expected_output="hello world"
        )
        
        result = scorer.score(item)
        
        assert result.details["normalized_output"] == "hello world"
        assert result.details["normalized_expected"] == "hello world"
        assert result.details["raw_match"] is False


@pytest.mark.parametrize("output,expected,should_match", [
    ("Hello", "Hello", True),
    ("Hello ", " Hello", True),
    ("HELLO", "hello", False),
    ("Hello!", "Hello", False),
    ("", "", True),
    ("  ", "", True),
])
def test_exact_match_parametrized(output, expected, should_match):
    """Parametrized tests for exact match scorer."""
    scorer = ExactMatchScorer()
    item = EvaluationItem(
        input="Test",
        output=output,
        expected_output=expected
    )
    
    result = scorer.score(item)
    
    if should_match:
        assert result.score == 1.0
        assert result.passed is True
    else:
        assert result.score == 0.0
        assert result.passed is False
