"""
Integration tests for end-to-end evaluation flow.
"""
import pytest
import asyncio
import pandas as pd
from pathlib import Path
import tempfile
import json

from core.data_models import EvaluationItem, EvaluationMode, EvaluationResults
from core.ingestion import load_evaluation_data, create_sample_data
from core.evaluation import run_evaluation
from core.generation import generate_outputs
from core.reporting import results_to_csv, results_to_json, generate_summary_report
from core.scoring import get_available_scorers
from services.llm_clients import create_llm_client


class TestEndToEndEvaluation:
    """Test complete evaluation workflows."""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = create_sample_data(num_items=5, include_output=True)
            df.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()
    
    @pytest.fixture
    def sample_items(self):
        """Create sample evaluation items."""
        return [
            EvaluationItem(
                id="1",
                input="What is 2+2?",
                output="4",
                expected_output="4"
            ),
            EvaluationItem(
                id="2",
                input="What is the capital of France?",
                output="Paris",
                expected_output="Paris"
            ),
            EvaluationItem(
                id="3",
                input="Translate 'hello' to Spanish",
                output="Hola",
                expected_output="hola"  # Different case
            ),
        ]
    
    def test_load_evaluation_data_from_csv(self, sample_data_file):
        """Test loading evaluation data from CSV file."""
        items = load_evaluation_data(sample_data_file, EvaluationMode.EVALUATE_EXISTING)
        
        assert len(items) == 5
        assert all(isinstance(item, EvaluationItem) for item in items)
        assert all(item.input and item.output and item.expected_output for item in items)
    
    def test_load_evaluation_data_from_dataframe(self):
        """Test loading evaluation data from DataFrame."""
        df = pd.DataFrame({
            "input": ["Question 1", "Question 2"],
            "output": ["Answer 1", "Answer 2"],
            "expected_output": ["Answer 1", "Answer 2"],
            "metadata_source": ["test", "test"],
        })
        
        items = load_evaluation_data(df, EvaluationMode.EVALUATE_EXISTING)
        
        assert len(items) == 2
        assert items[0].metadata["metadata_source"] == "test"
    
    @pytest.mark.asyncio
    async def test_run_evaluation_single_scorer(self, sample_items):
        """Test running evaluation with a single scorer."""
        # Mock API keys
        api_keys = {"openai": "test-key"}
        
        # Run evaluation
        results = await run_evaluation(
            items=sample_items,
            selected_scorers=["exact_match"],
            scorer_configs={},
            api_keys=api_keys,
        )
        
        assert isinstance(results, EvaluationResults)
        assert len(results.items) == 3
        assert all(len(item.scores) == 1 for item in results.items)
        assert "exact_match" in results.summary_stats
        
        # Check summary stats
        stats = results.summary_stats["exact_match"]
        assert stats["total"] == 3
        assert stats["passed"] == 2  # First two should pass
        assert stats["failed"] == 1  # Third fails due to case
    
    @pytest.mark.asyncio
    async def test_run_evaluation_multiple_scorers(self, sample_items):
        """Test running evaluation with multiple scorers."""
        # Configure scorers
        scorer_configs = {
            "fuzzy_match": {"threshold": 0.8},
        }
        
        results = await run_evaluation(
            items=sample_items,
            selected_scorers=["exact_match", "fuzzy_match"],
            scorer_configs=scorer_configs,
            api_keys={},
        )
        
        assert len(results.items) == 3
        assert all(len(item.scores) == 2 for item in results.items)
        assert "exact_match" in results.summary_stats
        assert "fuzzy_match" in results.summary_stats
        
        # Fuzzy match should pass all items
        fuzzy_stats = results.summary_stats["fuzzy_match"]
        assert fuzzy_stats["passed"] == 3
    
    def test_results_to_csv(self, sample_items):
        """Test converting results to CSV format."""
        # Create mock results
        results = EvaluationResults(
            items=sample_items,
            config={"test": True},
            metadata={"mode": "test"},
        )
        
        # Add some scores
        from core.data_models import ScorerResult
        for item in results.items:
            item.scores.append(
                ScorerResult(
                    scorer_name="exact_match",
                    score=1.0 if item.output.lower() == item.expected_output.lower() else 0.0,
                    passed=item.output.lower() == item.expected_output.lower(),
                    reasoning="Test",
                )
            )
        
        csv_content = results_to_csv(results)
        
        assert "id,input,output,expected_output" in csv_content
        assert "exact_match_score" in csv_content
        assert "exact_match_passed" in csv_content
    
    def test_results_to_json(self, sample_items):
        """Test converting results to JSON format."""
        results = EvaluationResults(
            items=sample_items,
            config={"test": True},
            metadata={"mode": "test"},
        )
        
        json_content = results_to_json(results)
        parsed = json.loads(json_content)
        
        assert "items" in parsed
        assert len(parsed["items"]) == 3
        assert "config" in parsed
        assert "metadata" in parsed
    
    def test_generate_summary_report(self, sample_items):
        """Test generating a summary report."""
        # Create results with stats
        results = EvaluationResults(
            items=sample_items,
            config={"scorers": ["exact_match"]},
            metadata={"mode": "evaluate_existing", "duration_seconds": 1.5},
        )
        
        # Add scores
        from core.data_models import ScorerResult
        for i, item in enumerate(results.items):
            item.scores.append(
                ScorerResult(
                    scorer_name="exact_match",
                    score=1.0 if i < 2 else 0.0,
                    passed=i < 2,
                    reasoning="Match" if i < 2 else "No match",
                )
            )
        
        # Calculate stats
        results.calculate_summary_stats()
        
        report = generate_summary_report(results)
        
        assert "# Evaluation Summary Report" in report
        assert "Total Items Evaluated: 3" in report
        assert "Exact Match" in report
        assert "Accuracy: 66.7%" in report
        assert "Items Passed: 2/3" in report
    
    @pytest.mark.asyncio
    async def test_complete_workflow_mode_a(self, sample_data_file):
        """Test complete workflow for Mode A (evaluate existing)."""
        # Load data
        items = load_evaluation_data(sample_data_file, EvaluationMode.EVALUATE_EXISTING)
        
        # Run evaluation
        results = await run_evaluation(
            items=items,
            selected_scorers=["exact_match", "fuzzy_match"],
            scorer_configs={"fuzzy_match": {"threshold": 0.9}},
            api_keys={},
        )
        
        # Generate reports
        csv_report = results_to_csv(results)
        json_report = results_to_json(results)
        summary = generate_summary_report(results)
        
        # Verify outputs
        assert len(results.items) == 5
        assert all(len(item.scores) == 2 for item in results.items)
        assert "exact_match" in results.summary_stats
        assert "fuzzy_match" in results.summary_stats
        assert len(csv_report) > 0
        assert len(json_report) > 0
        assert "# Evaluation Summary Report" in summary
    
    def test_get_available_scorers(self):
        """Test getting list of available scorers."""
        scorers = get_available_scorers()
        
        assert "exact_match" in scorers
        assert "fuzzy_match" in scorers
        assert "llm_judge" in scorers
        
        for scorer_info in scorers.values():
            assert "class" in scorer_info
            assert "display_name" in scorer_info
            assert "description" in scorer_info

    # ADD THESE MARKED TESTS FOR MODE B (GENERATION) AND LLM JUDGE
    @pytest.mark.requires_api
    @pytest.mark.asyncio
    async def test_generate_outputs(self):
        """Test generating outputs using an Actor LLM."""
        items = [
            EvaluationItem(
                id="1",
                input="What is 2+2?",
                expected_output="4"
            ),
            EvaluationItem(
                id="2",
                input="What is the capital of France?",
                expected_output="Paris"
            ),
        ]
        
        actor_config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 100,
            "api_key": os.getenv("OPENAI_API_KEY"),  # This would need a real key
        }
        
        # This test would actually call the OpenAI API
        updated_items = await generate_outputs(items, actor_config)
        
        assert len(updated_items) == 2
        assert all(item.output is not None for item in updated_items)
        assert all(len(item.output) > 0 for item in updated_items)

    @pytest.mark.requires_api
    @pytest.mark.asyncio
    async def test_llm_judge_scorer(self):
        """Test LLM Judge scorer with real API calls."""
        from core.scoring.llm_judge import LLMJudgeScorer
        
        item = EvaluationItem(
            id="1",
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            expected_output="Paris"
        )
        
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "api_key": os.getenv("OPENAI_API_KEY"),  # This would need a real key
        }
        
        scorer = LLMJudgeScorer(config)
        result = await scorer.score(item)
        
        assert result.score >= 0.0 and result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.reasoning is not None
        assert len(result.reasoning) > 0

    @pytest.mark.requires_api
    def test_create_llm_client(self):
        """Test creating LLM clients with real API keys."""
        # This test would validate real API keys
        from services.llm_clients import create_llm_client
        
        # Test OpenAI client
        client = create_llm_client("openai", os.getenv("OPENAI_API_KEY"))
        assert client.validate_api_key()
        
        # Test Anthropic client
        client = create_llm_client("anthropic", os.getenv("ANTHROPIC_API_KEY"))
        assert client.validate_api_key()


class TestErrorHandling:
    """Test error handling in the evaluation pipeline."""
    
    def test_load_invalid_csv(self):
        """Test loading CSV with missing columns."""
        df = pd.DataFrame({
            "input": ["Q1", "Q2"],
            # Missing expected_output column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            load_evaluation_data(df, EvaluationMode.EVALUATE_EXISTING)
    
    @pytest.mark.asyncio
    async def test_scorer_error_handling(self):
        """Test that scorer errors are handled gracefully."""
        # Create item that might cause scorer issues
        items = [
            EvaluationItem(
                id="1",
                input="Test",
                output=None,  # This will cause issues
                expected_output="Expected"
            )
        ]
        
        results = await run_evaluation(
            items=items,
            selected_scorers=["exact_match"],
            scorer_configs={},
            api_keys={},
        )
        
        # Should still return results
        assert len(results.items) == 1
        assert len(results.items[0].scores) == 1
        assert results.items[0].scores[0].score == 0.0
        assert not results.items[0].scores[0].passed
    
    def test_invalid_scorer_name(self):
        """Test handling of invalid scorer name."""
        from core.scoring import create_scorer
        
        with pytest.raises(ValueError, match="Unknown scorer"):
            create_scorer("invalid_scorer_name")


class TestReporting:
    """Test reporting functionality."""
    
    def test_csv_export_with_metadata(self):
        """Test CSV export includes metadata columns."""
        items = [
            EvaluationItem(
                id="1",
                input="Test",
                output="Output",
                expected_output="Expected",
                metadata={"source": "test", "category": "math"}
            )
        ]
        
        results = EvaluationResults(items=items, config={}, metadata={})
        csv_content = results_to_csv(results)
        
        assert "metadata_source" in csv_content
        assert "metadata_category" in csv_content
        assert "test" in csv_content
        assert "math" in csv_content
    
    def test_json_export_structure(self):
        """Test JSON export has correct structure."""
        items = [
            EvaluationItem(
                id="1",
                input="Test",
                output="Output",
                expected_output="Expected"
            )
        ]
        
        from core.data_models import ScorerResult
        items[0].scores.append(
            ScorerResult(
                scorer_name="test_scorer",
                score=0.5,
                passed=False,
                reasoning="Test reasoning",
                details={"extra": "info"}
            )
        )
        
        results = EvaluationResults(
            items=items,
            config={"test": "config"},
            metadata={"test": "metadata"}
        )
        
        json_data = json.loads(results_to_json(results))
        
        assert json_data["items"][0]["id"] == "1"
        assert json_data["items"][0]["scores"][0]["scorer_name"] == "test_scorer"
        assert json_data["items"][0]["scores"][0]["score"] == 0.5
        assert json_data["items"][0]["scores"][0]["details"]["extra"] == "info"
        assert json_data["config"]["test"] == "config"
        assert json_data["metadata"]["test"] == "metadata"


# Note: Tests for Mode B (generation) and LLM Judge would require mocking
# the LLM clients, which would be implemented separately
