# core/registry.py
from typing import Dict, Type
from core.scoring.base import BaseScorer
from core.ingestion.base import BaseIngester

class ComponentRegistry:
    """Central registry for discoverable components"""
    _scorers: Dict[str, Type[BaseScorer]] = {}
    _ingesters: Dict[str, Type[BaseIngester]] = {}
    _aggregators: Dict[str, Type] = {}  # Add aggregator registry
    
    @classmethod
    def register_scorer(cls, name: str, scorer_class: Type[BaseScorer]):
        cls._scorers[name] = scorer_class
    
    @classmethod
    def get_scorer(cls, name: str) -> Type[BaseScorer]:
        if name not in cls._scorers:
            raise ValueError(f"Unknown scorer: {name}")
        return cls._scorers[name]
    
    @classmethod
    def register_ingester(cls, name: str, ingester_class: Type[BaseIngester]):
        cls._ingesters[name] = ingester_class
        
    @classmethod
    def get_ingester(cls, name: str) -> Type[BaseIngester]:
        if name not in cls._ingesters:
            raise ValueError(f"Unknown ingester: {name}")
        return cls._ingesters[name]

    @classmethod
    def register_aggregator(cls, name: str, aggregator_class: Type):
        """Register an aggregator class"""
        cls._aggregators[name] = aggregator_class
    
    @classmethod
    def get_aggregator(cls, name: str) -> Type:
        """Get an aggregator class by name"""
        if name not in cls._aggregators:
            raise ValueError(f"Unknown aggregator: {name}")
        return cls._aggregators[name]

    @classmethod
    def discover_builtins(cls):
        """Auto-register all built-in components, tolerating those not yet implemented."""
        # Register existing scorers
        try:
            from core.scoring import (
                ExactMatchScorer, FuzzyMatchScorer,
                LLMJudgeScorer, CriteriaSelectionJudgeScorer
            )
            cls.register_scorer("exact_match", ExactMatchScorer)
            cls.register_scorer("fuzzy_match", FuzzyMatchScorer)
            cls.register_scorer("llm_judge", LLMJudgeScorer)
            cls.register_scorer("criteria_selection_judge", CriteriaSelectionJudgeScorer)
        except ImportError:
            print("INFO: Core scorers not found, skipping registration.")

        try:
            from core.scoring.tool_usage_scorer import ToolUsageScorer
            cls.register_scorer("tool_usage", ToolUsageScorer)
        except ImportError:
            print("INFO: ToolUsageScorer not implemented yet, skipping.")

        # Register NEW FDL/BBQ scorers
        try:
            from core.scoring.fdl_alignment_scorer import FDLAlignmentScorer
            cls.register_scorer("fdl_alignment", FDLAlignmentScorer)
        except ImportError:
            print("INFO: FDLAlignmentScorer not found, skipping registration.")

        try:
            from core.scoring.fdl_disclosure_scorer import FDLDisclosureScorer
            cls.register_scorer("fdl_disclosure", FDLDisclosureScorer)
        except ImportError:
            print("INFO: FDLDisclosureScorer not found, skipping registration.")

        try:
            from core.scoring.choice_index_scorer import ChoiceIndexScorer
            cls.register_scorer("choice_index", ChoiceIndexScorer)
        except ImportError:
            print("INFO: ChoiceIndexScorer not found, skipping registration.")

        # Register existing ingesters
        try:
            from core.ingestion.csv_ingester import CSVIngester
            cls.register_ingester("csv", CSVIngester)
        except ImportError:
            print("INFO: CSVIngester not implemented yet, skipping.")

        try:
            from core.ingestion.json_ingester import JSONIngester
            cls.register_ingester("json", JSONIngester)
        except ImportError:
            print("INFO: JSONIngester not implemented yet, skipping.")

        try:
            from core.ingestion.openinference_ingester import OpenInferenceIngester
            cls.register_ingester("openinference", OpenInferenceIngester)
        except ImportError:
            print("INFO: OpenInferenceIngester not implemented yet, skipping.")

        try:
            from core.ingestion.generic_otel_ingester import GenericOtelIngester
            cls.register_ingester("generic_otel", GenericOtelIngester)
        except ImportError:
            print("INFO: GenericOtelIngester not implemented yet, skipping.")
        
        try:
            from core.otel.ingester import OTelTraceIngester
            cls.register_ingester("otel", OTelTraceIngester)
        except ImportError:
            print("INFO: OTelTraceIngester not implemented yet, skipping.")
        
        # Add alias for GenericOtelIngester for easier discovery
        try:
            from core.ingestion.generic_otel_ingester import GenericOtelIngester
            cls.register_ingester("otel_generic", GenericOtelIngester)
        except ImportError:
            print("INFO: GenericOtelIngester alias not registered, skipping.")
        
        # Register Python ingester
        try:
            from core.ingestion.python_ingester import PythonIngester
            cls.register_ingester("python", PythonIngester)
        except ImportError:
            print("INFO: PythonIngester not implemented yet, skipping.")

        # Register NEW aggregators
        try:
            from core.aggregators.fdl_metrics_aggregator import FDLMetricsAggregator
            cls.register_aggregator("FDLMetricsAggregator", FDLMetricsAggregator)
        except ImportError:
            print("INFO: FDLMetricsAggregator not found, skipping registration.")

        try:
            from core.aggregators.bbq_bias_aggregator import BBQBiasScoreAggregator
            cls.register_aggregator("BBQBiasScoreAggregator", BBQBiasScoreAggregator)
        except ImportError:
            print("INFO: BBQBiasScoreAggregator not found, skipping registration.")