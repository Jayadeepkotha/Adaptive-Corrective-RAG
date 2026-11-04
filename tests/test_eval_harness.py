import pytest
import tempfile
import json
from pathlib import Path

from unittest.mock import MagicMock
from eval.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_contains_answer,
    compute_retrieval_recall,
    normalize_answer,
    classify_error,
    compute_bootstrap_ci,
    compute_paired_bootstrap,
)
from eval.cache import EvalCache
from eval.dataset import DatasetLoader
from eval.run import resolve_strategies, ALL_STRATEGIES
from src.pipeline import AdaptiveRAGPipeline
from src.classifier import ClassificationOutput, ComplexityLevel
from src.retriever import DocumentChunk

def test_answer_normalization():
    assert normalize_answer("The United States of America.") == "united states of america"
    assert normalize_answer("  A quick brown fox!  ") == "quick brown fox"
    assert normalize_answer("An Apple, Inc.") == "apple inc"

def test_exact_match_metric():
    assert compute_exact_match("Paris", ["paris", "City of Paris"]) == 1.0
    assert compute_exact_match("London", ["Paris", "Rome"]) == 0.0
    assert compute_exact_match("The Louvre", ["louvre"]) == 1.0

def test_token_f1_metric():
    # Exact match has F1 = 1.0
    assert compute_token_f1("Barack Obama", ["Barack Obama"]) == 1.0
    # Partial overlap
    f1 = compute_token_f1("President Barack Obama", ["Barack Obama"])
    assert 0.6 < f1 < 1.0
    # No overlap
    assert compute_token_f1("Eiffel Tower", ["Statue of Liberty"]) == 0.0

def test_contains_answer_metric():
    assert compute_contains_answer("The capital is Paris.", ["Paris"]) == 1.0
    assert compute_contains_answer("He was born in Berlin in 1945.", ["Berlin"]) == 1.0
    assert compute_contains_answer("Answer is Rome.", ["Madrid"]) == 0.0

def test_retrieval_recall_metric():
    retrieved = [
        {"doc_name": "Albert Einstein.txt", "score": 0.95},
        {"doc_name": "Theory of Relativity", "score": 0.88},
        {"doc_name": "Nobel Prize in Physics", "score": 0.75}
    ]
    # Both gold titles retrieved
    gold = ["Albert Einstein", "Theory of Relativity"]
    assert compute_retrieval_recall(retrieved, gold) == 1.0

    # 1 of 2 retrieved
    gold_partial = ["Albert Einstein", "Isaac Newton"]
    assert compute_retrieval_recall(retrieved, gold_partial) == 0.5

    # 0 of 1 retrieved
    assert compute_retrieval_recall(retrieved, ["Marie Curie"]) == 0.0

def test_resolve_strategies():
    assert resolve_strategies("all") == ALL_STRATEGIES
    assert resolve_strategies("no_retrieval,single_step") == ["no_retrieval", "single_step"]
    with pytest.raises(ValueError):
        resolve_strategies("invalid_strat")

def test_eval_cache_disk_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = EvalCache(cache_dir=Path(tmp_dir))
        messages = [{"role": "user", "content": "Hello world"}]
        usage = {"prompt_tokens": 10, "completion_tokens": 5}

        # Cache miss
        assert cache.get_llm("test-model", messages) is None

        # Cache set
        cache.set_llm("test-model", messages, "Hello there!", usage=usage)

        # Cache hit
        cached = cache.get_llm("test-model", messages)
        assert cached is not None
        assert cached["text"] == "Hello there!"
        assert cached["usage"] == usage

def test_dataset_seeded_split_disjoint():
    loader = DatasetLoader()
    tune_samples = loader.load_samples("hotpotqa", split="tune", seed=42)
    test_samples = loader.load_samples("hotpotqa", split="test", seed=42)
    assert len(tune_samples) == 250
    assert len(test_samples) == 250
    tune_ids = set(s.id for s in tune_samples)
    test_ids = set(s.id for s in test_samples)
    assert tune_ids.isdisjoint(test_ids), "Tune and Test splits must be strictly disjoint!"

def test_error_taxonomy_classification():
    assert classify_error(1.0, 1.0, 1.0) == "correct"
    assert classify_error(0.0, 1.0, 1.0) == "format_error"
    assert classify_error(0.0, 1.0, 0.5) == "format_error"
    assert classify_error(0.0, 0.0, 0.5) == "retrieval_miss"
    assert classify_error(0.0, 0.0, 0.0) == "retrieval_miss"
    assert classify_error(0.0, 0.0, 1.0) == "reasoning_error"

def test_bootstrap_ci_and_paired():
    em_scores = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    ci_low, ci_high = compute_bootstrap_ci(em_scores, n_bootstrap=1000, seed=42)
    assert 0.0 <= ci_low <= ci_high <= 100.0

    scores_a = [1.0, 1.0, 0.0, 0.0]
    scores_b = [0.0, 1.0, 0.0, 0.0]
    paired = compute_paired_bootstrap(scores_a, scores_b, n_bootstrap=1000, seed=42)
    assert paired["wins"] == 1
    assert paired["losses"] == 0
    assert paired["ties"] == 3
    assert paired["diff_mean"] == 25.0

def test_call_accounting_and_cached_flags():
    pipeline = AdaptiveRAGPipeline(enable_graph=False, enable_rerank=False, enable_web=False)
    pipeline.reset_counters()

    # Mock generator and classifier
    pipeline.generator._call_llm = MagicMock(return_value="Mocked Answer")
    pipeline.generator.logical_calls = 1
    pipeline.generator.llm_calls = 1
    pipeline.generator.cache_hits = 1
    pipeline.generator.prompt_tokens = 50
    pipeline.generator.completion_tokens = 10
    pipeline.generator.call_records = [{
        "caller": "generator",
        "operation": "generate",
        "model": "test-model",
        "cached": True,
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "retries": 0,
        "latency_s": 0.01
    }]

    stats = pipeline.get_llm_stats()
    assert stats["logical_calls"] == 1
    assert stats["cache_hits"] == 1
    assert stats["is_cached"] is True
    assert stats["total_tokens"] == 60
    assert len(stats["call_records"]) == 1
    assert stats["call_records"][0]["cached"] is True

def test_oracle_strategy_and_crag_verdict_recorded():
    pipeline = AdaptiveRAGPipeline(enable_graph=False, enable_rerank=False, enable_web=False)
    pipeline.retriever.clear()
    pipeline.retriever.add_passages([("Gold Doc", "Gold supporting fact evidence.")])

    # Mock classifier and generator to avoid network dependency in unit tests
    pipeline.classifier.classify = MagicMock(return_value=ClassificationOutput(
        complexity=ComplexityLevel.SINGLE_HOP,
        reasoning="Test single hop",
        confidence=1.0
    ))
    pipeline.generator.generate_single_hop = MagicMock(return_value="Paris")
    pipeline.generator.generate_crag = MagicMock(return_value="Paris")

    # Test strategy 'oracle'
    resp_oracle = pipeline.process_query("What is capital?", strategy="oracle", eval_mode=True)
    assert resp_oracle.answer == "Paris"
    assert resp_oracle.crag_verdict == "CORRECT"
    assert len(resp_oracle.retrieved_sources) > 0

    # Test strategy 'adaptive_corrective' records CRAG verdict
    resp_crag = pipeline.process_query("What is capital?", strategy="adaptive_corrective", eval_mode=True)
    assert resp_crag.crag_verdict in ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    assert resp_crag.crag_verdict != "NONE"

