import json
import pytest
from unittest.mock import MagicMock
from src.classifier import AdaptiveClassifier, ComplexityLevel, ClassificationOutput

def create_mock_classifier():
    classifier = AdaptiveClassifier.__new__(AdaptiveClassifier)
    classifier.api_key = "dummy_key"
    classifier.model = "mock-model"
    classifier.fallback_model = "mock-fallback"
    classifier.client = MagicMock()
    classifier.llm_calls = 0
    classifier.prompt_tokens = 0
    classifier.completion_tokens = 0
    classifier.cache = None
    classifier.local_clf = None
    classifier._embedder = None
    return classifier

def test_classifier_empty_query():
    classifier = create_mock_classifier()
    res = classifier.classify("   ")
    assert res.complexity == ComplexityLevel.ZERO_RETRIEVAL
    assert res.confidence == 1.0

def test_classifier_heuristic_fallback_multi_hop():
    classifier = create_mock_classifier()
    # Comparative questions
    res1 = classifier._heuristic_fallback("Compare the revenue of Apple and Microsoft.", 10.0)
    assert res1.complexity == ComplexityLevel.MULTI_HOP

    res2 = classifier._heuristic_fallback("What is the difference between TCP and UDP?", 10.0)
    assert res2.complexity == ComplexityLevel.MULTI_HOP

    # Bridge reasoning
    res3 = classifier._heuristic_fallback("Where was the father of John's mother born?", 10.0)
    assert res3.complexity == ComplexityLevel.MULTI_HOP

def test_classifier_heuristic_fallback_zero_retrieval():
    classifier = create_mock_classifier()
    greetings = ["hi", "hello", "good morning", "how are you", "thank you"]
    for g in greetings:
        res = classifier._heuristic_fallback(g, 5.0)
        assert res.complexity == ComplexityLevel.ZERO_RETRIEVAL

def test_classifier_heuristic_fallback_single_hop():
    classifier = create_mock_classifier()
    res = classifier._heuristic_fallback("What year was the Eiffel Tower built?", 10.0)
    assert res.complexity == ComplexityLevel.SINGLE_HOP

def test_classifier_mock_groq_success():
    classifier = create_mock_classifier()
    # Mock chat.completions.create response
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "complexity": "MULTI_HOP",
        "reasoning": "Comparison of two distinct companies.",
        "sub_queries": ["What is Company A's revenue?", "What is Company B's revenue?"],
        "confidence": 0.98
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    classifier.client.chat.completions.create.return_value = mock_resp

    res = classifier.classify("Did Company A have higher revenue than Company B in 2023?")
    assert res.complexity == ComplexityLevel.MULTI_HOP
    assert len(res.sub_queries) == 2
    assert res.confidence == 0.98
    assert res.llm_calls == 1

def test_classifier_mock_groq_failure_triggers_fallback():
    classifier = create_mock_classifier()
    # Mock chat.completions.create to raise non-429 exception
    classifier.client.chat.completions.create.side_effect = RuntimeError("API service connection failed")

    res = classifier.classify("Compare the architectures of Transformer and RNN.")
    # Must gracefully fall back to heuristic without crashing
    assert res.complexity == ComplexityLevel.MULTI_HOP
    assert "Heuristic fallback" in res.reasoning
