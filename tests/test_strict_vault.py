import pytest
from unittest.mock import MagicMock
from src.pipeline import AdaptiveRAGPipeline, AdaptiveRAGResponse
from src.classifier import ComplexityLevel, ClassificationOutput
from src.retriever import DocumentChunk
from src.crag_evaluator import EvaluationVerdict, CRAGEvaluationResult

def create_mocked_pipeline():
    mock_classifier = MagicMock()
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    mock_crag = MagicMock()
    mock_web = MagicMock()

    mock_generator.generate_direct.return_value = "Mocked direct answer."
    mock_generator.generate_crag.return_value = "Mocked CRAG answer."
    mock_generator.generate_multi_hop.return_value = "Mocked multi-hop answer."

    pipeline = AdaptiveRAGPipeline(
        classifier=mock_classifier,
        retriever=mock_retriever,
        generator=mock_generator,
        crag_evaluator=mock_crag,
        web_search=mock_web,
        enable_graph=True,
        enable_rerank=True,
        enable_web=True
    )
    return pipeline, mock_classifier, mock_retriever, mock_generator, mock_crag, mock_web

def test_strict_vault_never_calls_web_single_hop():
    pipeline, mock_classifier, mock_retriever, mock_generator, mock_crag, mock_web = create_mocked_pipeline()

    # Route: SINGLE_HOP
    mock_classifier.classify.return_value = ClassificationOutput(
        complexity=ComplexityLevel.SINGLE_HOP,
        reasoning="Single hop query."
    )
    mock_retriever.count.return_value = 5
    mock_chunk = DocumentChunk("c1", "doc1.txt", "Some text")
    mock_retriever.search.return_value = [(mock_chunk, 0.10)]
    mock_retriever.graph_rag.query_graph.return_value = {"evidence_lines": []}

    # CRAG verdict: INCORRECT (which would normally trigger web search)
    mock_crag.evaluate.return_value = CRAGEvaluationResult(
        verdict=EvaluationVerdict.INCORRECT,
        confidence_score=0.10,
        refined_passages=[],
        action_summary="Confidence low. Web needed.",
        web_search_needed=True
    )

    # Execute in strict_vault mode
    res: AdaptiveRAGResponse = pipeline.process_query("What is the penalty?", search_mode="strict_vault")

    # Assert web search was NEVER called!
    mock_web.search.assert_not_called()
    assert res.web_sources == []
    assert res.metrics.web_searches_triggered == 0
    assert "Strict Grounding • Web Prohibited" in res.route_name or "Strict" in res.route_name

def test_strict_vault_never_calls_web_multi_hop():
    pipeline, mock_classifier, mock_retriever, mock_generator, mock_crag, mock_web = create_mocked_pipeline()

    # Route: MULTI_HOP with 2 sub-queries
    mock_classifier.classify.return_value = ClassificationOutput(
        complexity=ComplexityLevel.MULTI_HOP,
        reasoning="Multi hop comparison.",
        sub_queries=["Sub question 1", "Sub question 2"]
    )
    mock_retriever.count.return_value = 5
    mock_chunk = DocumentChunk("c1", "doc1.txt", "Some text")
    mock_retriever.search.return_value = [(mock_chunk, 0.15)]
    mock_retriever.graph_rag.query_graph.return_value = {"evidence_lines": []}

    # CRAG verdict: INCORRECT for sub-queries
    mock_crag.evaluate.return_value = CRAGEvaluationResult(
        verdict=EvaluationVerdict.INCORRECT,
        confidence_score=0.15,
        refined_passages=[],
        action_summary="Confidence low.",
        web_search_needed=True
    )

    # Execute in strict_vault mode
    res: AdaptiveRAGResponse = pipeline.process_query("Compare X and Y", search_mode="strict_vault")

    # Assert web search was NEVER called for any sub-query!
    mock_web.search.assert_not_called()
    assert res.web_sources == []
    assert res.metrics.web_searches_triggered == 0
    assert "Strict Vault" in res.route_name

def test_autonomous_mode_calls_web_when_needed():
    pipeline, mock_classifier, mock_retriever, mock_generator, mock_crag, mock_web = create_mocked_pipeline()

    # Route: SINGLE_HOP with INCORRECT verdict
    mock_classifier.classify.return_value = ClassificationOutput(
        complexity=ComplexityLevel.SINGLE_HOP,
        reasoning="Single hop query."
    )
    mock_retriever.count.return_value = 5
    mock_chunk = DocumentChunk("c1", "doc1.txt", "Some text")
    mock_retriever.search.return_value = [(mock_chunk, 0.10)]
    mock_retriever.graph_rag.query_graph.return_value = {"evidence_lines": []}

    mock_crag.evaluate.return_value = CRAGEvaluationResult(
        verdict=EvaluationVerdict.INCORRECT,
        confidence_score=0.10,
        refined_passages=[],
        action_summary="Web needed.",
        web_search_needed=True
    )
    mock_web.search.return_value = [{"title": "Web Title", "snippet": "Web text", "url": "https://example.com"}]

    # In autonomous mode, web search MUST be triggered
    res: AdaptiveRAGResponse = pipeline.process_query("Query needing web fallback", search_mode="autonomous")

    mock_web.search.assert_called_once_with("Query needing web fallback")
    assert len(res.web_sources) == 1
    assert res.metrics.web_searches_triggered == 1

def test_graph_evidence_ablation_flag():
    pipeline, mock_classifier, mock_retriever, mock_generator, mock_crag, mock_web = create_mocked_pipeline()
    # Ablate graph evidence: enable_graph = False
    pipeline.enable_graph = False

    mock_classifier.classify.return_value = ClassificationOutput(
        complexity=ComplexityLevel.SINGLE_HOP,
        reasoning="Single hop."
    )
    mock_retriever.count.return_value = 5
    mock_retriever.search.return_value = []
    mock_crag.evaluate.return_value = CRAGEvaluationResult(
        verdict=EvaluationVerdict.CORRECT,
        confidence_score=0.9,
        refined_passages=["Refined doc"],
        action_summary="OK"
    )

    res: AdaptiveRAGResponse = pipeline.process_query("Any entity query")
    # Graph query must NOT have been called and evidence must be empty
    mock_retriever.graph_rag.query_graph.assert_not_called()
    assert res.graph_evidence == []
