import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.retriever import HybridRetriever, DocumentChunk
from src.pipeline import AdaptiveRAGPipeline, AdaptiveRAGResponse
from src.crag_evaluator import RetrievalEvaluator, EvaluationVerdict
from src.graph_rag import KnowledgeGraphRAG
from src.storage import VaultStorage

def create_isolated_mock_retriever():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.embedding_model_name = "mock-mini-lm"
    retriever.reranker_model_name = "mock-cross-encoder"
    retriever.calibration_offset = 8.5
    retriever.chunks = []
    retriever.embeddings = None
    retriever.bm25 = None
    retriever.tokenized_corpus = []
    retriever._query_emb_cache = {}

    mock_encoder = MagicMock()
    mock_encoder.encode = MagicMock(return_value=np.ones(8) / np.linalg.norm(np.ones(8)))
    retriever.encoder = mock_encoder

    mock_reranker = MagicMock()
    mock_reranker.predict = MagicMock(return_value=np.array([1.5, 0.5, -0.5]))
    retriever.reranker = mock_reranker

    retriever.graph_rag = KnowledgeGraphRAG()
    retriever.storage = MagicMock()
    from src.multimodal_ingest import MultiModalIngestor
    retriever.ingestor = MultiModalIngestor()
    return retriever

def test_edge_case_empty_query():
    retriever = create_isolated_mock_retriever()
    pipeline = AdaptiveRAGPipeline(retriever=retriever)
    res = pipeline.process_query("")
    assert res.query == ""
    assert "Input Validation" in res.route_name
    assert "valid question" in res.answer

def test_edge_case_whitespace_query():
    retriever = create_isolated_mock_retriever()
    pipeline = AdaptiveRAGPipeline(retriever=retriever)
    res = pipeline.process_query("   \t\n  ")
    assert res.query == ""
    assert "Input Validation" in res.route_name

def test_edge_case_retriever_empty_vault():
    retriever = create_isolated_mock_retriever()
    results = retriever.search("sample query", top_k=5)
    assert results == []

def test_edge_case_retriever_zero_or_negative_top_k():
    retriever = create_isolated_mock_retriever()
    # Add fake chunks
    chunks = [DocumentChunk("c1", "doc1", "Text 1"), DocumentChunk("c2", "doc2", "Text 2")]
    retriever.chunks = chunks
    retriever.embeddings = np.ones((2, 8))
    retriever.bm25 = MagicMock()
    
    assert retriever.search("query", top_k=0) == []
    assert retriever.search("query", top_k=-3) == []

def test_edge_case_retriever_unmatched_allowed_docs():
    retriever = create_isolated_mock_retriever()
    chunks = [DocumentChunk("c1", "doc_alpha", "Content alpha")]
    retriever.chunks = chunks
    retriever.embeddings = np.ones((1, 8))
    retriever.bm25 = MagicMock()
    retriever.bm25.get_scores = MagicMock(return_value=np.array([1.0]))
    
    results = retriever.search("query", allowed_docs=["non_existent_doc.pdf"])
    assert results == []

def test_edge_case_crag_evaluator_empty_chunks():
    evaluator = RetrievalEvaluator()
    res = evaluator.evaluate("some query", [])
    assert res.verdict == EvaluationVerdict.INCORRECT
    assert res.confidence_score == 0.0
    assert res.web_search_needed is True
    assert res.refined_passages == []

def test_edge_case_graph_rag_empty_graph():
    graph_rag = KnowledgeGraphRAG()
    res = graph_rag.query_graph("Who founded Apple?")
    assert res["has_graph_evidence"] is False
    assert res["evidence_lines"] == []

def test_edge_case_graph_rag_disconnected_query():
    graph_rag = KnowledgeGraphRAG()
    graph_rag.add_triple("Solaris", "RUNS_ON", "AWS", doc_name="doc1")
    res = graph_rag.query_graph("quantum physics entanglement")
    assert res["has_graph_evidence"] is False
    assert res["evidence_lines"] == []

def test_edge_case_storage_malformed_triples(tmp_path):
    # Test that save_batch handles malformed or 2-element triples gracefully with real SQLite & FAISS
    storage = VaultStorage(data_dir=tmp_path)
    
    malformed_triples = [("A", "REL"), None, ("X", "Y", "Z", "EXTRA"), ("ValidSubject", "REL", "ValidObject")]
    chunks = [DocumentChunk("c1", "doc1", "text content")]
    embs = np.ones((1, 384), dtype="float32")
    
    try:
        storage.save_batch("doc1", "text", chunks, embs, triples=malformed_triples)
        triples = storage.load_all_triples()
        # Exactly the 1 valid triple should be preserved
        assert len(triples) == 1
        assert triples[0][0] == "ValidSubject"
    except Exception as e:
        pytest.fail(f"save_batch raised unexpected exception on malformed triples: {e}")
