import pytest
from unittest.mock import MagicMock
import numpy as np
from src.retriever import HybridRetriever, DocumentChunk

def create_mock_retriever(calibration_offset=8.5):
    # Mock SentenceTransformer and CrossEncoder to avoid slow disk/model loading
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.embedding_model_name = "mock-mini-lm"
    retriever.reranker_model_name = "mock-cross-encoder"
    retriever.calibration_offset = calibration_offset
    retriever.chunks = []
    retriever.embeddings = None
    retriever.bm25 = None
    retriever.tokenized_corpus = []

    mock_encoder = MagicMock()
    # Mock encoding: returns orthogonal or simple 1D vector embeddings
    def mock_encode(texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.ones(8) / np.linalg.norm(np.ones(8))
        embs = []
        for i, t in enumerate(texts):
            vec = np.zeros(8)
            vec[i % 8] = 1.0
            embs.append(vec)
        return np.array(embs)

    mock_encoder.encode = mock_encode
    retriever.encoder = mock_encoder

    mock_reranker = MagicMock()
    # predict returns raw logits
    mock_reranker.predict = MagicMock(return_value=np.array([2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0]))
    retriever.reranker = mock_reranker

    from src.graph_rag import KnowledgeGraphRAG
    from src.multimodal_ingest import MultiModalIngestor
    retriever.graph_rag = KnowledgeGraphRAG()
    retriever.ingestor = MultiModalIngestor()
    retriever.storage = MagicMock()
    retriever._query_emb_cache = {}

    return retriever

def test_retriever_top_k():
    retriever = create_mock_retriever()
    chunks = [
        DocumentChunk(f"c_{i}", f"doc_{i}", f"Text chunk content {i}")
        for i in range(10)
    ]
    retriever._index_chunks(chunks)

    # Mock reranker predict to return descending logits
    retriever.reranker.predict = MagicMock(return_value=np.linspace(2.0, -2.0, 10))

    results = retriever.search("test query", top_k=3, candidate_k=10, enable_rerank=True)
    assert len(results) == 3
    # Scores must be sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)

def test_retriever_per_doc_limits():
    retriever = create_mock_retriever()
    # Create 6 chunks from the SAME document 'doc_monopoly' and 3 from 'doc_other'
    chunks = [
        DocumentChunk(f"c_{i}", "doc_monopoly", f"Text chunk {i} of doc_monopoly")
        for i in range(6)
    ] + [
        DocumentChunk(f"c_other_{i}", "doc_other", f"Text chunk {i} of other")
        for i in range(3)
    ]
    retriever._index_chunks(chunks)

    # Let the first 5 chunks of doc_monopoly have highest logits
    retriever.reranker.predict = MagicMock(
        return_value=np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])
    )

    # Set max_per_doc = 2
    results = retriever.search("query", top_k=4, candidate_k=9, max_per_doc=2, enable_rerank=True)

    doc_counts = {}
    for chunk, _ in results:
        doc_counts[chunk.doc_name] = doc_counts.get(chunk.doc_name, 0) + 1

    # 'doc_monopoly' must NOT exceed 2 chunks in top results
    assert doc_counts.get("doc_monopoly", 0) <= 2
    # Total results should still reach top_k=4 by using other docs or backfilling
    assert len(results) == 4

def test_retriever_calibration_offset():
    # Test that configured calibration offset directly shifts the sigmoid score
    retriever = create_mock_retriever(calibration_offset=1.0)
    chunk = DocumentChunk("c_1", "doc_1", "Sample text")
    retriever._index_chunks([chunk, DocumentChunk("c_2", "doc_2", "Sample text 2")])

    # raw logit = 0.0 -> score with offset 1.0 is 1 / (1 + exp(-1.0))
    retriever.reranker.predict = MagicMock(return_value=np.array([0.0, -1.0]))
    expected_score = round(1.0 / (1.0 + np.exp(-1.0)), 4)

    results = retriever.search("query", top_k=1, enable_rerank=True)
    assert len(results) == 1
    assert results[0][1] == expected_score

def test_retriever_without_rerank():
    retriever = create_mock_retriever()
    chunks = [
        DocumentChunk(f"c_{i}", f"doc_{i}", f"Text chunk {i}")
        for i in range(5)
    ]
    retriever._index_chunks(chunks)

    results = retriever.search("query", top_k=3, enable_rerank=False)
    assert len(results) == 3
    # CrossEncoder.predict must NOT have been called when enable_rerank=False
    retriever.reranker.predict.assert_not_called()
