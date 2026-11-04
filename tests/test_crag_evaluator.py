import pytest
from src.crag_evaluator import RetrievalEvaluator, EvaluationVerdict, CRAGEvaluationResult
from src.retriever import DocumentChunk

def make_chunk(doc_name: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_name}_chunk_1",
        doc_name=doc_name,
        text=text,
        page=page
    )

def test_crag_empty_chunks():
    evaluator = RetrievalEvaluator(upper_threshold=0.60, lower_threshold=0.25)
    res = evaluator.evaluate("Any question", [])
    assert res.verdict == EvaluationVerdict.INCORRECT
    assert res.confidence_score == 0.0
    assert res.web_search_needed is True
    assert len(res.refined_passages) == 0

def test_crag_verdict_correct():
    evaluator = RetrievalEvaluator(upper_threshold=0.60, lower_threshold=0.25)
    chunks = [(make_chunk("doc1.txt", "High relevance answer text"), 0.85)]
    res = evaluator.evaluate("What is the answer?", chunks)
    assert res.verdict == EvaluationVerdict.CORRECT
    assert res.confidence_score == 0.85
    assert res.web_search_needed is False
    assert len(res.refined_passages) == 1
    assert "doc1.txt" in res.refined_passages[0]

def test_crag_verdict_ambiguous():
    evaluator = RetrievalEvaluator(upper_threshold=0.60, lower_threshold=0.25)
    chunks = [(make_chunk("doc1.txt", "Moderate match text"), 0.45)]
    res = evaluator.evaluate("What is the answer?", chunks)
    assert res.verdict == EvaluationVerdict.AMBIGUOUS
    assert res.confidence_score == 0.45
    assert res.web_search_needed is True

def test_crag_verdict_incorrect():
    evaluator = RetrievalEvaluator(upper_threshold=0.60, lower_threshold=0.25)
    chunks = [(make_chunk("doc1.txt", "Completely irrelevant text"), 0.12)]
    res = evaluator.evaluate("What is the answer?", chunks)
    assert res.verdict == EvaluationVerdict.INCORRECT
    assert res.confidence_score == 0.12
    assert res.web_search_needed is True

def test_crag_custom_thresholds():
    # Test that configured thresholds are strictly respected
    evaluator = RetrievalEvaluator(upper_threshold=0.80, lower_threshold=0.40)
    chunks = [(make_chunk("doc1.txt", "Text"), 0.70)]
    res = evaluator.evaluate("Query", chunks)
    # 0.70 is below 0.80 but >= 0.40 -> AMBIGUOUS
    assert res.verdict == EvaluationVerdict.AMBIGUOUS
    assert res.web_search_needed is True

def test_crag_no_keyword_override():
    # Previously, words like "course", "notes", "unit" forced CORRECT at >= 0.15.
    # Now, verdict must depend solely on thresholds!
    evaluator = RetrievalEvaluator(upper_threshold=0.60, lower_threshold=0.25)
    chunks = [(make_chunk("notes.txt", "Lecture notes content"), 0.18)]
    # Query with old keyword "course notes"
    res = evaluator.evaluate("What is in the course notes unit 1?", chunks)
    # Score 0.18 is below lower_threshold 0.25 -> INCORRECT, NOT CORRECT!
    assert res.verdict == EvaluationVerdict.INCORRECT
    assert res.web_search_needed is True
