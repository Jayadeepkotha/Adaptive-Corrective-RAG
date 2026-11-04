import time
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.pipeline import AdaptiveRAGPipeline, AdaptiveRAGResponse
from src.retriever import HybridRetriever
from src.config import settings

app = FastAPI(
    title="Confidence-RAG Multi-Hop Intelligence API",
    description=(
        "Production-grade, complexity-adaptive cognitive retrieval and knowledge graph API. "
        "Integrates NAACL 2024 Adaptive Routing, Two-Stage Cross-Encoder Reranking, "
        "CRAG Automated Self-Correction, and In-Memory Entity Knowledge Graphs."
    ),
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Enterprise Pipeline Instance
retriever = HybridRetriever()
pipeline = AdaptiveRAGPipeline(retriever=retriever)

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the statutory penalties for non-compliance with data encryption standards?")
    search_mode: str = Field(default="autonomous", description="Options: 'autonomous', 'strict_vault', 'web_augmented'")
    selected_docs: Optional[List[str]] = Field(default=None, description="Optional list of document names to restrict retrieval to")

class QueryResponse(BaseModel):
    query: str
    complexity: str
    route_name: str
    crag_verdict: Optional[str]
    crag_confidence: float
    answer: str
    retrieved_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    graph_evidence: List[str]
    latency_ms: float

@app.get("/api/v1/health", tags=["System Telemetry"])
def health_check():
    return {
        "status": "healthy",
        "platform": "Confidence-RAG Multi-Hop Cognitive Engine",
        "version": "2.4.0",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": getattr(settings, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "total_chunks_indexed": retriever.count(),
        "knowledge_graph_nodes": retriever.graph_rag.graph.number_of_nodes(),
        "knowledge_graph_edges": retriever.graph_rag.graph.number_of_edges(),
        "search_modes": ["autonomous", "strict_vault", "web_augmented"]
    }

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Cognitive Retrieval"])
def process_query(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    res: AdaptiveRAGResponse = pipeline.process_query(
        query=payload.query.strip(),
        search_mode=payload.search_mode,
        selected_docs=payload.selected_docs
    )

    return QueryResponse(
        query=res.query,
        complexity=res.complexity.value,
        route_name=res.route_name,
        crag_verdict=res.crag_verdict,
        crag_confidence=res.crag_confidence,
        answer=res.answer,
        retrieved_sources=res.retrieved_sources,
        web_sources=res.web_sources,
        graph_evidence=res.graph_evidence,
        latency_ms=res.metrics.total_latency_ms
    )

@app.post("/api/v1/query/stream", tags=["Cognitive Retrieval"])
def process_query_stream_endpoint(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def event_generator():
        for chunk in pipeline.process_query_stream(
            query=payload.query.strip(),
            search_mode=payload.search_mode,
            selected_docs=payload.selected_docs
        ):
            yield f"data: {chunk.strip()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/v1/query/compare", tags=["Cognitive Retrieval"])
def process_query_compare_endpoint(payload: QueryRequest):
    """
    Live Head-to-Head Benchmark:
    Executes standard Naive RAG vs. Confidence-RAG (Adaptive-CRAG + GraphRAG)
    simultaneously to demonstrate architectural superiority and efficiency gains.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    clean_q = payload.query.strip()
    # 1. Run Traditional Naive RAG
    naive_result = pipeline.process_naive_rag(clean_q)

    # 2. Run Confidence-RAG (Adaptive-CRAG + GraphRAG)
    adaptive_res = pipeline.process_query(
        query=clean_q,
        search_mode=payload.search_mode,
        selected_docs=payload.selected_docs
    )

    adaptive_result = {
        "architecture": "Confidence-RAG (Adaptive-CRAG + GraphRAG)",
        "answer": adaptive_res.answer,
        "complexity": adaptive_res.complexity.value,
        "route_name": adaptive_res.route_name,
        "crag_verdict": adaptive_res.crag_verdict,
        "crag_confidence": adaptive_res.crag_confidence,
        "graph_evidence": adaptive_res.graph_evidence,
        "retrieved_sources": adaptive_res.retrieved_sources,
        "web_sources": adaptive_res.web_sources,
        "latency_ms": adaptive_res.metrics.total_latency_ms
    }

    latency_diff = round(naive_result["latency_ms"] - adaptive_result["latency_ms"], 1)

    return {
        "query": clean_q,
        "naive_rag": naive_result,
        "adaptive_rag": adaptive_result,
        "telemetry_delta": {
            "latency_savings_ms": latency_diff,
            "complexity_level": adaptive_res.complexity.value,
            "crag_verdict": adaptive_res.crag_verdict,
            "has_graph_evidence": len(adaptive_res.graph_evidence) > 0,
            "cross_encoder_filtered": len(adaptive_res.retrieved_sources)
        }
    }

@app.post("/api/v1/upload", tags=["Vault Management"])
async def upload_document(file: UploadFile = File(...)):
    fname = file.filename or "uploaded_file.txt"
    content = await file.read()
    suffix = Path(fname).suffix or ".txt"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    count_before = retriever.count()
    try:
        chunks_added = retriever.add_document(tmp_path, doc_name=fname)
    except Exception as e:
        logger.error(f"Failed to ingest document '{fname}': {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process document '{fname}': {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return {
        "filename": fname,
        "file_type": suffix.replace(".", "").lower(),
        "chunks_indexed": chunks_added,
        "total_vault_chunks": retriever.count(),
        "knowledge_graph_nodes": retriever.graph_rag.graph.number_of_nodes(),
        "status": "Successfully indexed into Hybrid Multi-Modal & Knowledge Graph Store"
    }

@app.post("/api/v1/upload_batch", tags=["Vault Management"])
async def upload_batch_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        fname = file.filename or "uploaded_file.txt"
        content = await file.read()
        suffix = Path(fname).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            chunks_added = retriever.add_document(tmp_path, doc_name=fname)
            results.append({"filename": fname, "chunks": chunks_added})
        except Exception as e:
            logger.warning(f"Batch file '{fname}' failed: {e}")
            results.append({"filename": fname, "chunks": 0, "error": str(e)})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return {
        "uploaded_count": len(results),
        "results": results,
        "total_vault_chunks": retriever.count(),
        "knowledge_graph_nodes": retriever.graph_rag.graph.number_of_nodes(),
        "knowledge_graph_edges": retriever.graph_rag.graph.number_of_edges(),
        "status": f"Successfully indexed {len(results)} documents into Knowledge Graph Store"
    }

@app.get("/api/v1/vault", tags=["Vault Management"])
def list_vault():
    docs = {}
    for c in retriever.chunks:
        docs[c.doc_name] = docs.get(c.doc_name, 0) + 1
    return {
        "total_chunks": retriever.count(),
        "documents": [{"name": k, "chunks": v} for k, v in docs.items()],
        "knowledge_graph_nodes": retriever.graph_rag.graph.number_of_nodes(),
        "knowledge_graph_edges": retriever.graph_rag.graph.number_of_edges()
    }

@app.delete("/api/v1/vault", tags=["Vault Management"])
def clear_vault():
    retriever.clear_vault()
    return {"status": "Vault storage and knowledge graph cleared successfully"}

@app.get("/api/v1/graph", tags=["Knowledge Graph"])
def get_knowledge_graph():
    return {
        "nodes": list(retriever.graph_rag.graph.nodes()),
        "edges_count": retriever.graph_rag.graph.number_of_edges(),
        "triples": [t.to_dict() for t in retriever.graph_rag.triples[:300]]
    }

# Mount Modern React 18 + Vite Frontend
dist_path = Path(__file__).resolve().parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        return FileResponse(str(dist_path / "index.html"))

    @app.get("/favicon.svg", include_in_schema=False)
    def serve_favicon():
        favicon_file = dist_path / "favicon.svg"
        if favicon_file.exists():
            return FileResponse(str(favicon_file))
        return FileResponse(str(dist_path / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

