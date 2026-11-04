import time
import json
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from src.classifier import AdaptiveClassifier, ComplexityLevel, ClassificationOutput
from src.retriever import HybridRetriever, DocumentChunk
from src.generator import AnswerGenerator
from src.crag_evaluator import RetrievalEvaluator, EvaluationVerdict, CRAGEvaluationResult
from src.web_search import WebSearchFallback

class ExecutionMetrics(BaseModel):
    classification_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieval_steps: int = 0
    chunks_retrieved: int = 0
    web_searches_triggered: int = 0

class AdaptiveRAGResponse(BaseModel):
    query: str
    complexity: ComplexityLevel
    route_name: str
    classifier_reasoning: str
    sub_queries: List[str] = Field(default_factory=list)
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list)
    web_sources: List[Dict[str, Any]] = Field(default_factory=list)
    graph_evidence: List[str] = Field(default_factory=list)
    crag_verdict: Optional[str] = None
    crag_confidence: float = 0.0
    crag_action: str = ""
    answer: str
    metrics: ExecutionMetrics

class AdaptiveRAGPipeline:
    """
    Unified Adaptive-CRAG + GraphRAG Pipeline:
    1. Query-Complexity Classification (NAACL 2024 Adaptive-RAG)
    2. Two-Stage Hybrid (Bi-Encoder + BM25) + Cross-Encoder Precision Reranking
    3. Knowledge Graph Entity Traversal (GraphRAG for corpus aggregation)
    4. Corrective Retrieval Evaluation & Automated Web Search Fallback (CRAG, Yan et al. 2024)
    """

    def __init__(
        self,
        classifier: Optional[AdaptiveClassifier] = None,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        crag_evaluator: Optional[RetrievalEvaluator] = None,
        web_search: Optional[WebSearchFallback] = None,
        enable_graph: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
        enable_web: Optional[bool] = None,
    ):
        self.classifier = classifier or AdaptiveClassifier()
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or AnswerGenerator()
        self.crag_evaluator = crag_evaluator or RetrievalEvaluator(
            upper_threshold=settings.CRAG_UPPER_THRESHOLD,
            lower_threshold=settings.CRAG_LOWER_THRESHOLD
        )
        self.web_search = web_search or WebSearchFallback(max_results=3)
        self.enable_graph = enable_graph if enable_graph is not None else settings.ENABLE_GRAPH_RAG
        if hasattr(self.retriever, "graph_rag") and self.retriever.graph_rag:
            self.retriever.graph_rag.is_enabled = self.enable_graph
        self.enable_rerank = enable_rerank if enable_rerank is not None else settings.ENABLE_RERANK
        self.enable_web = enable_web if enable_web is not None else settings.ENABLE_WEB_SEARCH

    def reset_counters(self):
        if hasattr(self.classifier, "reset_counters"):
            self.classifier.reset_counters()
        if hasattr(self.generator, "reset_counters"):
            self.generator.reset_counters()

    def get_llm_stats(self) -> Dict[str, Any]:
        c_calls = getattr(self.classifier, "logical_calls", getattr(self.classifier, "llm_calls", 0))
        g_calls = getattr(self.generator, "logical_calls", getattr(self.generator, "llm_calls", 0))
        c_retries = getattr(self.classifier, "retry_calls", 0)
        g_retries = getattr(self.generator, "retry_calls", 0)
        c_hits = getattr(self.classifier, "cache_hits", 0)
        g_hits = getattr(self.generator, "cache_hits", 0)
        c_prompt = getattr(self.classifier, "prompt_tokens", 0)
        g_prompt = getattr(self.generator, "prompt_tokens", 0)
        c_comp = getattr(self.classifier, "completion_tokens", 0)
        g_comp = getattr(self.generator, "completion_tokens", 0)

        c_records = getattr(self.classifier, "call_records", [])
        g_records = getattr(self.generator, "call_records", [])
        call_records = list(c_records) + list(g_records)

        tot_logical = c_calls + g_calls
        tot_hits = c_hits + g_hits
        is_cached = (tot_hits >= tot_logical) if tot_logical > 0 else True

        return {
            "llm_calls": tot_logical,
            "logical_calls": tot_logical,
            "retry_calls": c_retries + g_retries,
            "cache_hits": tot_hits,
            "is_cached": is_cached,
            "classifier_calls": c_calls,
            "generator_calls": g_calls,
            "prompt_tokens": c_prompt + g_prompt,
            "completion_tokens": c_comp + g_comp,
            "total_tokens": c_prompt + g_prompt + c_comp + g_comp,
            "call_records": call_records
        }

    def process_query(
        self,
        query: str,
        search_mode: str = "autonomous",
        strategy: str = "adaptive_corrective",
        eval_mode: bool = False,
        enable_rerank: Optional[bool] = None,
        enable_graph: Optional[bool] = None,
        enable_web: Optional[bool] = None,
        selected_docs: Optional[List[str]] = None
    ) -> AdaptiveRAGResponse:
        total_start = time.perf_counter()
        clean_query = query.strip()

        eff_rerank = enable_rerank if enable_rerank is not None else self.enable_rerank
        eff_graph = enable_graph if enable_graph is not None else self.enable_graph
        eff_web = enable_web if enable_web is not None else self.enable_web

        # Edge Case Guard 1: Empty or whitespace query
        if not clean_query:
            return AdaptiveRAGResponse(
                query="",
                complexity=ComplexityLevel.ZERO_RETRIEVAL,
                route_name="Zero-Retrieval (Input Validation)",
                classifier_reasoning="Empty or whitespace query provided.",
                sub_queries=[],
                retrieved_sources=[],
                web_sources=[],
                graph_evidence=[],
                answer="Please enter a valid question to get started.",
                metrics=ExecutionMetrics()
            )

        logger.info(f"Processing query: '{clean_query[:70]}' (strategy='{strategy}', eval_mode={eval_mode})...")

        # Query Graph-RAG for any entity-relationship connections if enabled
        graph_evidence = []
        if eff_graph:
            graph_data = self.retriever.graph_rag.query_graph(clean_query, enabled=True)
            graph_evidence = graph_data.get("evidence_lines", [])

        # -------------------------------------------------------------
        # STRATEGY 1: No-Retrieval Baseline (Direct Parametric LLM)
        # -------------------------------------------------------------
        if strategy == "no_retrieval":
            class_res = ClassificationOutput(
                complexity=ComplexityLevel.ZERO_RETRIEVAL,
                reasoning="No-retrieval baseline: direct parametric generation without search.",
                confidence=1.0,
                latency_ms=0.0
            )
            gen_start = time.perf_counter()
            answer = self.generator.generate_direct(clean_query, eval_mode=eval_mode)
            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return AdaptiveRAGResponse(
                query=clean_query,
                complexity=class_res.complexity,
                route_name="No-Retrieval Baseline (Direct LLM)",
                classifier_reasoning=class_res.reasoning,
                sub_queries=[],
                retrieved_sources=[],
                web_sources=[],
                graph_evidence=[],
                answer=answer,
                metrics=ExecutionMetrics(
                    classification_ms=0.0,
                    retrieval_ms=0.0,
                    generation_ms=gen_ms,
                    total_latency_ms=total_latency,
                    retrieval_steps=0,
                    chunks_retrieved=0,
                    web_searches_triggered=0
                )
            )

        # -------------------------------------------------------------
        # STRATEGY 2: Single-Step Baseline (Fixed 1-Hop + Rerank, No Router, No CRAG)
        # -------------------------------------------------------------
        if strategy == "single_step":
            class_res = ClassificationOutput(
                complexity=ComplexityLevel.SINGLE_HOP,
                reasoning="Fixed single-step retrieval baseline.",
                confidence=1.0,
                latency_ms=0.0
            )
            ret_start = time.perf_counter()
            chunks = self.retriever.search(
                clean_query,
                top_k=settings.TOP_K,
                enable_rerank=eff_rerank,
                allowed_docs=selected_docs
            )
            ret_ms = round((time.perf_counter() - ret_start) * 1000, 2)
            retrieved_sources = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "score": score,
                    "initial_score": getattr(c, "initial_score", score),
                    "snippet": c.text[:120] + "..."
                }
                for c, score in chunks
            ]
            gen_start = time.perf_counter()
            answer = self.generator.generate_single_hop(clean_query, chunks, eval_mode=eval_mode)
            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return AdaptiveRAGResponse(
                query=clean_query,
                complexity=class_res.complexity,
                route_name="Single-Step Baseline (Fixed 1-Hop + Rerank)",
                classifier_reasoning=class_res.reasoning,
                sub_queries=[],
                retrieved_sources=retrieved_sources,
                web_sources=[],
                graph_evidence=[],
                answer=answer,
                metrics=ExecutionMetrics(
                    classification_ms=0.0,
                    retrieval_ms=ret_ms,
                    generation_ms=gen_ms,
                    total_latency_ms=total_latency,
                    retrieval_steps=1,
                    chunks_retrieved=len(chunks),
                    web_searches_triggered=0
                )
            )

        # -------------------------------------------------------------
        # STRATEGY: Oracle Ceiling (Gold Evidence Context Only)
        # -------------------------------------------------------------
        if strategy == "oracle":
            class_res = ClassificationOutput(
                complexity=ComplexityLevel.SINGLE_HOP,
                reasoning="Oracle ceiling: generation conditioned strictly on gold supporting passages.",
                confidence=1.0,
                latency_ms=0.0
            )
            ret_start = time.perf_counter()
            chunks = self.retriever.search(
                clean_query,
                top_k=max(settings.TOP_K, 10),
                enable_rerank=False,
                allowed_docs=selected_docs
            )
            ret_ms = round((time.perf_counter() - ret_start) * 1000, 2)
            retrieved_sources = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "score": score,
                    "initial_score": getattr(c, "initial_score", score),
                    "snippet": c.text[:120] + "..."
                }
                for c, score in chunks
            ]
            gen_start = time.perf_counter()
            answer = self.generator.generate_single_hop(clean_query, chunks, eval_mode=eval_mode)
            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return AdaptiveRAGResponse(
                query=clean_query,
                complexity=class_res.complexity,
                route_name="Oracle Reader Ceiling (Gold Context Only)",
                classifier_reasoning=class_res.reasoning,
                sub_queries=[],
                retrieved_sources=retrieved_sources,
                web_sources=[],
                graph_evidence=[],
                crag_verdict="CORRECT",
                crag_confidence=1.0,
                crag_action="Oracle gold evidence provided directly.",
                answer=answer,
                metrics=ExecutionMetrics(
                    classification_ms=0.0,
                    retrieval_ms=ret_ms,
                    generation_ms=gen_ms,
                    total_latency_ms=total_latency,
                    retrieval_steps=1,
                    chunks_retrieved=len(chunks),
                    web_searches_triggered=0
                )
            )

        # -------------------------------------------------------------
        # STRATEGY 3: Multi-Step Baseline (Always Decompose & Multi-Retrieve, No Router, No CRAG)
        # -------------------------------------------------------------
        if strategy == "multi_step":
            class_res = ClassificationOutput(
                complexity=ComplexityLevel.MULTI_HOP,
                reasoning="Fixed multi-step decomposition baseline.",
                confidence=1.0,
                latency_ms=0.0
            )
            sub_queries = self.classifier.decompose_query(clean_query)
            if not sub_queries:
                sub_queries = [clean_query]

            ret_start = time.perf_counter()
            subquery_evidence = []
            retrieved_sources = []
            total_chunks = 0
            for sq in sub_queries:
                sub_chunks = self.retriever.search(sq, top_k=settings.TOP_K, enable_rerank=eff_rerank, allowed_docs=selected_docs)
                total_chunks += len(sub_chunks)
                subquery_evidence.append({"sub_query": sq, "chunks": sub_chunks, "web_results": []})
                for c, score in sub_chunks:
                    retrieved_sources.append({
                        "sub_query": sq,
                        "chunk_id": c.chunk_id,
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "score": score,
                        "initial_score": getattr(c, "initial_score", score),
                        "snippet": c.text[:120] + "..."
                    })
            ret_ms = round((time.perf_counter() - ret_start) * 1000, 2)
            gen_start = time.perf_counter()
            answer = self.generator.generate_multi_hop(
                clean_query,
                subquery_evidence,
                graph_evidence=graph_evidence if eff_graph else None,
                eval_mode=eval_mode
            )
            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return AdaptiveRAGResponse(
                query=clean_query,
                complexity=class_res.complexity,
                route_name="Multi-Step Baseline (Decompose & Retrieve)",
                classifier_reasoning=class_res.reasoning,
                sub_queries=sub_queries,
                retrieved_sources=retrieved_sources,
                web_sources=[],
                graph_evidence=graph_evidence,
                answer=answer,
                metrics=ExecutionMetrics(
                    classification_ms=0.0,
                    retrieval_ms=ret_ms,
                    generation_ms=gen_ms,
                    total_latency_ms=total_latency,
                    retrieval_steps=len(sub_queries),
                    chunks_retrieved=total_chunks,
                    web_searches_triggered=0
                )
            )

        # -------------------------------------------------------------
        # Step 1: Adaptive Complexity Classification (for adaptive & adaptive_corrective)
        # -------------------------------------------------------------
        class_res: ClassificationOutput = self.classifier.classify(clean_query)
        logger.info(f"Complexity: {class_res.complexity.value} (Classification Time: {class_res.latency_ms}ms)")

        # -------------------------------------------------------------
        # STRATEGY 4: Adaptive (Router Only, No Corrective / No CRAG / No Web Fallback)
        # -------------------------------------------------------------
        if strategy == "adaptive":
            if class_res.complexity == ComplexityLevel.ZERO_RETRIEVAL:
                gen_start = time.perf_counter()
                answer = self.generator.generate_direct(clean_query, eval_mode=eval_mode)
                gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                total_latency = round((time.perf_counter() - total_start) * 1000, 2)
                return AdaptiveRAGResponse(
                    query=clean_query,
                    complexity=class_res.complexity,
                    route_name="Adaptive Route (Zero-Retrieval)",
                    classifier_reasoning=class_res.reasoning,
                    sub_queries=[],
                    retrieved_sources=[],
                    web_sources=[],
                    graph_evidence=[],
                    answer=answer,
                    metrics=ExecutionMetrics(
                        classification_ms=class_res.latency_ms,
                        retrieval_ms=0.0,
                        generation_ms=gen_ms,
                        total_latency_ms=total_latency,
                        retrieval_steps=0,
                        chunks_retrieved=0,
                        web_searches_triggered=0
                    )
                )
            elif class_res.complexity == ComplexityLevel.SINGLE_HOP:
                ret_start = time.perf_counter()
                chunks = self.retriever.search(clean_query, top_k=settings.TOP_K, enable_rerank=eff_rerank, allowed_docs=selected_docs)
                ret_ms = round((time.perf_counter() - ret_start) * 1000, 2)
                retrieved_sources = [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "score": score,
                        "initial_score": getattr(c, "initial_score", score),
                        "snippet": c.text[:120] + "..."
                    }
                    for c, score in chunks
                ]
                gen_start = time.perf_counter()
                answer = self.generator.generate_single_hop(clean_query, chunks, eval_mode=eval_mode)
                gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                total_latency = round((time.perf_counter() - total_start) * 1000, 2)
                return AdaptiveRAGResponse(
                    query=clean_query,
                    complexity=class_res.complexity,
                    route_name="Adaptive Route (Single-Hop)",
                    classifier_reasoning=class_res.reasoning,
                    sub_queries=[],
                    retrieved_sources=retrieved_sources,
                    web_sources=[],
                    graph_evidence=[],
                    answer=answer,
                    metrics=ExecutionMetrics(
                        classification_ms=class_res.latency_ms,
                        retrieval_ms=ret_ms,
                        generation_ms=gen_ms,
                        total_latency_ms=total_latency,
                        retrieval_steps=1,
                        chunks_retrieved=len(chunks),
                        web_searches_triggered=0
                    )
                )
            else:  # MULTI_HOP
                raw_sq = class_res.sub_queries or [clean_query]
                sub_queries = list(dict.fromkeys([sq.strip() for sq in raw_sq if sq.strip()])) or [clean_query]
                ret_start = time.perf_counter()
                subquery_evidence = []
                retrieved_sources = []
                total_chunks = 0
                for sq in sub_queries:
                    sub_chunks = self.retriever.search(sq, top_k=settings.TOP_K, enable_rerank=eff_rerank, allowed_docs=selected_docs)
                    total_chunks += len(sub_chunks)
                    subquery_evidence.append({"sub_query": sq, "chunks": sub_chunks, "web_results": []})
                    for c, score in sub_chunks:
                        retrieved_sources.append({
                            "sub_query": sq,
                            "chunk_id": c.chunk_id,
                            "doc_name": c.doc_name,
                            "page": c.page,
                            "score": score,
                            "initial_score": getattr(c, "initial_score", score),
                            "snippet": c.text[:120] + "..."
                        })
                ret_ms = round((time.perf_counter() - ret_start) * 1000, 2)
                gen_start = time.perf_counter()
                answer = self.generator.generate_multi_hop(
                    clean_query,
                    subquery_evidence,
                    graph_evidence=graph_evidence if eff_graph else None,
                    eval_mode=eval_mode
                )
                gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                total_latency = round((time.perf_counter() - total_start) * 1000, 2)
                return AdaptiveRAGResponse(
                    query=clean_query,
                    complexity=class_res.complexity,
                    route_name="Adaptive Route (Multi-Hop)",
                    classifier_reasoning=class_res.reasoning,
                    sub_queries=sub_queries,
                    retrieved_sources=retrieved_sources,
                    web_sources=[],
                    graph_evidence=graph_evidence,
                    answer=answer,
                    metrics=ExecutionMetrics(
                        classification_ms=class_res.latency_ms,
                        retrieval_ms=ret_ms,
                        generation_ms=gen_ms,
                        total_latency_ms=total_latency,
                        retrieval_steps=len(sub_queries),
                        chunks_retrieved=total_chunks,
                        web_searches_triggered=0
                    )
                )

        # -------------------------------------------------------------
        # STRATEGY 5: Adaptive-Corrective (Full System with CRAG & Fallbacks)
        # -------------------------------------------------------------

        # Edge Case Guard: Document queries when index is empty
        if class_res.complexity != ComplexityLevel.ZERO_RETRIEVAL and self.retriever.count() == 0:
            if search_mode != "strict_vault" and eff_web:
                web_results = self.web_search.search(clean_query)
                if web_results:
                    gen_start = time.perf_counter()
                    answer = self.generator.generate_crag(
                        clean_query,
                        refined_passages=[],
                        web_results=web_results,
                        verdict="INCORRECT",
                        eval_mode=eval_mode
                    )
                    gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                    total_latency = round((time.perf_counter() - total_start) * 1000, 2)
                    return AdaptiveRAGResponse(
                        query=clean_query,
                        complexity=class_res.complexity,
                        route_name="System Fallback (Index Empty -> Live Web Intelligence)",
                        classifier_reasoning="Knowledge vault is empty. Executed live web fallback.",
                        sub_queries=[],
                        retrieved_sources=[],
                        web_sources=web_results,
                        graph_evidence=[],
                        crag_verdict="INCORRECT (EMPTY_INDEX)",
                        crag_confidence=0.0,
                        crag_action="Document vault is empty. Synthesized via secure web intelligence.",
                        answer=answer,
                        metrics=ExecutionMetrics(
                            classification_ms=class_res.latency_ms,
                            retrieval_ms=0.0,
                            generation_ms=gen_ms,
                            total_latency_ms=total_latency,
                            retrieval_steps=0,
                            chunks_retrieved=0,
                            web_searches_triggered=1
                        )
                    )

            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return AdaptiveRAGResponse(
                query=clean_query,
                complexity=class_res.complexity,
                route_name=f"{class_res.complexity.value} (Vault Empty)",
                classifier_reasoning=class_res.reasoning,
                sub_queries=class_res.sub_queries,
                retrieved_sources=[],
                web_sources=[],
                graph_evidence=[],
                answer="Document Vault is currently empty.",
                metrics=ExecutionMetrics(
                    classification_ms=class_res.latency_ms,
                    retrieval_ms=0.0,
                    generation_ms=0.0,
                    total_latency_ms=total_latency,
                    retrieval_steps=0,
                    chunks_retrieved=0,
                    web_searches_triggered=0
                )
            )

        retrieval_time_ms = 0.0
        generation_time_ms = 0.0
        retrieval_steps = 0
        chunks_retrieved_count = 0
        web_searches_triggered = 0
        retrieved_sources = []
        web_sources = []
        crag_verdict_str = None
        crag_conf_val = 0.0
        crag_action_str = ""
        answer = ""

        # Step 2: Adaptive Execution Branching
        if class_res.complexity == ComplexityLevel.ZERO_RETRIEVAL:
            route_name = "Zero-Retrieval (Direct LLM)"
            gen_start = time.perf_counter()
            answer = self.generator.generate_direct(clean_query, eval_mode=eval_mode)
            generation_time_ms = round((time.perf_counter() - gen_start) * 1000, 2)

        elif class_res.complexity == ComplexityLevel.SINGLE_HOP:
            ret_start = time.perf_counter()
            chunks = self.retriever.search(
                clean_query,
                top_k=settings.TOP_K,
                enable_rerank=eff_rerank,
                allowed_docs=selected_docs
            )
            retrieval_time_ms = round((time.perf_counter() - ret_start) * 1000, 2)
            retrieval_steps = 1
            chunks_retrieved_count = len(chunks)

            for c, score in chunks:
                retrieved_sources.append({
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "score": score,
                    "initial_score": getattr(c, "initial_score", score),
                    "snippet": c.text[:120] + "..."
                })

            # Step 2b: CRAG Retrieval Quality Evaluation
            crag_res: CRAGEvaluationResult = self.crag_evaluator.evaluate(clean_query, chunks)
            crag_verdict_str = crag_res.verdict.value
            crag_conf_val = crag_res.confidence_score
            crag_action_str = crag_res.action_summary

            # Execute Web Fallback according to Compliance Search Mode
            if search_mode == "strict_vault" or not eff_web:
                web_sources = []
                web_searches_triggered = 0
                if crag_res.verdict == EvaluationVerdict.INCORRECT:
                    route_name = "Confidence Vault (Strict Grounding | Web Prohibited)"
                else:
                    route_name = f"Confidence Vault ({crag_res.verdict.value} | 100% Internal Grounded)"
            elif search_mode == "web_augmented":
                web_sources = self.web_search.search(clean_query)
                web_searches_triggered = 1
                route_name = "Market Intelligence (Hybrid Vault + Live Web Augmented)"
            else:  # autonomous
                if crag_res.web_search_needed:
                    web_sources = self.web_search.search(clean_query)
                    web_searches_triggered = 1

                if crag_res.verdict == EvaluationVerdict.CORRECT:
                    route_name = "Adaptive-CRAG (Verdict: CORRECT -> Refined Document Grounding)"
                elif crag_res.verdict == EvaluationVerdict.AMBIGUOUS:
                    route_name = "Adaptive-CRAG (Verdict: AMBIGUOUS -> Hybrid Document + Web)"
                else:
                    route_name = "Adaptive-CRAG (Verdict: INCORRECT -> Automated Web Fallback)"

            gen_start = time.perf_counter()
            # If Graph-RAG has strong evidence and query is entity-relational, enrich refined passages
            enriched_passages = list(crag_res.refined_passages)
            if eff_graph and graph_evidence:
                enriched_passages.append("[Knowledge Graph Traversal]: " + " | ".join(graph_evidence[:4]))

            answer = self.generator.generate_crag(
                clean_query,
                refined_passages=enriched_passages,
                web_results=web_sources,
                verdict=crag_res.verdict.value,
                eval_mode=eval_mode
            )
            generation_time_ms = round((time.perf_counter() - gen_start) * 1000, 2)

        else:  # MULTI_HOP
            raw_subqueries = class_res.sub_queries or [clean_query]
            sub_queries = list(dict.fromkeys([sq.strip() for sq in raw_subqueries if sq.strip()]))
            if not sub_queries:
                sub_queries = [clean_query]
            subquery_evidence = []
            sub_verdicts = []
            sub_confs = []

            ret_start = time.perf_counter()
            for sub_q in sub_queries:
                sub_chunks = self.retriever.search(
                    sub_q,
                    top_k=settings.TOP_K,
                    enable_rerank=eff_rerank,
                    allowed_docs=selected_docs
                )
                retrieval_steps += 1
                chunks_retrieved_count += len(sub_chunks)

                for c, score in sub_chunks:
                    retrieved_sources.append({
                        "sub_query": sub_q,
                        "chunk_id": c.chunk_id,
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "score": score,
                        "initial_score": getattr(c, "initial_score", score),
                        "snippet": c.text[:120] + "..."
                    })

                # Step 2b: CRAG Quality Evaluation for each sub-query
                sub_crag: CRAGEvaluationResult = self.crag_evaluator.evaluate(sub_q, sub_chunks)
                sub_verdicts.append(sub_crag.verdict)
                sub_confs.append(sub_crag.confidence_score)

                # Corrective web fallback per sub-query respecting search_mode
                sub_web = []
                if search_mode == "strict_vault" or not eff_web:
                    sub_web = []
                elif search_mode == "web_augmented":
                    sub_web = self.web_search.search(sub_q)
                    web_searches_triggered += 1
                else:  # autonomous
                    if sub_crag.web_search_needed:
                        sub_web = self.web_search.search(sub_q)
                        web_searches_triggered += 1

                if sub_web:
                    web_sources.extend(sub_web)

                subquery_evidence.append({
                    "sub_query": sub_q,
                    "chunks": sub_chunks,
                    "verdict": sub_crag.verdict.value,
                    "confidence": sub_crag.confidence_score,
                    "refined_passages": sub_crag.refined_passages,
                    "web_results": sub_web
                })

            retrieval_time_ms = round((time.perf_counter() - ret_start) * 1000, 2)

            # Aggregate CRAG verdict & confidence across multi-hop sub-queries
            crag_conf_val = round(float(np.mean(sub_confs)), 4) if sub_confs else 0.0
            if any(v == EvaluationVerdict.INCORRECT for v in sub_verdicts):
                crag_verdict_str = EvaluationVerdict.INCORRECT.value
            elif any(v == EvaluationVerdict.AMBIGUOUS for v in sub_verdicts):
                crag_verdict_str = EvaluationVerdict.AMBIGUOUS.value
            else:
                crag_verdict_str = EvaluationVerdict.CORRECT.value

            crag_action_str = (
                f"Multi-hop CRAG: {sum(1 for v in sub_verdicts if v == EvaluationVerdict.CORRECT)} Correct, "
                f"{sum(1 for v in sub_verdicts if v == EvaluationVerdict.AMBIGUOUS)} Ambiguous, "
                f"{sum(1 for v in sub_verdicts if v == EvaluationVerdict.INCORRECT)} Incorrect."
            )

            if search_mode == "strict_vault":
                route_name = f"Multi-Hop Decomposition (Strict Vault • {crag_verdict_str})"
            elif web_searches_triggered > 0:
                route_name = f"Multi-Hop Decomposition & Corrective Web Fallback ({crag_verdict_str})"
            else:
                route_name = f"Multi-Hop Decomposition & Synthesis ({crag_verdict_str})"

            gen_start = time.perf_counter()
            answer = self.generator.generate_multi_hop(
                clean_query,
                subquery_evidence,
                graph_evidence=graph_evidence if eff_graph else None,
                eval_mode=eval_mode
            )
            generation_time_ms = round((time.perf_counter() - gen_start) * 1000, 2)

        total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

        metrics = ExecutionMetrics(
            classification_ms=class_res.latency_ms,
            retrieval_ms=retrieval_time_ms,
            generation_ms=generation_time_ms,
            total_latency_ms=total_latency_ms,
            retrieval_steps=retrieval_steps,
            chunks_retrieved=chunks_retrieved_count,
            web_searches_triggered=web_searches_triggered
        )

        return AdaptiveRAGResponse(
            query=clean_query,
            complexity=class_res.complexity,
            route_name=route_name,
            classifier_reasoning=class_res.reasoning,
            sub_queries=class_res.sub_queries,
            retrieved_sources=retrieved_sources,
            web_sources=web_sources,
            graph_evidence=graph_evidence,
            crag_verdict=crag_verdict_str,
            crag_confidence=crag_conf_val,
            crag_action=crag_action_str,
            answer=answer,
            metrics=metrics
        )

    def process_query_stream(
        self,
        query: str,
        search_mode: str = "autonomous",
        selected_docs: Optional[List[str]] = None
    ):
        """
        Real-Time Streaming Generator for Adaptive-CRAG + GraphRAG:
        1. Classifies complexity & retrieves relevant evidence.
        2. Immediately yields 'metadata' event (sources, complexity, CRAG verdict, graph evidence).
        3. Streams generated answer tokens in real-time as 'token' events.
        4. Yields 'done' event with final latency telemetry.
        """
        total_start = time.perf_counter()
        clean_query = query.strip()

        if not clean_query:
            yield json.dumps({"type": "token", "content": "Please enter a valid question to get started."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Query Graph-RAG
        graph_evidence = []
        if self.enable_graph:
            graph_data = self.retriever.graph_rag.query_graph(clean_query, enabled=True)
            graph_evidence = graph_data.get("evidence_lines", [])

        # Step 1: Classification
        class_res: ClassificationOutput = self.classifier.classify(clean_query)
        retrieved_sources = []
        web_sources = []
        crag_verdict_str = None
        crag_conf_val = 0.0
        route_name = "Adaptive Pipeline"

        # Handle empty vault
        if class_res.complexity != ComplexityLevel.ZERO_RETRIEVAL and self.retriever.count() == 0:
            if search_mode != "strict_vault" and self.enable_web:
                web_sources = self.web_search.search(clean_query)
                route_name = "Web Fallback (Vault Empty)"
                crag_verdict_str = "INCORRECT (EMPTY_INDEX)"
                yield json.dumps({
                    "type": "metadata",
                    "data": {
                        "complexity": class_res.complexity.value,
                        "route_name": route_name,
                        "classifier_reasoning": class_res.reasoning,
                        "sub_queries": [],
                        "retrieved_sources": [],
                        "web_sources": web_sources,
                        "graph_evidence": graph_evidence,
                        "crag_verdict": crag_verdict_str,
                        "crag_confidence": 0.0
                    }
                }) + "\n"
                for tok in self.generator.generate_crag_stream(clean_query, [], web_sources, "INCORRECT"):
                    yield json.dumps({"type": "token", "content": tok}) + "\n"
                yield json.dumps({"type": "done", "latency_ms": round((time.perf_counter() - total_start) * 1000, 2)}) + "\n"
                return
            else:
                yield json.dumps({
                    "type": "metadata",
                    "data": {
                        "complexity": class_res.complexity.value,
                        "route_name": "Vault Empty",
                        "classifier_reasoning": class_res.reasoning,
                        "sub_queries": [],
                        "retrieved_sources": [],
                        "web_sources": [],
                        "graph_evidence": graph_evidence,
                        "crag_verdict": None,
                        "crag_confidence": 0.0
                    }
                }) + "\n"
                yield json.dumps({"type": "token", "content": "Document Vault is currently empty."}) + "\n"
                yield json.dumps({"type": "done", "latency_ms": round((time.perf_counter() - total_start) * 1000, 2)}) + "\n"
                return

        # Execution branching
        if class_res.complexity == ComplexityLevel.ZERO_RETRIEVAL:
            route_name = "Zero-Retrieval (Direct LLM)"
            yield json.dumps({
                "type": "metadata",
                "data": {
                    "complexity": class_res.complexity.value,
                    "route_name": route_name,
                    "classifier_reasoning": class_res.reasoning,
                    "sub_queries": [],
                    "retrieved_sources": [],
                    "web_sources": [],
                    "graph_evidence": [],
                    "crag_verdict": None,
                    "crag_confidence": 1.0
                }
            }) + "\n"

            for tok in self.generator.generate_direct_stream(clean_query):
                yield json.dumps({"type": "token", "content": tok}) + "\n"

        elif class_res.complexity == ComplexityLevel.SINGLE_HOP:
            chunks = self.retriever.search(
                clean_query,
                top_k=settings.TOP_K,
                enable_rerank=self.enable_rerank,
                allowed_docs=selected_docs
            )
            for c, score in chunks:
                retrieved_sources.append({
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "score": score,
                    "initial_score": getattr(c, "initial_score", score),
                    "snippet": c.text[:120] + "..."
                })

            crag_res = self.crag_evaluator.evaluate(clean_query, chunks)
            crag_verdict_str = crag_res.verdict.value
            crag_conf_val = crag_res.confidence_score

            if search_mode == "strict_vault" or not self.enable_web:
                web_sources = []
                route_name = f"Confidence Vault ({crag_res.verdict.value} | Internal Only)"
            elif search_mode == "web_augmented":
                web_sources = self.web_search.search(clean_query)
                route_name = "Market Intelligence (Web-Augmented)"
            else:
                if crag_res.web_search_needed:
                    web_sources = self.web_search.search(clean_query)
                route_name = f"Adaptive-CRAG ({crag_res.verdict.value})"

            enriched_passages = list(crag_res.refined_passages)
            if self.enable_graph and graph_evidence:
                enriched_passages.append("[Knowledge Graph]: " + " | ".join(graph_evidence[:4]))

            yield json.dumps({
                "type": "metadata",
                "data": {
                    "complexity": class_res.complexity.value,
                    "route_name": route_name,
                    "classifier_reasoning": class_res.reasoning,
                    "sub_queries": [],
                    "retrieved_sources": retrieved_sources,
                    "web_sources": web_sources,
                    "graph_evidence": graph_evidence,
                    "crag_verdict": crag_verdict_str,
                    "crag_confidence": crag_conf_val
                }
            }) + "\n"

            for tok in self.generator.generate_crag_stream(
                clean_query,
                refined_passages=enriched_passages,
                web_results=web_sources,
                verdict=crag_res.verdict.value
            ):
                yield json.dumps({"type": "token", "content": tok}) + "\n"

        else:  # MULTI_HOP
            raw_subqueries = class_res.sub_queries or [clean_query]
            sub_queries = list(dict.fromkeys([sq.strip() for sq in raw_subqueries if sq.strip()])) or [clean_query]
            subquery_evidence = []

            for sq in sub_queries:
                sub_chunks = self.retriever.search(sq, top_k=settings.TOP_K, enable_rerank=self.enable_rerank, allowed_docs=selected_docs)
                for c, score in sub_chunks:
                    retrieved_sources.append({
                        "sub_query": sq,
                        "chunk_id": c.chunk_id,
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "score": score,
                        "initial_score": getattr(c, "initial_score", score),
                        "snippet": c.text[:120] + "..."
                    })
                sub_crag = self.crag_evaluator.evaluate(sq, sub_chunks)
                sub_web = []
                if search_mode != "strict_vault" and self.enable_web and sub_crag.web_search_needed:
                    sub_web = self.web_search.search(sq)
                    web_sources.extend(sub_web)
                subquery_evidence.append({
                    "sub_query": sq,
                    "chunks": sub_chunks,
                    "refined_passages": sub_crag.refined_passages,
                    "web_results": sub_web,
                    "verdict": sub_crag.verdict.value
                })

            route_name = "Multi-Hop Reasoning (Adaptive Decomposition)"
            yield json.dumps({
                "type": "metadata",
                "data": {
                    "complexity": class_res.complexity.value,
                    "route_name": route_name,
                    "classifier_reasoning": class_res.reasoning,
                    "sub_queries": sub_queries,
                    "retrieved_sources": retrieved_sources,
                    "web_sources": web_sources,
                    "graph_evidence": graph_evidence,
                    "crag_verdict": "MULTI_STEP",
                    "crag_confidence": 0.85
                }
            }) + "\n"

            for tok in self.generator.generate_multi_hop_stream(
                clean_query,
                subquery_evidence=subquery_evidence,
                graph_evidence=graph_evidence if self.enable_graph else None
            ):
                yield json.dumps({"type": "token", "content": tok}) + "\n"

        total_lat = round((time.perf_counter() - total_start) * 1000, 2)
        yield json.dumps({"type": "done", "latency_ms": total_lat}) + "\n"

    def process_naive_rag(self, query: str) -> Dict[str, Any]:
        """
        Executes standard Traditional/Naive RAG for side-by-side benchmarking:
        - No complexity classification (blind retrieval even for general knowledge)
        - Vector-only search (dense embeddings only, no BM25 lexical search, no Cross-Encoder)
        - No CRAG evaluation or confidence calibration (hallucinates on missing data)
        - No Knowledge Graph traversal or Web Fallback
        """
        start = time.perf_counter()
        clean_query = query.strip()

        if not clean_query:
            return {
                "architecture": "Traditional Naive RAG (Vector Top-4)",
                "answer": "Please enter a valid query.",
                "retrieved_sources": [],
                "latency_ms": 0.0,
                "complexity": "NONE",
                "reranker_active": False,
                "crag_verdict": "NONE"
            }

        # 1. Blind Dense Vector Search (Top 4 chunks)
        raw_chunks = self.retriever.search(
            clean_query,
            top_k=4,
            candidate_k=4,
            dense_weight=1.0,
            sparse_weight=0.0,
            enable_rerank=False
        )

        sources = []
        for c, score in raw_chunks:
            sources.append({
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "page": c.page,
                "score": round(float(score), 4),
                "snippet": c.text[:120] + "..."
            })

        # 2. Direct Uncalibrated Generation
        if raw_chunks:
            answer = self.generator.generate_single_hop(clean_query, raw_chunks)
        else:
            answer = "No documents found in vector index to answer this query."

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "architecture": "Traditional Naive RAG (Vector Top-4)",
            "answer": answer,
            "retrieved_sources": sources,
            "latency_ms": latency_ms,
            "complexity": "BLIND_VECTOR_RETRIEVAL",
            "reranker_active": False,
            "crag_verdict": "UNVERIFIED"
        }
