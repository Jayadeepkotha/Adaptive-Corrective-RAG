import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure offline initialization so local cached models load in milliseconds
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from loguru import logger
from src.config import settings
from src.graph_rag import KnowledgeGraphRAG
from src.multimodal_ingest import MultiModalIngestor, MultiModalDocument
from src.storage import VaultStorage

class DocumentChunk:
    def __init__(self, chunk_id: str, doc_name: str, text: str, page: int = 1):
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.text = text
        self.page = page
        self.initial_score: float = 0.0
        self.rerank_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "page": self.page,
            "initial_score": self.initial_score,
            "rerank_score": self.rerank_score,
        }

class HybridRetriever:
    """
    Two-Stage Hybrid Retriever:
    1. Stage 1 (Fast Recall): Hybrid Dense (SentenceTransformer) + Sparse (BM25Okapi) candidate retrieval.
    2. Stage 2 (Precision Reranking): Deep Cross-Encoder (MS-MARCO MiniLM) cross-attention reranking
       and calibrated confidence scoring.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        calibration_offset: Optional[float] = None,
    ):
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.reranker_model_name = reranker_model or settings.RERANKER_MODEL
        self.calibration_offset = (
            calibration_offset
            if calibration_offset is not None
            else settings.CROSS_ENCODER_CALIBRATION_OFFSET
        )

        logger.info(f"Loading Stage-1 Bi-Encoder: '{self.embedding_model_name}'...")
        try:
            self.encoder = SentenceTransformer(self.embedding_model_name, local_files_only=True)
        except Exception:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        logger.info(f"Loading Stage-2 Cross-Encoder: '{self.reranker_model_name}'...")
        try:
            self.reranker = CrossEncoder(self.reranker_model_name, local_files_only=True)
        except Exception:
            self.reranker = CrossEncoder(self.reranker_model_name)

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.graph_rag = KnowledgeGraphRAG()
        self.ingestor = MultiModalIngestor()
        self.storage = VaultStorage()
        self._query_emb_cache: Dict[str, np.ndarray] = {}

        # Restore any existing persistent documents and knowledge graph
        self._restore_from_storage()

    def _restore_from_storage(self):
        """Restores chunks, embeddings, and knowledge graph from persistent storage upon startup."""
        stored_chunks = self.storage.load_all_chunks()
        if not stored_chunks:
            return

        dim = self.encoder.get_sentence_embedding_dimension()
        faiss_index = self.storage.load_faiss_index(dim=dim)

        # Restore chunks
        self.chunks = []
        for sc in stored_chunks:
            chunk = DocumentChunk(
                chunk_id=sc["chunk_id"],
                doc_name=sc["doc_name"],
                text=sc["text"],
                page=sc["page"]
            )
            self.chunks.append(chunk)

        # Reconstruct embeddings array from FAISS or re-encode if index mismatch
        if faiss_index.ntotal == len(self.chunks) and faiss_index.ntotal > 0:
            try:
                # Reconstruct vectors from FAISS index
                raw_vectors = np.zeros((faiss_index.ntotal, dim), dtype="float32")
                for i in range(faiss_index.ntotal):
                    raw_vectors[i] = faiss_index.reconstruct(i)
                self.embeddings = raw_vectors
            except Exception:
                # Some IndexFlatIP types allow direct buffer access
                texts = [c.text for c in self.chunks]
                self.embeddings = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elif self.chunks:
            texts = [c.text for c in self.chunks]
            self.embeddings = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Build BM25
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Restore Knowledge Graph triples
        stored_triples = self.storage.load_all_triples()
        for s, r, o, doc in stored_triples:
            self.graph_rag.add_triple(s, r, o, doc_name=doc)

        logger.info(f"Successfully restored {len(self.chunks)} chunks and {len(stored_triples)} graph triples from persistent vault storage.")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def add_document(self, file_path: Path, doc_name: Optional[str] = None) -> int:
        """
        Universal multi-modal ingestion gateway:
        Automatically routes PDF, CSV, JSON, PNG, JPG, TXT, and Markdown to the specialized parser.
        Extracts semantic passages and knowledge graph triples into the index.
        Persists to SQLite and FAISS storage.
        """
        resolved_name = doc_name or file_path.name
        multi_doc: MultiModalDocument = self.ingestor.process_file(file_path, doc_name=resolved_name)
        if not multi_doc.passages:
            return 0

        doc_chunks = []
        for idx, passage_text in enumerate(multi_doc.passages):
            c_id = f"{resolved_name}_chunk_{len(self.chunks) + len(doc_chunks)}"
            doc_chunks.append(DocumentChunk(c_id, resolved_name, passage_text, page=idx + 1))

        # Ingest extracted semantic triples into Knowledge Graph
        for s, r, o in multi_doc.triples:
            self.graph_rag.add_triple(s, r, o, doc_name=resolved_name)

        # Also run pattern-based heuristic extractor on combined text
        full_text = "\n".join(multi_doc.passages)
        if full_text:
            self.graph_rag.extract_and_index_text(resolved_name, full_text[:12000])

        all_triples = [(t.subject, t.relation, t.obj) for t in self.graph_rag.triples if t.doc_name == resolved_name]

        self._index_chunks(
            doc_chunks,
            doc_name=resolved_name,
            doc_type=multi_doc.doc_type,
            triples=all_triples
        )

        logger.info(f"Ingested & Persisted '{resolved_name}' ({multi_doc.doc_type}): {len(doc_chunks)} chunks, {len(all_triples)} triples.")
        return len(doc_chunks)

    def add_text_document(self, doc_name: str, text: str, chunk_size: int = 250, chunk_overlap: int = 40):
        """Adds text or markdown document using semantic paragraph splitting."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        if not paragraphs:
            return

        doc_chunks = []
        curr_words = []
        for p in paragraphs:
            words = p.split()
            if len(curr_words) + len(words) > chunk_size and curr_words:
                c_id = f"{doc_name}_chunk_{len(self.chunks) + len(doc_chunks)}"
                doc_chunks.append(DocumentChunk(c_id, doc_name, " ".join(curr_words), page=len(doc_chunks) + 1))
                curr_words = []
            curr_words.extend(words)

        if curr_words:
            c_id = f"{doc_name}_chunk_{len(self.chunks) + len(doc_chunks)}"
            doc_chunks.append(DocumentChunk(c_id, doc_name, " ".join(curr_words), page=len(doc_chunks) + 1))

        self.graph_rag.extract_and_index_text(doc_name, text)
        triples = [(t.subject, t.relation, t.obj) for t in self.graph_rag.triples if t.doc_name == doc_name]

        self._index_chunks(doc_chunks, doc_name=doc_name, doc_type="text", triples=triples)

    def add_pdf_document(self, pdf_path: Path, doc_name: Optional[str] = None):
        """Layout-aware PDF ingestion using PyMuPDF blocks."""
        return self.add_document(pdf_path, doc_name=doc_name)

    def add_csv_document(self, csv_path: Path, doc_name: Optional[str] = None):
        """Structured tabular CSV ingestion with row-level semantics."""
        return self.add_document(csv_path, doc_name=doc_name)

    def add_json_document(self, json_path: Path, doc_name: Optional[str] = None):
        """Hierarchical JSON/JSONL ingestion with path flattening."""
        return self.add_document(json_path, doc_name=doc_name)

    def add_image_document(self, image_path: Path, doc_name: Optional[str] = None):
        """Multi-modal image/photo ingestion with visual OCR and metadata."""
        return self.add_document(image_path, doc_name=doc_name)

    def _index_chunks(
        self,
        new_chunks: List[DocumentChunk],
        doc_name: str = "document",
        doc_type: str = "text",
        triples: Optional[List[Tuple[str, str, str]]] = None,
        persist: bool = True
    ):
        if not new_chunks:
            return

        texts = [c.text for c in new_chunks]
        new_embs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Save to SQLite + FAISS only if persistence requested (e.g. permanent vault ingestion)
        if persist:
            self.storage.save_batch(
                doc_name=doc_name,
                doc_type=doc_type,
                chunks=new_chunks,
                embeddings=new_embs,
                triples=triples or []
            )

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.chunks.extend(new_chunks)

        # Rebuild BM25 index
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def clear_vault(self):
        """Clears all persistent data from SQLite and FAISS, as well as in-memory state."""
        self.storage.clear()
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.graph_rag.clear()
        self._query_emb_cache.clear()
        logger.info("Retriever vault cleared.")

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 60,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        enable_rerank: Optional[bool] = None,
        max_per_doc: Optional[int] = None,
        allowed_docs: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Two-stage search:
        1. Hybrid Bi-Encoder + BM25 retrieves top candidate_k chunks.
        2. Cross-Encoder re-ranks candidate pool using deep cross-attention.
        3. Smart relevance filtering & context preservation.
        """
        if not query or not query.strip() or top_k <= 0 or not self.chunks or self.embeddings is None or self.bm25 is None:
            return []

        if enable_rerank is None:
            enable_rerank = settings.ENABLE_RERANK

        effective_max_per_doc = (
            max_per_doc
            if max_per_doc is not None
            else (settings.MAX_PER_DOC or min(top_k, 5))
        )

        # -------------------------------------------------------------
        # STAGE 1: Candidate Generation (RRF Hybrid Dense + Sparse)
        # -------------------------------------------------------------
        if query not in self._query_emb_cache:
            if len(self._query_emb_cache) > 300:
                self._query_emb_cache.clear()
            self._query_emb_cache[query] = self.encoder.encode(query, normalize_embeddings=True)
        q_emb = self._query_emb_cache[query]
        dense_scores = np.dot(self.embeddings, q_emb)

        tokenized_q = self._tokenize(query)
        bm25_raw = self.bm25.get_scores(tokenized_q)

        # Reciprocal Rank Fusion (RRF) with k=60 for optimal hybrid balance
        dense_ranks = np.argsort(-dense_scores)
        bm25_ranks = np.argsort(-bm25_raw)

        dense_rank_pos = np.empty_like(dense_ranks)
        dense_rank_pos[dense_ranks] = np.arange(len(dense_ranks))

        bm25_rank_pos = np.empty_like(bm25_ranks)
        bm25_rank_pos[bm25_ranks] = np.arange(len(bm25_ranks))

        rrf_scores = (dense_weight / (60.0 + dense_rank_pos)) + (sparse_weight / (60.0 + bm25_rank_pos))

        # Multi-select document filtering
        if allowed_docs:
            allowed_set = set(allowed_docs)
            valid_mask = np.array([c.doc_name in allowed_set for c in self.chunks])
            if not np.any(valid_mask):
                return []
            rrf_scores = np.where(valid_mask, rrf_scores, -1e9)

        # Take pool of top candidates for reranking
        pool_size = min(candidate_k, len(self.chunks))
        candidate_indices = np.argsort(-rrf_scores)[:pool_size]

        candidate_chunks = [self.chunks[i] for i in candidate_indices]
        for rank, idx in enumerate(candidate_indices):
            candidate_chunks[rank].initial_score = round(float(rrf_scores[idx]), 4)

        if not enable_rerank or len(candidate_chunks) <= 1:
            return [(c, c.initial_score) for c in candidate_chunks[:top_k]]

        # -------------------------------------------------------------
        # STAGE 2: Deep Cross-Encoder Reranking
        # -------------------------------------------------------------
        pairs = [(query, c.text) for c in candidate_chunks]
        raw_logits = self.reranker.predict(pairs)

        # Calibrated Sigmoid confidence: 1 / (1 + exp(-(logit + offset)))
        calibrated_scores = 1.0 / (1.0 + np.exp(-(raw_logits + self.calibration_offset)))
        calibrated_scores = np.clip(calibrated_scores, 0.01, 0.99)

        # Sort candidates by reranker score descending
        rerank_order = np.argsort(-calibrated_scores)

        results: List[Tuple[DocumentChunk, float]] = []
        doc_counts: Dict[str, int] = {}
        deferred: List[Tuple[DocumentChunk, float]] = []

        top_confidence = float(calibrated_scores[rerank_order[0]]) if len(rerank_order) > 0 else 0.0
        # Adaptive noise floor cutoff: aggressively rejects irrelevant distractor chunks
        if top_confidence >= 0.70:
            noise_floor = max(0.35, top_confidence * 0.45)
        elif top_confidence >= 0.45:
            noise_floor = max(0.25, top_confidence * 0.35)
        else:
            noise_floor = 0.15

        for idx in rerank_order:
            chunk = candidate_chunks[idx]
            conf = round(float(calibrated_scores[idx]), 4)
            chunk.rerank_score = conf

            # Reject chunks that fall below the dynamic noise floor
            if conf < noise_floor:
                if len(results) >= 1:
                    continue
                if conf < 0.15:
                    continue

            current_count = doc_counts.get(chunk.doc_name, 0)
            if current_count < effective_max_per_doc:
                results.append((chunk, conf))
                doc_counts[chunk.doc_name] = current_count + 1
                if len(results) >= top_k:
                    break
            else:
                deferred.append((chunk, conf))

        # Backfill from deferred only if they meet the strict relevance floor
        if len(results) < top_k:
            for chunk, conf in deferred:
                if conf >= noise_floor:
                    results.append((chunk, conf))
                    if len(results) >= top_k:
                        break

        return results

    def add_passages(self, items: List[Tuple[str, str]]):
        """
        Batch-indexes multiple (doc_name/title, text) passages simultaneously.
        Ideal for evaluation benchmarks (distractor and pooled settings).
        """
        if not items:
            return 0
        new_chunks = []
        for title, text in items:
            clean_text = text.strip()
            if not clean_text:
                continue
            c_id = f"{title}_chunk_{len(self.chunks) + len(new_chunks)}"
            new_chunks.append(DocumentChunk(c_id, title, clean_text, page=1))

        self._index_chunks(new_chunks, persist=False)
        if settings.ENABLE_GRAPH_RAG and getattr(self.graph_rag, "is_enabled", True):
            for title, text in items:
                self.graph_rag.extract_and_index_text(title, text[:4000])
        return len(new_chunks)

    def clear(self):
        """Resets the in-memory retriever index and knowledge graph completely."""
        self.chunks.clear()
        self.embeddings = None
        self.bm25 = None
        self.tokenized_corpus.clear()
        self.graph_rag.clear()
        self._query_emb_cache.clear()

    def count(self) -> int:
        return len(self.chunks)
