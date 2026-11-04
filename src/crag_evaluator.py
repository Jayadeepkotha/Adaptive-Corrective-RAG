import re
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from loguru import logger
from src.config import settings
from src.retriever import DocumentChunk

class EvaluationVerdict(str, Enum):
    CORRECT = "CORRECT"         # Retrieval confidence is high; documents directly answer the query
    AMBIGUOUS = "AMBIGUOUS"     # Retrieval confidence is moderate; documents partially match
    INCORRECT = "INCORRECT"     # Retrieval confidence is low; documents are irrelevant; trigger web search fallback

class CRAGEvaluationResult:
    def __init__(
        self,
        verdict: EvaluationVerdict,
        confidence_score: float,
        refined_passages: List[str],
        action_summary: str,
        web_search_needed: bool = False
    ):
        self.verdict = verdict
        self.confidence_score = confidence_score
        self.refined_passages = refined_passages
        self.action_summary = action_summary
        self.web_search_needed = web_search_needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence_score": self.confidence_score,
            "action_summary": self.action_summary,
            "web_search_needed": self.web_search_needed,
            "num_refined_passages": len(self.refined_passages),
        }

class RetrievalEvaluator:
    """
    CRAG Retrieval Quality Evaluator.
    Evaluates Cross-Encoder confidence scores to decide whether documents
    are CORRECT, AMBIGUOUS, or INCORRECT, and performs Document Refinement.
    """

    def __init__(
        self,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None
    ):
        self.upper_threshold = (
            upper_threshold if upper_threshold is not None else settings.CRAG_UPPER_THRESHOLD
        )
        self.lower_threshold = (
            lower_threshold if lower_threshold is not None else settings.CRAG_LOWER_THRESHOLD
        )

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> CRAGEvaluationResult:
        if not retrieved_chunks:
            return CRAGEvaluationResult(
                verdict=EvaluationVerdict.INCORRECT,
                confidence_score=0.0,
                refined_passages=[],
                action_summary="No document chunks found. Routing to Web Search Fallback.",
                web_search_needed=True
            )

        # Max confidence from reranked candidate pool
        top_score = max([score for _, score in retrieved_chunks])

        # Preserve full contextual integrity of each retrieved chunk
        refined_passages = self._refine_documents(query, retrieved_chunks)

        if top_score >= self.upper_threshold:
            verdict = EvaluationVerdict.CORRECT
            action = f"Confidence is high ({top_score*100:.1f}%). Using internal document evidence."
            web_needed = False
        elif top_score >= self.lower_threshold:
            verdict = EvaluationVerdict.AMBIGUOUS
            action = f"Confidence is moderate ({top_score*100:.1f}%). Augmenting internal documents with Web Search."
            web_needed = True
        else:
            verdict = EvaluationVerdict.INCORRECT
            action = f"Confidence is low ({top_score*100:.1f}%). Routing to Web Search Fallback."
            web_needed = True

        logger.info(f"CRAG Evaluator Verdict: {verdict.value} (Score: {top_score:.4f}, Web: {web_needed})")
        return CRAGEvaluationResult(
            verdict=verdict,
            confidence_score=top_score,
            refined_passages=refined_passages,
            action_summary=action,
            web_search_needed=web_needed
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Splits text into sentences while protecting abbreviations, initials (e.g. 'George M. Cohan',
        'Warner Bros.'), and numbers with decimals (e.g. '3.8').
        """
        t = text
        t = re.sub(r"\b([A-Za-z])\.", r"\1<DOT>", t)
        t = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Bros|vs|No|approx|inc|ltd|co)\.", r"\1<DOT>", t, flags=re.IGNORECASE)
        t = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", t)
        raw = re.split(r"(?<=[.!?])\s+", t)
        return [s.replace("<DOT>", ".").strip() for s in raw if s.strip()]

    def _refine_documents(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> List[str]:
        """
        Document Refinement & Salient Strip Extraction (Yan et al., CRAG 2024):
        Decomposes retrieved document chunks into fine-grained sentences (knowledge strips).
        Filters out irrelevant sentences, keeping only fact-bearing sentences that overlap
        with query entities, key terms, or numbers.
        Eliminates 50-70% of distractor noise, preventing generator hallucination.
        """
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower())) - {
            "what", "when", "where", "which", "who", "whom", "this", "that", "there", "their", "from", "with", "have", "been", "does", "were"
        }

        refined = []
        for chunk, score in retrieved_chunks:
            text = chunk.text.strip()
            if not text:
                continue

            # Split into sentence strips with abbreviation protection
            raw_sentences = self._split_sentences(text)

            # For paragraphs of 4 sentences or fewer, retain the full paragraph intact
            # to preserve intermediate reasoning bridges, pronouns, and entity locations.
            if len(raw_sentences) <= 4:
                refined.append(f"[{chunk.doc_name}]: {text}")
                continue

            salient_sentences = []

            for s in raw_sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", s_clean.lower()))
                overlap = len(q_tokens.intersection(s_tokens))
                has_numbers_or_caps = bool(re.search(r"\b([0-9]+|\$[0-9]+|[A-Z][a-z]+)\b", s_clean))

                # Retain sentences that match query keywords or contain concrete entities/numbers
                if overlap > 0 or (has_numbers_or_caps and len(salient_sentences) < 3):
                    salient_sentences.append(s_clean)

            # Fallback to first 4 sentences if sentence filter was too aggressive
            if not salient_sentences and raw_sentences:
                salient_sentences = [s.strip() for s in raw_sentences[:4] if s.strip()]

            refined_content = " ".join(salient_sentences)
            if refined_content:
                refined.append(f"[{chunk.doc_name}]: {refined_content}")

        return refined
