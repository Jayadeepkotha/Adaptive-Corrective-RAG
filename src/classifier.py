import json
import time
import random
import pickle
from pathlib import Path
from enum import Enum
from typing import List, Optional, Any, Dict
import numpy as np
from pydantic import BaseModel, Field
from loguru import logger
from src.config import settings
from src.llm import UnifiedLLMClient

class ComplexityLevel(str, Enum):
    ZERO_RETRIEVAL = "ZERO_RETRIEVAL"   # Answerable directly by LLM parametric knowledge
    SINGLE_HOP = "SINGLE_HOP"           # Requires 1-step retrieval from knowledge base
    MULTI_HOP = "MULTI_HOP"             # Requires breaking into sub-queries and multi-step retrieval

class ClassificationOutput(BaseModel):
    complexity: ComplexityLevel
    reasoning: str = Field(description="1-2 sentences explaining why this complexity level was chosen")
    sub_queries: List[str] = Field(
        default_factory=list,
        description="List of 2-3 atomic sub-queries if complexity is MULTI_HOP; empty otherwise"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    llm_calls: int = 0

CLASSIFIER_SYSTEM_PROMPT = """You are the Adaptive Complexity Classifier of an Adaptive-RAG system based on the NAACL 2024 research framework.
Your task is to analyze user queries and categorize their computational retrieval complexity strictly by retrieval need into one of three distinct classes:

1. ZERO_RETRIEVAL:
   - Queries answerable directly from parametric knowledge without retrieving external corpus documents.
   - Examples: conversational greetings ("hello", "how are you"), general knowledge/definitions not specific to a private corpus ("what is photosynthesis"), common programming syntax ("how to reverse a string in python"), or basic math calculations.

2. SINGLE_HOP:
   - Queries that require exactly one retrieval step to look up a specific fact, entity, property, definition, event, or answer from a corpus.
   - Examples: "What year was the Eiffel Tower completed?", "Who directed the film Inception?", "What is the capital of Australia?"

3. MULTI_HOP:
   - Queries requiring multiple dependent, sequential, or parallel retrieval steps across different facts or entities.
   - Includes: comparative queries ("Did Company A have higher revenue than Company B?"), bridge/compositional reasoning ("Where was the director of Inception born?"), and multi-constraint lookups.
   - When classified as MULTI_HOP, you MUST decompose the original question into 2-3 atomic, self-contained sub-queries.

Output MUST be strictly valid JSON following this schema:
{
  "complexity": "ZERO_RETRIEVAL" | "SINGLE_HOP" | "MULTI_HOP",
  "reasoning": "Brief explanation of why this complexity level was assigned based on retrieval need.",
  "sub_queries": ["Sub-query 1", "Sub-query 2"] (ONLY if MULTI_HOP, otherwise empty []),
  "confidence": 0.95
}
"""

class AdaptiveClassifier:
    """
    Classifies incoming user queries by complexity to adaptively determine
    the optimal retrieval and reasoning strategy.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        cache: Optional[Any] = None,
        embedder: Optional[Any] = None
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or (settings.GEMINI_MODEL if self.provider == "gemini" else settings.LLM_MODEL)
        self.fallback_model = settings.GEMINI_FALLBACK_MODEL if self.provider == "gemini" else settings.FALLBACK_MODEL
        self.cache = cache
        self.llm_calls: int = 0
        self.logical_calls: int = 0
        self.retry_calls: int = 0
        self.cache_hits: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.call_records: List[Dict[str, Any]] = []
        self.client = UnifiedLLMClient(api_key=api_key, provider=self.provider)
        self.api_key = self.client.api_key
        self._embedder = embedder

        # Load pre-trained local semantic classifier artifact
        self.local_model_path = Path("data/models/query_complexity_clf.pkl")
        self.local_clf = None
        if self.local_model_path.exists():
            try:
                with open(self.local_model_path, "rb") as f:
                    self.local_clf = pickle.load(f)
                logger.info(f"Loaded local complexity classifier from {self.local_model_path}")
            except Exception as e:
                logger.warning(f"Could not load local complexity classifier: {e}")

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer embedder for classifier: {e}")
        return self._embedder

    def reset_counters(self):
        self.llm_calls = 0
        self.logical_calls = 0
        self.retry_calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_records = []

    def classify(self, query: str) -> ClassificationOutput:
        start_time = time.perf_counter()
        clean_query = query.strip()
        if not hasattr(self, "llm_calls"):
            self.llm_calls = 0
        if not hasattr(self, "logical_calls"):
            self.logical_calls = 0
        if not hasattr(self, "retry_calls"):
            self.retry_calls = 0
        if not hasattr(self, "cache_hits"):
            self.cache_hits = 0
        if not hasattr(self, "prompt_tokens"):
            self.prompt_tokens = 0
        if not hasattr(self, "completion_tokens"):
            self.completion_tokens = 0
        if not hasattr(self, "call_records"):
            self.call_records = []

        if not clean_query:
            return ClassificationOutput(
                complexity=ComplexityLevel.ZERO_RETRIEVAL,
                reasoning="Empty query.",
                confidence=1.0,
                latency_ms=0.0
            )

        # 1. Zero-retrieval quick check (Greetings, conversational fillers, etc.)
        q_lower = clean_query.lower()
        zero_markers = [
            "hi", "hello", "hey", "greetings", "good morning", "good evening",
            "how are you", "who are you", "what can you do", "thanks", "thank you", "help"
        ]
        if q_lower in zero_markers or (len(q_lower.split()) <= 2 and q_lower.startswith(("hi", "hello", "hey"))):
            lat_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ClassificationOutput(
                complexity=ComplexityLevel.ZERO_RETRIEVAL,
                reasoning="Conversational greeting or generic system inquiry.",
                confidence=0.99,
                latency_ms=lat_ms,
                llm_calls=0
            )

        # 2. Local Semantic Dense Classifier (Runs in ~15ms, 0 API Calls)
        local_pred_comp = None
        local_conf = 0.5
        local_clf = getattr(self, "local_clf", None)
        if local_clf is not None and self.embedder is not None:
            try:
                emb = self.embedder.encode([clean_query], normalize_embeddings=True)
                probs = local_clf.predict_proba(emb)[0]  # [p_single, p_multi]
                pred_idx = int(np.argmax(probs))
                local_conf = float(probs[pred_idx])
                local_pred_comp = ComplexityLevel.MULTI_HOP if pred_idx == 1 else ComplexityLevel.SINGLE_HOP

                # Route locally for any clear decision (>= 0.52)
                if local_conf >= 0.52:
                    lat_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    sub_q = []
                    if local_pred_comp == ComplexityLevel.MULTI_HOP:
                        sub_q = self.decompose_query(clean_query)
                        if not sub_q:
                            sub_q = [clean_query]
                    logger.info(f"Local Semantic Route: {local_pred_comp.value} (Confidence: {local_conf*100:.1f}%, Latency: {lat_ms}ms, Sub-queries: {len(sub_q)})")
                    return ClassificationOutput(
                        complexity=local_pred_comp,
                        reasoning=f"Classified locally via dense semantic projection (Confidence: {local_conf*100:.1f}%).",
                        sub_queries=sub_q,
                        confidence=local_conf,
                        latency_ms=lat_ms,
                        llm_calls=0
                    )
            except Exception as e:
                logger.warning(f"Local classification error: {e}. Falling back to cloud/cache.")

        # 3. Ambiguous Edge-Case Resolution (Cache or Cloud LLM fallback)
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze and classify this query:\n\n\"{clean_query}\""}
        ]

        # Check cache if available
        cache = getattr(self, "cache", None)
        if cache is not None:
            cached = cache.get_llm(self.model, messages, max_tokens=500)
            if cached and "text" in cached:
                try:
                    parsed = json.loads(cached["text"])
                    comp_str = parsed.get("complexity", "SINGLE_HOP").upper()
                    if "ZERO" in comp_str or "NONE" in comp_str:
                        comp_level = ComplexityLevel.ZERO_RETRIEVAL
                    elif "MULTI" in comp_str:
                        comp_level = ComplexityLevel.MULTI_HOP
                    else:
                        comp_level = ComplexityLevel.SINGLE_HOP
                    sub_q = parsed.get("sub_queries", []) if comp_level == ComplexityLevel.MULTI_HOP else []
                    
                    self.logical_calls += 1
                    self.llm_calls += 1
                    self.cache_hits += 1
                    usage = cached.get("usage", {})
                    p_tok = usage.get("prompt_tokens", 0)
                    c_tok = usage.get("completion_tokens", 0)
                    self.prompt_tokens += p_tok
                    self.completion_tokens += c_tok
                    lat_s = round(time.perf_counter() - start_time, 4)
                    self.call_records.append({
                        "caller": "classifier",
                        "operation": "classify",
                        "model": self.model,
                        "cached": True,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "retries": 0,
                        "latency_s": lat_s
                    })
                    return ClassificationOutput(
                        complexity=comp_level,
                        reasoning=parsed.get("reasoning", "Retrieved from cache."),
                        sub_queries=sub_q,
                        confidence=float(parsed.get("confidence", 0.95)),
                        latency_ms=round(lat_s * 1000, 2),
                        llm_calls=1
                    )
                except Exception:
                    pass

        # Call Cloud LLM with at most 1 attempt to avoid 40s hangs on rate limits
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            self.logical_calls += 1
            self.llm_calls += 1
            usage_dict = {}
            if hasattr(response, "usage") and response.usage:
                usage_dict = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0
                }
                self.prompt_tokens += usage_dict["prompt_tokens"]
                self.completion_tokens += usage_dict["completion_tokens"]

            raw_json = response.choices[0].message.content
            if raw_json and raw_json.strip():
                lat_s = round(time.perf_counter() - start_time, 4)
                self.call_records.append({
                    "caller": "classifier",
                    "operation": "classify",
                    "model": self.model,
                    "cached": False,
                    "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                    "completion_tokens": usage_dict.get("completion_tokens", 0),
                    "retries": 0,
                    "latency_s": lat_s
                })

                cache = getattr(self, "cache", None)
                if cache is not None:
                    cache.set_llm(self.model, messages, raw_json.strip(), usage_dict, max_tokens=500)

                parsed = json.loads(raw_json.strip())
                complexity_str = parsed.get("complexity", "SINGLE_HOP").upper()

                if "ZERO" in complexity_str or "NO_RETRIEVAL" in complexity_str or "NONE" in complexity_str:
                    comp_level = ComplexityLevel.ZERO_RETRIEVAL
                elif "MULTI" in complexity_str or "ITERATIVE" in complexity_str:
                    comp_level = ComplexityLevel.MULTI_HOP
                else:
                    comp_level = ComplexityLevel.SINGLE_HOP

                sub_queries = parsed.get("sub_queries", [])
                if comp_level != ComplexityLevel.MULTI_HOP:
                    sub_queries = []

                latency_ms = round(lat_s * 1000, 2)
                return ClassificationOutput(
                    complexity=comp_level,
                    reasoning=parsed.get("reasoning", "Classified according to query structure."),
                    sub_queries=sub_queries,
                    confidence=float(parsed.get("confidence", 0.95)),
                    latency_ms=latency_ms,
                    llm_calls=1
                )
        except Exception as e:
            logger.warning(f"Cloud classifier call bypassed or timed out: {e}. Resolving with local model.")

        # Fallback to local prediction or heuristic
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        if local_pred_comp is not None:
            sub_q = self.decompose_query(clean_query) if local_pred_comp == ComplexityLevel.MULTI_HOP else []
            if local_pred_comp == ComplexityLevel.MULTI_HOP and not sub_q:
                sub_q = [clean_query]
            return ClassificationOutput(
                complexity=local_pred_comp,
                reasoning="Classified via local semantic model (cloud unavailable).",
                sub_queries=sub_q,
                confidence=local_conf,
                latency_ms=latency_ms,
                llm_calls=0
            )

        return self._heuristic_fallback(clean_query, latency_ms)

    def decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a complex multi-hop query into 2-3 focused sub-queries.
        Used when multi_step strategy is explicitly selected.
        """
        start_time = time.perf_counter()
        clean_query = query.strip()
        if not hasattr(self, "llm_calls"):
            self.llm_calls = 0
        if not hasattr(self, "logical_calls"):
            self.logical_calls = 0
        if not hasattr(self, "retry_calls"):
            self.retry_calls = 0
        if not hasattr(self, "cache_hits"):
            self.cache_hits = 0
        if not hasattr(self, "prompt_tokens"):
            self.prompt_tokens = 0
        if not hasattr(self, "completion_tokens"):
            self.completion_tokens = 0
        if not hasattr(self, "call_records"):
            self.call_records = []

        if not clean_query:
            return [query]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query decomposition assistant.\n"
                    "Decompose the complex multi-part question into 2-3 atomic, self-contained sub-questions.\n"
                    "Output strictly valid JSON: {\"sub_queries\": [\"sub_question 1\", \"sub_question 2\"]}"
                )
            },
            {"role": "user", "content": f"Decompose this question:\n\"{clean_query}\""}
        ]

        cache = getattr(self, "cache", None)
        if cache is not None:
            cached = cache.get_llm(self.model, messages, max_tokens=300)
            if cached and "text" in cached:
                try:
                    p = json.loads(cached["text"])
                    sqs = p.get("sub_queries", [])
                    if sqs and isinstance(sqs, list):
                        self.logical_calls += 1
                        self.llm_calls += 1
                        self.cache_hits += 1
                        usage = cached.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        self.prompt_tokens += p_tok
                        self.completion_tokens += c_tok
                        lat_s = round(time.perf_counter() - start_time, 4)
                        self.call_records.append({
                            "caller": "classifier",
                            "operation": "decompose_query",
                            "model": self.model,
                            "cached": True,
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "retries": 0,
                            "latency_s": lat_s
                        })
                        return [s.strip() for s in sqs if s.strip()]
                except Exception:
                    pass

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            self.logical_calls += 1
            self.llm_calls += 1
            usage_dict = {}
            if hasattr(resp, "usage") and resp.usage:
                usage_dict = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0
                }
                self.prompt_tokens += usage_dict["prompt_tokens"]
                self.completion_tokens += usage_dict["completion_tokens"]
            text = resp.choices[0].message.content
            if text and text.strip():
                lat_s = round(time.perf_counter() - start_time, 4)
                self.call_records.append({
                    "caller": "classifier",
                    "operation": "decompose_query",
                    "model": self.model,
                    "cached": False,
                    "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                    "completion_tokens": usage_dict.get("completion_tokens", 0),
                    "retries": 0,
                    "latency_s": lat_s
                })
                if cache is not None:
                    cache.set_llm(self.model, messages, text.strip(), usage_dict, max_tokens=300)
                p = json.loads(text.strip())
                sqs = p.get("sub_queries", [])
                if sqs and isinstance(sqs, list):
                    return [s.strip() for s in sqs if s.strip()]
        except Exception as e:
            logger.warning(f"Decomposition LLM call failed: {e}. Falling back to heuristic.")

        # Fallback decomposition: split on comparative or bridge words
        q_lower = clean_query.lower()
        if " and " in q_lower:
            parts = clean_query.split(" and ")
            if len(parts) >= 2:
                return [p.strip() for p in parts if p.strip()]
        return [clean_query]

    def _heuristic_fallback(self, query: str, latency_ms: float) -> ClassificationOutput:
        """Domain-neutral heuristic rule-based fallback if cloud API is temporarily unreachable."""
        q_lower = query.lower().strip()

        # 1. Zero retrieval: greetings, conversational fillers, simple arithmetic
        if q_lower in ["hi", "hello", "hey", "greetings", "good morning", "good evening", "how are you", "who are you", "thanks", "thank you"]:
            return ClassificationOutput(
                complexity=ComplexityLevel.ZERO_RETRIEVAL,
                reasoning="Heuristic fallback: Conversational greeting or filler detected.",
                confidence=0.90,
                latency_ms=latency_ms,
                llm_calls=0
            )

        # 2. Multi-hop: explicit comparative or multi-step conjunction markers
        comparative_markers = [
            "compare", "comparison", "difference between", "differ between",
            "versus", "vs.", " vs ", "both", "in common", "respectively"
        ]
        if any(marker in q_lower for marker in comparative_markers):
            return ClassificationOutput(
                complexity=ComplexityLevel.MULTI_HOP,
                reasoning="Heuristic fallback: Comparative or multi-entity structure detected.",
                sub_queries=[query],
                confidence=0.75,
                latency_ms=latency_ms,
                llm_calls=0
            )

        # Bridge reasoning heuristics: "father of X's mother", "director of film X ... birth"
        if (" of " in q_lower and any(w in q_lower for w in ["'s", " who ", " whose ", " where "])) or (" and " in q_lower and (" both " in q_lower or " also " in q_lower)):
            return ClassificationOutput(
                complexity=ComplexityLevel.MULTI_HOP,
                reasoning="Heuristic fallback: Multi-hop relation or bridge structure detected.",
                sub_queries=[query],
                confidence=0.70,
                latency_ms=latency_ms,
                llm_calls=0
            )

        # 3. Default to Single-Hop for factual queries
        return ClassificationOutput(
            complexity=ComplexityLevel.SINGLE_HOP,
            reasoning="Heuristic fallback: Single retrieval step lookup query.",
            confidence=0.70,
            latency_ms=latency_ms,
            llm_calls=0
        )
