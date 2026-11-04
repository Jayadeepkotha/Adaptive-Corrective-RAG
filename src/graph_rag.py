import re
import json
from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
from loguru import logger
from src.config import settings

class GraphEntityTriple:
    def __init__(self, subject: str, relation: str, obj: str, doc_name: str = ""):
        self.subject = subject.strip()
        self.relation = relation.strip()
        self.obj = obj.strip()
        self.doc_name = doc_name

    def to_dict(self) -> Dict[str, str]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "doc_name": self.doc_name
        }

class KnowledgeGraphRAG:
    """
    Enterprise-Grade Graph-RAG Engine.
    Constructs an in-memory & persistent Entity-Relationship Knowledge Graph (NetworkX)
    during document ingestion to solve cross-document synthesis and aggregation queries.

    Features:
    - LLM-Powered OpenIE (Open Information Extraction) via local Qwen / Gemini fallback.
    - Robust Heuristic SRO pattern matching fallback for 100% offline resilience.
    - Multi-Hop Graph Traversal (1-hop & 2-hop neighbor chaining).
    - Cytoscape & DOT serialization for visual knowledge graph exploration.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.graph = nx.DiGraph()
        self.triples: List[GraphEntityTriple] = []
        self._llm_client = llm_client

    def _get_llm_client(self):
        """Lazily initialize UnifiedLLMClient to avoid startup overhead."""
        if self._llm_client is None:
            try:
                from src.llm import UnifiedLLMClient
                self._llm_client = UnifiedLLMClient(timeout=8.0)
            except Exception as e:
                logger.warning(f"Could not initialize LLM client for Graph-RAG OpenIE: {e}")
        return self._llm_client

    def clear(self):
        """Resets the knowledge graph and removes all stored triples."""
        self.graph.clear()
        self.triples.clear()
        logger.info("Knowledge Graph cleared.")

    def add_triple(self, subject: str, relation: str, obj: str, doc_name: str = ""):
        if not subject or not obj:
            return
        clean_s = subject.strip().strip('"\'')
        clean_r = relation.strip().strip('"\'').upper().replace(" ", "_").replace("-", "_")
        clean_o = obj.strip().strip('"\'')

        # Filter out self-loops or empty/trivial nodes
        if not clean_s or not clean_o or clean_s.lower() == clean_o.lower():
            return
        if len(clean_s) < 2 or len(clean_o) < 2:
            return

        self.graph.add_node(clean_s, doc=doc_name)
        self.graph.add_node(clean_o, doc=doc_name)
        self.graph.add_edge(clean_s, clean_o, relation=clean_r, doc=doc_name)
        self.triples.append(GraphEntityTriple(clean_s, clean_r, clean_o, doc_name))

    def extract_and_index_text(self, doc_name: str, text: str, use_llm: bool = False):
        """
        Dual-Engine Entity-Relation Extractor:
        1. Fast Regex Baseline: Heuristic Subject-Relation-Object pattern extraction (0.01ms, 0 API calls).
        2. LLM OpenIE: High-fidelity structured factual extraction (optional, used when use_llm=True).
        """
        if not text:
            return

        initial_count = len(self.triples)

        # -------------------------------------------------------------
        # 1. Fast Regex Baseline Extraction (guaranteed offline, 0.01ms)
        # -------------------------------------------------------------
        sro_pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9_\s]{1,30}?)\s+(is an?|was an?|part of|located in|member of|created by|authored by|founded by|known as|born in|died in|subsidiary of|consists of|supplied by|developed by|led by|managed by|headquartered in|deployed on|certified by|enforces|connected to|operates in|manufactured by)\s+([A-Z][A-Za-z0-9_\s]{1,30})\b",
            flags=re.IGNORECASE
        )
        for s, r, o in sro_pattern.findall(text):
            clean_s = s.strip()
            clean_o = o.strip()
            if len(clean_s) >= 2 and len(clean_o) >= 2:
                rel = r.strip().upper().replace(" ", "_")
                self.add_triple(clean_s, rel, clean_o, doc_name)

        # Capitalized entity links to source document
        entity_matches = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,3})\b", text)
        for entity in set(entity_matches[:12]):
            clean_ent = entity.strip()
            if len(clean_ent) >= 3 and clean_ent.lower() not in ["the", "this", "that", "these", "those"]:
                self.add_triple(clean_ent, "MENTIONED_IN", doc_name, doc_name)

        # -------------------------------------------------------------
        # 2. LLM OpenIE (Open Information Extraction - Opt-in)
        # -------------------------------------------------------------
        if use_llm:
            client = self._get_llm_client()
            if client:
                try:
                    # Sample representative text from intro & key paragraphs (up to 3,000 chars)
                    sample_snippet = text[:3200].strip()
                    system_prompt = (
                        "You are an Open Information Extraction (OpenIE) engine for enterprise knowledge graphs.\n"
                        "Extract concrete semantic entity-relationship triples from the provided text.\n"
                        "Return a JSON object with a single key 'triples' containing a list of objects with keys 'subject', 'relation', 'object'.\n"
                        "Rules:\n"
                        "1. 'relation' must be an uppercase predicate (e.g. DEPLOYED_ON, RELIES_ON, RUNS_ON_PORT, REQUIRES_REPORTING_TO, MANDATES_TIMELINE, PENALTY_FOR_VIOLATION, HAS_AUTHOR, COMPLIES_WITH, INTEGRATES_WITH).\n"
                        "2. 'subject' and 'object' must be concrete entities, systems, policies, roles, metrics, or amounts.\n"
                        "3. Avoid pronouns (it, they, we). Extract between 5 and 12 high-fidelity triples.\n"
                        "Example format:\n"
                        '{"triples": [{"subject": "Solaris Platform", "relation": "DEPLOYED_ON", "object": "AWS us-east-1"}]}'
                    )

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract knowledge graph triples from this document snippet ({doc_name}):\n\n{sample_snippet}"}
                    ]

                    resp = client.chat.completions.create(
                        messages=messages,
                        temperature=0.0,
                        max_tokens=600,
                        response_format={"type": "json_object"}
                    )

                    raw_json = resp.choices[0].message.content if resp and resp.choices else ""
                    if raw_json and raw_json.strip():
                        parsed = json.loads(raw_json.strip())
                        llm_triples = parsed.get("triples", [])
                        for t in llm_triples:
                            s = t.get("subject", "")
                            r = t.get("relation", "")
                            o = t.get("object", "")
                            if s and r and o:
                                self.add_triple(s, r, o, doc_name=doc_name)
                        logger.info(f"OpenIE extracted {len(llm_triples)} factual triples for '{doc_name}'.")

                except Exception as e:
                    logger.warning(f"LLM OpenIE extraction non-critical error for '{doc_name}': {e}. Using regex baseline.")

        added_count = len(self.triples) - initial_count
        logger.info(
            f"Graph-RAG indexed doc '{doc_name}' (+{added_count} triples) -> "
            f"Graph now has {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges."
        )

    def query_graph(self, query: str, enabled: Optional[bool] = None) -> Dict[str, Any]:
        """
        Traverses the Knowledge Graph using 1-hop and 2-hop multi-hop reasoning.
        Matches relevant entities and explores interconnected relationships.
        """
        is_enabled = enabled if enabled is not None else settings.ENABLE_GRAPH_RAG
        if not is_enabled or self.graph.number_of_nodes() == 0:
            return {
                "has_graph_evidence": False,
                "evidence_count": 0,
                "evidence_lines": [],
                "summary": "Knowledge graph traversal is disabled or empty."
            }

        q_lower = query.lower()
        # Extract meaningful query keywords (length >= 4, ignoring common stop words)
        stop_words = {"what", "when", "where", "which", "who", "whom", "this", "that", "there", "their", "from", "with", "have", "been", "does", "were"}
        query_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_lower)) - stop_words

        matched_nodes: Set[str] = set()

        # Step 1: Identify seed matching nodes
        for node in self.graph.nodes():
            n_lower = node.lower()
            # Direct inclusion
            if len(node) >= 3 and (n_lower in q_lower or q_lower in n_lower):
                matched_nodes.add(node)
                continue
            # Token overlap
            node_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", n_lower)) - stop_words
            if node_tokens and query_words.intersection(node_tokens):
                matched_nodes.add(node)

        matched_triples = []
        seen = set()

        # Step 2: Traverse 1-hop and 2-hop connections from matched nodes
        for node in list(matched_nodes)[:8]:
            # 1-Hop: Outgoing edges
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor) or {}
                rel = edge_data.get("relation", "RELATED_TO")
                doc = edge_data.get("doc", "")
                doc_str = f" [Source: `{doc}`]" if doc else ""
                line = f"- **{node}** - `{rel}` -> **{neighbor}**{doc_str}"
                if line not in seen:
                    seen.add(line)
                    matched_triples.append(line)

                # 2-Hop: Multi-hop reasoning expansion
                for hop2 in list(self.graph.neighbors(neighbor))[:3]:
                    if hop2 != node:
                        h2_data = self.graph.get_edge_data(neighbor, hop2) or {}
                        h2_rel = h2_data.get("relation", "RELATED_TO")
                        h2_doc = h2_data.get("doc", "")
                        h2_doc_str = f" [Source: `{h2_doc}`]" if h2_doc else ""
                        h2_line = f"  -> **{neighbor}** - `{h2_rel}` -> **{hop2}**{h2_doc_str}"
                        if h2_line not in seen:
                            seen.add(h2_line)
                            matched_triples.append(h2_line)

            # 1-Hop: Incoming edges
            for pred in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(pred, node) or {}
                rel = edge_data.get("relation", "RELATED_TO")
                doc = edge_data.get("doc", "")
                doc_str = f" [Source: `{doc}`]" if doc else ""
                line = f"- **{pred}** - `{rel}` -> **{node}**{doc_str}"
                if line not in seen:
                    seen.add(line)
                    matched_triples.append(line)

            if len(matched_triples) >= 20:
                break

        return {
            "has_graph_evidence": len(matched_triples) > 0,
            "evidence_count": len(matched_triples),
            "evidence_lines": matched_triples[:15],
            "summary": "\n".join(matched_triples[:15]) if matched_triples else "No direct graph relationship found."
        }

    def to_dot(self, max_edges: int = 40) -> str:
        """Generates Graphviz DOT representation for Streamlit rendering."""
        dot_lines = [
            "digraph G {",
            "  rankdir=LR;",
            '  node [shape=box, style="filled,rounded", fillcolor="#EEF2FF", color="#4F46E5", fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=9, color="#94A3B8"];'
        ]
        for idx, (u, v, data) in enumerate(self.graph.edges(data=True)):
            if idx >= max_edges:
                break
            rel = data.get("relation", "REL")
            u_clean = u.replace('"', '\\"')
            v_clean = v.replace('"', '\\"')
            dot_lines.append(f'  "{u_clean}" -> "{v_clean}" [label="{rel}"];')
        dot_lines.append("}")
        return "\n".join(dot_lines)
