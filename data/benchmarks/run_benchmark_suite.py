import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import AdaptiveRAGPipeline, AdaptiveRAGResponse
from src.retriever import HybridRetriever
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Confidence-RAG Benchmark Test Suite Runner")
    parser.add_argument(
        "--category",
        choices=["all", "graph", "multihop", "precision", "crag", "zero"],
        default="all",
        help="Select benchmark category to test"
    )
    parser.add_argument("--limit", type=int, default=5, help="Max questions to evaluate")
    args = parser.parse_args()

    suite_path = Path(__file__).parent / "benchmark_questions.json"
    if not suite_path.exists():
        print(f"Error: {suite_path} not found.")
        sys.exit(1)

    with open(suite_path, "r", encoding="utf-8") as f:
        suite = json.load(f)

    category_map = {
        "all": None,
        "graph": "knowledge_graph_and_entities",
        "multihop": "multi_hop_comparison",
        "precision": "single_hop_precision",
        "crag": "crag_evaluator_and_web_fallback",
        "zero": "zero_retrieval_parametric"
    }
    filter_cat = category_map.get(args.category)

    questions = suite["questions"]
    if filter_cat:
        questions = [q for q in questions if q["category"] == filter_cat]

    questions = questions[:args.limit]

    print("=" * 80)
    print("🚀 CONFIDENCE-RAG BENCHMARK TEST SUITE RUNNER")
    print(f"Version: {suite.get('version', '2.4.0')} | Total Tests Selected: {len(questions)}")
    print("=" * 80)

    # Initialize Retriever & Pipeline
    print("\n[1/3] Initializing Hybrid Retriever & Cross-Encoder Reranker...")
    retriever = HybridRetriever()

    # Pre-load documents into vault
    docs_dir = settings.DOCS_DIR
    txt_files = list(docs_dir.glob("*.txt"))
    pdf_files = list(docs_dir.glob("*.pdf"))

    print(f"[2/3] Ingesting Turnkey Documents from '{docs_dir}'...")
    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                retriever.add_text_document(txt.name, f.read())
            print(f"  + Ingested TXT: {txt.name}")
        except Exception as e:
            print(f"  - Failed TXT {txt.name}: {e}")

    for pdf in pdf_files:
        try:
            count_before = retriever.count()
            retriever.add_pdf_document(pdf, doc_name=pdf.name)
            print(f"  + Ingested PDF: {pdf.name} (+{retriever.count() - count_before} chunks)")
        except Exception as e:
            print(f"  - Failed PDF {pdf.name}: {e}")

    print(f"  -> Vault Total: {retriever.count()} Chunks, {retriever.graph_rag.graph.number_of_nodes()} Knowledge Graph Nodes\n")

    print("[3/3] Executing Cognitive Retrieval Evaluation...")
    pipeline = AdaptiveRAGPipeline(retriever=retriever)

    results = []
    print("-" * 80)
    print(f"{'ID':<8} | {'Category':<20} | {'Route':<14} | {'Verdict':<10} | {'Conf':<6} | {'Latency':<8}")
    print("-" * 80)

    for q_item in questions:
        qid = q_item["id"]
        cat = q_item["category"][:18]
        q_text = q_item["question"]
        mode = q_item.get("test_mode", "autonomous")

        start = time.perf_counter()
        resp: AdaptiveRAGResponse = pipeline.process_query(q_text, search_mode=mode)
        latency_ms = (time.perf_counter() - start) * 1000

        verdict_str = resp.crag_verdict or "DIRECT"
        conf_str = f"{resp.crag_confidence:.2f}" if resp.crag_confidence else "N/A"

        print(f"{qid:<8} | {cat:<20} | {resp.complexity.value:<14} | {verdict_str:<10} | {conf_str:<6} | {latency_ms:>6.0f}ms")

        results.append({
            "id": qid,
            "category": q_item["category"],
            "question": q_text,
            "complexity": resp.complexity.value,
            "crag_verdict": resp.crag_verdict,
            "crag_confidence": resp.crag_confidence,
            "retrieved_chunks": len(resp.retrieved_sources),
            "web_sources_count": len(resp.web_sources),
            "graph_evidence_count": len(resp.graph_evidence),
            "latency_ms": latency_ms,
            "answer_preview": resp.answer[:200] + "..."
        })

    print("-" * 80)
    out_log = Path(__file__).parent / "benchmark_run_log.json"
    with open(out_log, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

    print(f"\n✅ Benchmark run complete! Detailed log persisted to: {out_log}\n")

if __name__ == "__main__":
    main()
