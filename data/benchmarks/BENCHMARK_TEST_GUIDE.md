# 📊 Confidence-RAG Benchmark Test Suite & Evaluation Guide

This directory contains gold-standard benchmark datasets, research papers (PDFs), enterprise policies, and curated evaluation questions designed to test all capabilities of **Confidence-RAG**.

---

## 📁 Benchmark Assets in `data/`

```
data/
├── docs/                                    # Turnkey Knowledge Vault (PDFs & Policies)
│   ├── adaptive_rag_naacl2024.pdf           # Official NAACL 2024 Research Paper (20 pages)
│   ├── crag_self_correction_2024.pdf        # Official CRAG Self-Correction Paper (16 pages)
│   ├── attention_is_all_you_need.pdf        # Canonical Transformer Paper (Vaswani et al., 15 pages)
│   ├── cloud_infrastructure_sla.txt         # Multi-Cloud Enterprise SLA Agreement
│   ├── soc2_cybersecurity_policy.txt        # SOC 2 Type II Security & Encryption Policy
│   └── enterprise_vendor_msa.txt            # Master Services Agreement (Zero Model Retraining)
│
├── benchmarks/                              # Benchmark Datasets & Question Sets
│   ├── benchmark_questions.json             # 20 Curated Multi-Category Test Questions
│   ├── hotpotqa_50_eval.json                # 50 Multi-Hop Questions from official HotpotQA Dev
│   ├── 2wikimultihop_50_eval.json           # 50 Reasoning-Chain Questions from 2WikiMultiHopQA
│   ├── run_benchmark_suite.py               # Automated Benchmark Scorecard Runner
│   └── BENCHMARK_TEST_GUIDE.md              # This Comprehensive Guide
│
└── benchmark_results.json                   # Quantitative RAG vs. Vanilla Baseline Scores
```

---

## 🎯 Benchmark Evaluation Categories

The test questions in [`benchmark_questions.json`](benchmark_questions.json) are organized into 5 functional categories:

| Category | Description | Primary Engine Tested |
| :--- | :--- | :--- |
| **`knowledge_graph_and_entities`** | Entity traversal, relationship triples (`SUBJECT -[RELATION]-> OBJECT`), Graphviz topology diagram. | KnowledgeGraphRAG & In-Memory Graph |
| **`multi_hop_comparison`** | Complex queries comparing two documents, architectures, or legal clauses. | Adaptive-RAG Query Decomposition & Synthesizer |
| **`single_hop_precision`** | Laser-accurate retrieval of formulas, exact penalty tiers, and compliance schedules. | Stage-2 Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **`crag_evaluator_and_web_fallback`**| Tests self-correction confidence scores (`CORRECT`, `AMBIGUOUS`, `INCORRECT`), DuckDuckGo web fallback, and Air-Gapped Strict Vault isolation. | CRAG Evaluator & WebSearchFallback |
| **`zero_retrieval_parametric`** | Direct math, code, or conversational requests requiring zero retrieval overhead. | Adaptive Complexity Classifier Gate |

---

## 🧪 How to Run the Tests

### Method 1: Interactive Streamlit UI Studio (Visual)
1. Open [`http://localhost:8501`](http://localhost:8501) in your browser.
2. In the left sidebar:
   - Click **"📂 Ingest Default Enterprise Policies"** or drag-and-drop any PDF from `data/docs/` (e.g. `adaptive_rag_naacl2024.pdf`).
   - Notice the live **Vault Chunks** and **Graph Nodes** counters update immediately!
3. Go to **`💬 Intelligence Studio`**:
   - Paste any question from [`benchmark_questions.json`](benchmark_questions.json).
   - Observe the **Verified Grounding Confidence Meter**, **CRAG Verdict badge**, and **Decomposed Task Graph**.
4. Go to **`🕸️ Knowledge Graph Explorer`**:
   - Inspect the interactive Graphviz topology diagram and searchable triples table.
5. Go to **`📈 Observatory & Benchmark`**:
   - Review live metrics, latency telemetry, and the Vanilla RAG vs. Confidence-RAG comparison matrix.

---

### Method 2: Automated Benchmark Scorecard CLI (Single Command)
Run the automated test runner in your terminal:
```bash
# Run all benchmark categories
python data/benchmarks/run_benchmark_suite.py --category all

# Or run a specific category
python data/benchmarks/run_benchmark_suite.py --category graph
python data/benchmarks/run_benchmark_suite.py --category multihop
python data/benchmarks/run_benchmark_suite.py --category precision
python data/benchmarks/run_benchmark_suite.py --category crag
```
This produces a structured terminal report detailing:
- Query latency (ms)
- Complexity route assigned
- CRAG verdict & confidence score
- Chunks and graph triples retrieved
- Grounded answer preview

---

### Method 3: Headless Production REST API (cURL / Python)
With the FastAPI server running at [`http://localhost:8000`](http://localhost:8000):

#### 1. Ingest a Benchmark PDF:
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@data/docs/adaptive_rag_naacl2024.pdf"
```

#### 2. Query in Strict Air-Gapped Vault Mode:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the three complexity levels defined in the Adaptive-RAG paper?",
    "search_mode": "strict_vault"
  }'
```

#### 3. Inspect Knowledge Graph Triples:
```bash
curl -X GET "http://localhost:8000/api/v1/graph"
```
Or open the interactive Swagger UI at **[`http://localhost:8000/docs`](http://localhost:8000/docs)**.
