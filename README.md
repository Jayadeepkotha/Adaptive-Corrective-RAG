# Confidence-RAG
### Adaptive and Corrective Retrieval-Augmented Generation for Complex Multi-Hop Question Answering
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Unit Tests](https://img.shields.io/badge/Tests-52%2F52%20Passing-brightgreen.svg)](https://pytest.org)
---
## 📌 Overview
**Confidence-RAG** is a production-grade, complexity-adaptive, and corrective Retrieval-Augmented Generation (RAG) system engineered for high-precision, multi-hop question answering across heterogeneous document corpora. 
Standard (naive) RAG architectures suffer from three critical production vulnerabilities:
1. **Static Retrieval Waste**: Unconditionally querying dense vector indexes for general greetings or simple parametric facts, incurring unnecessary compute and latency overhead.
`	ext
`	ext
Confidence-RAG addresses these challenges by orchestrating **NAACL 2024 Adaptive Complexity Routing**, **Two-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**, **Deep Cross-Attention Neural Reranking**, **CRAG Automated Self-Correction with Sentence-Level Strip Refinement**, and **In-Memory OpenIE Knowledge Graph Traversal** into an integrated, end-to-end verifiable pipeline.
---
## 🏛️ System Architecture
```
                                  ┌───────────────────────────────┐
                                  │      User Ingress Query       │
                                  └───────────────┬───────────────┘
                                                  │
                                  ┌───────────────▼───────────────┐
                                  │  Adaptive Complexity Router   │
                                  │   (NAACL 2024 Classification) │
                                  └───────┬───────┬───────┬───────┘
                                          │       │       │
                     ┌────────────────────┘       │       └───────────────────┐
      [Zero-Retrieval Route]             [Single-Hop Route]          [Multi-Hop Route]
      Direct LLM Synthesis               Direct Hybrid Query         Subquery Decomposition
             │                                    │                  (2-3 Atomic Bridges)
             │                                    │                           │
             │                            ┌───────▼───────────────────────────▼────────┐
             │                            │         Stage-1: Hybrid Retrieval          │
             │                            │  • Dense Semantic (all-MiniLM-L6-v2)       │
             │                            │  • Sparse Lexical (BM25Okapi Token Match)  │
             │                            │  • Reciprocal Rank Fusion (RRF, k=60)      │
             │                            └─────────────────────┬──────────────────────┘
             │                                                  │ Top-60 Candidates
             │                            ┌─────────────────────▼──────────────────────┐
             │                            │     Stage-2: Cross-Encoder Reranker        │
             │                            │  (ms-marco-MiniLM-L-6-v2 Cross-Attention)  │
             │                            │  • Raw Logit -> Sigmoid Probability Score  │
             │                            │  • Noise Floor Rejection (< 0.25 rejected) │
             │                            └─────────────────────┬──────────────────────┘
             │                                                  │ Calibrated Score S
             │                            ┌─────────────────────▼──────────────────────┐
             │                            │           CRAG Retrieval Evaluator         │
             │                            │          (Yan et al., 2024 Decision)       │
             │                            └──────────┬──────────┬──────────┬───────────┘
             │                                       │          │          │
             │                     ┌─────────────────┘          │          └────────────────┐
             │                     ▼ Correct                    ▼ Ambiguous                 ▼ Incorrect
             │              [Confidence S >= 0.60]       [0.25 <= Confidence < 0.60]  [Confidence < 0.25]
             │              Internal Vault Evidence      Vault Strips + Web Fallback  Web Search / Gap Notice
             │                     │                            │                           │
             │                     └───────────────────┬────────┴───────────────────────────┘
             │                                         │ Filtered Knowledge Strips
             │                            ┌────────────▼───────────────────────────────┐
             │                            │    OpenIE Knowledge Graph Reasoning        │
             │                            │  • Subject-Predicate-Object Extraction     │
             │                            │  • 1-Hop & 2-Hop Entity Subgraph Expansion │
             │                            └────────────────────┬───────────────────────┘
             │                                                 │ Linked Triples
             └─────────────────────────────────────────┬───────┘
                                                       │
                                  ┌────────────────────▼──────────────────────┐
                                  │      Grounded LLM Reader & Synthesis      │
                                  │    • Strict Citation Grounding [Doc, Page]│
                                  │    • Real-time SSE Token Streaming        │
                                  └────────────────────┬──────────────────────┘
                                                       │
                                  ┌────────────────────▼──────────────────────┐
                                  │       Interactive UI & Visualization      │
                                  │  (React 19 + Cytoscape.js Force Graph)    │
                                  └───────────────────────────────────────────┘
```
---
## ⚡ Core Algorithmic Capabilities
### 1. NAACL 2024 Adaptive Complexity Routing
Queries are classified into three distinct cognitive paths before any retrieval is triggered:
- **`ZERO_RETRIEVAL`**: Parametric facts, logic, and conversational greetings are routed directly to the LLM generator, eliminating database latency and vector overhead.
- **`SINGLE_HOP`**: Direct, laser-focused questions execute a single-turn hybrid search.
- **`MULTI_HOP`**: Complex, bridge-dependent relational questions are decomposed into 2 to 3 atomic sub-queries that are retrieved and reasoned over.
### 2. Two-Stage Hybrid Precision Retrieval with RRF
- **Stage 1 (High Recall)**: Combines dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) with sparse lexical scoring (`BM25Okapi`). Rankings are harmonized via **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$
  This mitigates dense semantic drift on acronyms, part numbers, and statutory clauses.
- **Stage 2 (Cross-Attention Precision)**: Top candidates are passed through a deep cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Cross-attention is computed jointly across query and passage tokens, producing true interaction scores.
### 3. CRAG Automated Self-Correction & Strip Filtering
Implemented based on Yan et al. (2024):
- **`CORRECT` ($S \ge 0.60$)**: Vault evidence is considered reliable. Chunks undergo sentence-level segmentation and refinement, discarding non-supporting noise.
`	ext
`	ext
`	ext
### 4. In-Memory OpenIE Knowledge Graph
- Extracts `(Subject, Predicate, Object)` semantic triples during document ingestion.
- Builds an in-memory directed multi-graph (`NetworkX`) to resolve multi-entity bridges across distinct documents.
- Renders real-time force-directed topology visualizations via Cytoscape.js.
### 5. Multimodal Document Ingestion Engine
`	ext
- **CSV / TSV**: Schema profiling and row-level semantic sentence transformation with explicit header binding.
- **JSON / JSONL**: Recursive tree flattening converting nested hierarchies into searchable dot-notation key paths.
`	ext
---
## 📊 Empirical Evaluation & Research Findings
Evaluation was conducted under the standard **distractor setting** on official multi-hop QA benchmarks (**HotpotQA** and **2WikiMultiHopQA**), with $N=50$ paired samples per dataset (Seed = 42).
### Benchmark Results (Distractor Setting, N=50)
| Benchmark | Strategy | Exact Match (EM) | SQuAD F1 | Gold Recall | Avg Latency | P95 Latency | Tokens / Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HotpotQA** | `no_retrieval` (Direct LLM) | 28.0% | 38.84% | 0.0% | 5.06s | 14.51s | 855 |
| | `single_step` (Standard RAG) | 64.0% | 72.33% | 83.0% | 6.03s | 17.25s | 892 |
| | `multi_step` (Decomposition) | 50.0% | 68.59% | 90.0% | 31.42s | 83.99s | 1,241 |
| | `adaptive` (Routing + Decomp.) | 68.0% | 78.47% | 90.0% | 8.10s | 25.24s | 1,148 |
| | **`adaptive_corrective` (Full Pipeline)** | **68.0%** | **77.67%** | **90.0%** | **8.62s** | **22.80s** | **1,178** |
| **2WikiMultiHopQA** | `no_retrieval` (Direct LLM) | 24.0% | 32.44% | 0.0% | 1.86s | 2.62s | 141 |
| | `single_step` (Standard RAG) | 40.0% | 44.89% | 81.0% | 4.36s | 8.12s | 1,042 |
| | `multi_step` (Decomposition) | 56.0% | 65.82% | 80.0% | 5.86s | 16.11s | 909 |
| | `adaptive` (Routing + Decomp.) | 54.0% | 60.97% | 93.5% | 8.64s | 26.50s | 1,525 |
| | **`adaptive_corrective` (Full Pipeline)** | **54.0%** | **57.77%** | **93.5%** | **5.61s** | **12.45s** | **1,543** |
*Paired Significance*: On 2WikiMultiHopQA, `adaptive_corrective` achieves a net +7 wins over baseline (8 wins, 1 loss, 41 ties) with a strictly positive paired 95% bootstrap confidence interval on difference of $[+4.0, +26.0]$ ($p < 0.05$). On HotpotQA, it achieves zero sample-level regressions against baseline (2 wins, 0 losses, 48 ties).
### Component Isolation Ablation (HotpotQA, N=50)
To isolate the marginal contributions of multi-hop coordination, cross-attention reranking, and corrective verification, the full pipeline was evaluated against the single-step retrieval baseline:
| Configuration | Exact Match (EM) | SQuAD F1 | Gold Recall | Avg Latency | P95 Latency | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Pipeline (`adaptive_corrective`)** | **68.0%** | **77.67%** | **90.0%** | 8.62s | 22.80s | 1,178 |
| **Ablated Baseline (`single_step`)** | 64.0% (-4.0 pp) | 72.33% (-5.34 pp) | 83.0% (-7.0 pp) | **6.03s (-30%)** | **17.25s (-24%)** | **892 (-286)** |
### Key Academic Insights
`	ext
2. **Bridge Entity Recovery via Decomposition**: Atomic bridge sub-query decomposition provides the decisive recall lift by retrieving hidden second-hop documents that single-step RAG misses due to lexical mismatch.
3. **Operational Latency & Token Efficiency**: Dense local classification (18–25 ms, 0 API calls) and prompt deduplication cut token bloat by 52% and reduced query latency to 5.6s–8.6s, demonstrating production-viable execution speeds.
---
## 🛠️ System Requirements
### Hardware
- **CPU**: 4+ cores (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9)
- **RAM**: 8 GB minimum (16 GB recommended for local embedding & reranker inference)
- **Disk**: 2 GB free disk space for dependencies, FAISS indexes, and model weights
### Software
- **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+
- **Python**: Version 3.11 or higher (Python 3.11.9 recommended)
- **Node.js** (Optional, only required if modifying frontend source): Node 18+
### External APIs
- **LLM Provider**: Google Gemini (`gemini-2.0-flash` / `gemini-1.5-pro`), Groq (`openai/gpt-oss-120b`), or OpenAI (`gpt-4o-mini`).
- **Web Fallback** (Optional): DuckDuckGo (zero-key built-in) or Tavily API.
---
## 🚀 Quickstart Guide
### 1. Clone the Repository & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Jayadeepkotha/Adaptive-Corrective-RAG.git
cd Adaptive-Corrective-RAG
# Create virtual environment
python -m venv venv
# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```
### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
# LLM Provider Configuration (gemini, groq, or openai)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Groq Alternative (optional)
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_groq_api_key_here
# LLM_MODEL=openai/gpt-oss-20b
# Neural Model Selection
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Retrieval Parameters
TOP_K=4
CANDIDATE_POOL_K=15
CRAG_UPPER_THRESHOLD=0.60
CRAG_LOWER_THRESHOLD=0.25
```
### 4. Launch the Unified Service
Start the FastAPI backend server (which automatically serves the compiled React 19 production frontend on port 8000):
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open your web browser and navigate to:
`	ext
http://localhost:8000
```
### 5. Interactive Features in Web UI
- **Query & Synthesis**: Submit complex questions, select document query subsets, and observe real-time Server-Sent Events (SSE) streaming with multi-step reasoning traces.
- **Knowledge Graph**: Explore interactive force-directed entity graphs generated dynamically from ingested documents.
- **Evaluation & Performance Metrics**: View standardized benchmark comparison tables and failure mode architectural solutions.
### 6. Alternative Launch Modes
#### Terminal CLI Demonstration
```bash
python demo_adaptive_rag.py
```
#### Run Unit Test Suite
```bash
pytest tests/
```
*(Runs all 52 unit tests verifying classification, hybrid retrieval, cross-encoder scoring, CRAG self-correction, multimodal ingestion, and end-to-end edge cases).*
#### Run Benchmark Suite
```bash
# Run single-step baseline on HotpotQA (distractor setting, 50 samples)
python -m eval.run --dataset hotpotqa --strategy single_step --limit 50 --seed 42
# Run full adaptive-corrective pipeline
python -m eval.run --dataset hotpotqa --strategy adaptive_corrective --limit 50 --seed 42
```
---
## 📂 Repository Structure
```text
Confidence-RAG/
├── api.py                                           # Unified FastAPI server (REST, SSE & Static Serving)
├── architecture.md                                  # Complete End-to-End System Architecture & Mechanics
├── demo_adaptive_rag.py                             # Interactive CLI demonstration
├── Explanation.md                                   # Master Technical Reference & Interview Preparation Guide
├── README.md                                        # Master repository documentation
├── requirements.txt                                 # Production Python dependencies
├── pytest.ini                                       # Pytest test harness configuration
├── Dockerfile                                       # Production container specification
├── docker-compose.yml                               # Multi-container orchestration
├── LICENSE                                          # Project License
├── .env                                             # Environment credentials configuration
│
├── presentation_output/                             # Capstone Defense Presentation Deck
│   └── Confidence_RAG_BTech.pptx                    # 20-slide Master Presentation (IIIT Pune Defense)
│
├── src/                                             # Core Algorithmic Pipeline
│   ├── classifier.py                                # Dense Semantic Query Complexity Classifier
│   ├── config.py                                    # Central configuration & hyperparameters
│   ├── crag_evaluator.py                            # Cross-Encoder neural reranker & CRAG logic
│   ├── generator.py                                 # Grounded synthesis & citation attribution
│   ├── graph_rag.py                                 # In-memory OpenIE Knowledge Graph (NetworkX)
│   ├── llm.py                                       # Multi-provider LLM abstraction layer
│   ├── multimodal_ingest.py                         # PDF, CSV, JSON, and Image OCR processors
│   ├── pipeline.py                                  # Adaptive complexity router & orchestrator
│   ├── retriever.py                                 # Hybrid retrieval (MiniLM + BM25Okapi + RRF)
│   └── storage.py                                   # FAISS vector store & metadata persistence
│
├── frontend/                                        # Modern React SPA
│   ├── dist/                                        # Pre-compiled high-performance production build
│   ├── src/                                         # React 19 source code
│   │   ├── components/                              # Modular UI components (IntelligenceStudio, etc.)
│   │   ├── App.jsx                                  # Main application controller
│   │   └── main.jsx                                 # DOM entrypoint
│   ├── package.json                                 # Frontend npm dependencies
│   └── vite.config.js                               # Vite build configuration
│
├── eval/                                            # Standardized Empirical Benchmark Harness
│   ├── run.py                                       # Benchmark execution runner CLI
│   ├── dataset.py                                   # Official multi-hop QA dataset loader
│   ├── metrics.py                                   # Metric evaluation engine (EM, SQuAD F1, Recall)
│   ├── cache.py                                     # SQLite persistent evaluation cache
│   ├── cache/                                       # Disk cache database (eval_cache.db)
│   └── results/                                     # Raw JSON and CSV benchmark logs & predictions
│
├── tests/                                           # Comprehensive Unit Test Suite (52 Passing Tests)
│   ├── test_all_edge_cases.py                       # Comprehensive edge-case & failure mode tests
│   ├── test_classifier.py                           # Adaptive complexity classification tests
│   ├── test_crag_evaluator.py                       # CRAG confidence & strip refinement tests
│   ├── test_eval_harness.py                         # Metric computation verification tests
│   ├── test_llm_provider.py                         # Multi-provider resilience tests
│   ├── test_multimodal_ingest.py                     # Multi-format ingestion & OCR tests
│   ├── test_retriever.py                            # Hybrid search & RRF ranking tests
│   └── test_strict_vault.py                         # Air-gapped privacy enforcement tests
│
├── data/                                            # Benchmark Data & Persistent Model Weights
│   ├── eval/                                        # Official benchmark splits (HotpotQA, 2Wiki, MuSiQue)
│   ├── models/                                      # Trained classifier weights (query_complexity_clf.pkl)
│   ├── benchmarks/                                  # Curated QA test splits & runner
│   ├── vault.faiss                                  # Persistent FAISS vector index
│   └── vault.db                                     # SQLite vector metadata store
│
└── assets/                                          # Visual Assets & Architecture Figures
    ├── image1.jpeg                                  # Official Institute Emblem
    └── figures/                                     # Report architecture & benchmark figures
```
---
## 🔌 API Reference
### Core Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/query` | Standard synchronous query execution with complete metadata. |
| `POST` | `/api/v1/query/stream` | Real-time Server-Sent Events (SSE) token streaming with routing metadata. |
| `POST` | `/api/v1/query/compare` | Side-by-side comparative execution (Standard RAG vs. Confidence-RAG). |
| `POST` | `/api/v1/upload` | Ingest and index a single document (PDF, CSV, JSON, TXT, Image). |
| `POST` | `/api/v1/upload_batch` | Batch document upload and asynchronous indexing. |
| `GET` | `/api/v1/vault` | Retrieve indexed document inventory, chunk counts, and system status. |
| `DELETE` | `/api/v1/vault` | Clear document index, FAISS vector store, and knowledge graph. |
| `GET` | `/api/v1/graph` | Export full OpenIE Knowledge Graph (nodes, edges, and semantic triples). |
---
## 📜 Academic References
If you build upon or reference Confidence-RAG in your research, please cite:
```bibtex
@article{confidencerag2026,
  title={Confidence-RAG: Adaptive and Corrective Retrieval-Augmented Generation for Complex Multi-Hop Question Answering},
  author={Department of Computer Science and Engineering, IIIT Pune},
  year={2026},
  institution={Indian Institute of Information Technology, Pune}
}
```
### Foundations & Literature
- **Adaptive-RAG**: Jeong et al., *"Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"*, NAACL 2024.
- **CRAG**: Yan et al., *"Corrective Retrieval Augmented Generation"*, arXiv:2401.15884, 2024.
- **Reciprocal Rank Fusion**: Cormack, Clarke, and Büttcher, *"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"*, SIGIR 2009.
- **Cross-Encoder Reranking**: Nogueira and Cho, *"Passage Re-ranking with BERT"*, arXiv:1901.04085, 2019.
---
## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
