import React, { useState } from 'react';
import {
  Send, Sparkles, CheckCircle2, AlertTriangle, XCircle,
  FileText, Globe, Layers, Download, Copy, Check, Clock,
  Zap, Swords, ArrowRight, ShieldCheck, Database, RefreshCw
} from 'lucide-react';

export default function IntelligenceStudio({
  onRunQuery,
  onRunCompare,
  comparisonResult,
  onClearComparison,
  isProcessing,
  currentResult,
  queryHistory = []
}) {
  const [queryInput, setQueryInput] = useState('');
  const [copied, setCopied] = useState(false);
  const [isCompareMode, setIsCompareMode] = useState(false);

  // Curated 4-Step Academic / College Demonstration Suite
  const presets = [
    {
      badge: 'Demo 1 • Adaptive Zero-Retrieval',
      title: '⚡ General Knowledge Query',
      desc: 'Bypasses vector search & cuts 85% latency',
      query: 'What is the fundamental difference between supervised and unsupervised machine learning?'
    },
    {
      badge: 'Demo 2 • Hybrid BM25 + Cross-Encoder',
      title: '🎯 Exact Policy & Section Lookup',
      desc: 'BM25 locks on keywords + Reranker removes noise',
      query: 'What is the statutory penalty for data encryption violations and mandatory breach reporting timeline?'
    },
    {
      badge: 'Demo 3 • Multi-Hop & Knowledge Graph',
      title: '🕸️ Multi-Document Synthesis',
      desc: 'Decomposes query & traverses graph entity paths',
      query: 'Which cloud region does Project Solaris run on, and what database does it connect to?'
    },
    {
      badge: 'Demo 4 • Corrective RAG (CRAG)',
      title: '🛡️ Self-Correction & Fallback',
      desc: 'Detects vault gap and auto-pivots to web search',
      query: 'What are the core requirements and risk tiers of the European Union AI Act?'
    }
  ];

  const handlePresetClick = (q) => {
    setQueryInput(q);
    if (onClearComparison) onClearComparison();
    if (isCompareMode && onRunCompare) {
      onRunCompare(q);
    } else {
      onRunQuery(q);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!queryInput.trim() || isProcessing) return;
    if (onClearComparison) onClearComparison();
    if (isCompareMode && onRunCompare) {
      onRunCompare(queryInput.trim());
    } else {
      onRunQuery(queryInput.trim());
    }
  };

  const handleCopyAnswer = (text) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadBrief = () => {
    if (!currentResult) return;
    const content = `# Evidence & Synthesis Brief\n\n**Query**: ${currentResult.query}\n**Complexity Route**: ${currentResult.complexity}\n**Evidence Verdict**: ${currentResult.crag_verdict || 'N/A'} (Confidence: ${((currentResult.crag_confidence || 1) * 100).toFixed(1)}%)\n**Processing Latency**: ${currentResult.latency_ms?.toFixed(0)} ms\n\n---\n\n## Synthesized Answer\n\n${currentResult.answer}\n\n---\n\n## Retrieved Evidence\n${currentResult.retrieved_sources?.map(s => `- **${s.doc_name}** (Page/Row ${s.page}): ${(s.snippet || s.text || '').slice(0, 150)}...`).join('\n')}\n`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `confidence_rag_brief_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getVerdictBadge = (verdict, conf) => {
    if (verdict === 'CORRECT') {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-medium">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>EVIDENCE: SUFFICIENT LOCAL ({((conf || 1) * 100).toFixed(0)}%)</span>
        </div>
      );
    }
    if (verdict === 'AMBIGUOUS') {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-xs font-medium">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
          <span>EVIDENCE: AMBIGUOUS (Web-Augmented)</span>
        </div>
      );
    }
    if (verdict === 'INCORRECT') {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-50 border border-rose-200 text-rose-800 text-xs font-medium">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
          <span>EVIDENCE: INSUFFICIENT (Web Fallback)</span>
        </div>
      );
    }
    if (currentResult?.complexity === 'MULTI_HOP') {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-800 text-xs font-medium">
          <span>ROUTE: MULTI-HOP DECOMPOSITION</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-medium">
        <span>ROUTE: ZERO-RETRIEVAL (PARAMETRIC)</span>
      </div>
    );
  };

  return (
    <div className="flex-1 flex flex-col p-6 max-w-6xl mx-auto w-full gap-6 overflow-y-auto">
      {/* 1. College Demo Suite Quick-Action Chips */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-blue-600" />
            <span>Academic Demonstration Suite (4 RAG Concepts)</span>
          </label>
          <span className="text-[11px] text-slate-500 font-medium">
            Click any demo to test architecture live
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handlePresetClick(p.query)}
              disabled={isProcessing}
              className="text-left p-3.5 rounded-xl bg-white border border-slate-200 hover:border-blue-300 hover:bg-blue-50/40 transition group cursor-pointer shadow-xs flex flex-col justify-between"
            >
              <div>
                <div className="text-[10px] uppercase font-bold text-blue-700 tracking-wider">
                  {p.badge}
                </div>
                <div className="text-xs font-semibold text-slate-900 group-hover:text-blue-700 transition mt-1">
                  {p.title}
                </div>
              </div>
              <div className="text-[11px] text-slate-500 mt-2 line-clamp-2">{p.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 2. Main Query Input Bar with Live Comparison Toggle */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="relative flex items-center bg-white rounded-xl border border-slate-300 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-100 shadow-xs transition">
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Ask a question across multi-modal documents, or run a live comparative benchmark..."
            className="w-full bg-transparent px-5 py-4 text-sm text-slate-900 placeholder-slate-400 focus:outline-none"
            disabled={isProcessing}
          />
          <div className="flex items-center gap-2 mr-3">
            {/* Compare Toggle */}
            <button
              type="button"
              onClick={() => setIsCompareMode(!isCompareMode)}
              className={`px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                isCompareMode
                  ? 'bg-amber-100 text-amber-900 border border-amber-300 shadow-xs'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200'
              }`}
              title="Toggle side-by-side head-to-head comparison against standard Naive RAG"
            >
              <Swords className="w-3.5 h-3.5" />
              <span>{isCompareMode ? 'Compare Mode ON' : 'Compare vs Naive'}</span>
            </button>

            {/* Run Button */}
            <button
              type="submit"
              disabled={!queryInput.trim() || isProcessing}
              className={`px-5 py-2.5 rounded-lg text-white font-medium text-xs flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition shadow-xs ${
                isCompareMode ? 'bg-amber-600 hover:bg-amber-700' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>{isCompareMode ? 'Run Live Arena' : 'Search & Stream'}</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* 3. Live Processing Pipeline State */}
      {isProcessing && !currentResult?.answer && !comparisonResult && (
        <div className="p-4 rounded-xl bg-blue-50/80 border border-blue-200 flex items-center gap-4 animate-pulse">
          <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center text-blue-600">
            <Sparkles className="w-4 h-4 animate-spin" />
          </div>
          <div>
            <div className="text-xs font-semibold text-blue-950">
              {isCompareMode
                ? 'Running Head-to-Head Comparative Benchmark (Naive RAG vs. Confidence-RAG)...'
                : 'Adaptive Retrieval & Evidence Evaluation in Progress...'}
            </div>
            <div className="text-[11px] text-slate-600 mt-0.5">
              Query Complexity Classification → Hybrid Bi-Encoder / BM25 Recall → Cross-Encoder Reranking → CRAG Calibrated Evidence Verdict
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 4. LIVE COMPARISON ARENA: SIDE-BY-SIDE BENCHMARK RESULT        */}
      {/* ------------------------------------------------------------- */}
      {comparisonResult && (
        <div className="space-y-4">
          {/* Arena Header & Telemetry Delta */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-950 text-white shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-300">
                <Swords className="w-4 h-4" />
                <span>Live Head-to-Head Architectural Demonstration</span>
              </div>
              <p className="text-xs text-slate-200">
                Query: <span className="font-semibold text-white">"{comparisonResult.query}"</span>
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="px-3 py-1.5 rounded-lg bg-white/10 backdrop-blur-xs text-xs font-medium border border-white/10 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-blue-300" />
                <span>
                  Latency Delta: <strong className="text-emerald-300 font-bold">{Math.abs(comparisonResult.telemetry_delta?.latency_savings_ms || 0)} ms</strong>
                </span>
              </div>
              <button
                onClick={onClearComparison}
                className="px-2.5 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-xs text-white border border-white/20 transition flex items-center gap-1"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Reset</span>
              </button>
            </div>
          </div>

          {/* Two-Column Side-by-Side Comparison Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left Column: Standard Naive RAG */}
            <div className="p-5 rounded-xl bg-white border-2 border-rose-200/80 shadow-xs space-y-4 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-rose-100 pb-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                    <span className="text-xs font-bold text-rose-900 uppercase tracking-wide">
                      Traditional Naive RAG
                    </span>
                  </div>
                  <span className="text-xs font-mono font-bold text-slate-600 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded">
                    {comparisonResult.naive_rag?.latency_ms?.toFixed(0)} ms
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono text-slate-600 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <div>• Router: <span className="text-rose-700 font-bold">None (Blind)</span></div>
                  <div>• Reranker: <span className="text-rose-700 font-bold">None (Cosine)</span></div>
                  <div>• Self-Correction: <span className="text-rose-700 font-bold">Disabled</span></div>
                  <div>• Knowledge Graph: <span className="text-rose-700 font-bold">Disabled</span></div>
                </div>

                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Generated Output (Uncalibrated):
                  </div>
                  <div className="p-3.5 rounded-lg bg-rose-50/40 border border-rose-200/60 text-xs text-slate-800 leading-relaxed font-sans min-h-[140px] whitespace-pre-wrap">
                    {comparisonResult.naive_rag?.answer}
                  </div>
                </div>

                {/* Naive Retrieved Sources */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Retrieved Dense Chunks ({comparisonResult.naive_rag?.retrieved_sources?.length || 0}):
                  </div>
                  <div className="space-y-1 max-h-36 overflow-y-auto">
                    {comparisonResult.naive_rag?.retrieved_sources?.map((s, idx) => (
                      <div key={idx} className="p-2 rounded bg-slate-50 border border-slate-200 text-[11px] space-y-0.5">
                        <div className="flex justify-between font-mono text-slate-600 font-semibold">
                          <span>{s.doc_name}</span>
                          <span className="text-rose-600">Sim: {s.score}</span>
                        </div>
                        <p className="text-slate-500 line-clamp-1 font-sans">{s.snippet}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 italic">
                ⚠️ Blind top-k retrieval: Cannot reject irrelevant chunks, bypass simple queries, or self-correct errors.
              </div>
            </div>

            {/* Right Column: Confidence-RAG Multi-Stage Architecture */}
            <div className="p-5 rounded-xl bg-white border-2 border-blue-300 shadow-xs space-y-4 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-blue-100 pb-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-600"></span>
                    <span className="text-xs font-bold text-blue-900 uppercase tracking-wide">
                      Confidence-RAG (Adaptive-CRAG + Graph)
                    </span>
                  </div>
                  <span className="text-xs font-mono font-bold text-blue-800 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded">
                    {comparisonResult.adaptive_rag?.latency_ms?.toFixed(0)} ms
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono text-slate-600 bg-blue-50/50 p-2.5 rounded-lg border border-blue-200/80">
                  <div>• Route: <span className="text-blue-700 font-bold">{comparisonResult.adaptive_rag?.complexity}</span></div>
                  <div>• Reranker: <span className="text-blue-700 font-bold">Cross-Encoder (ms-marco)</span></div>
                  <div>• CRAG Verdict: <span className="text-blue-700 font-bold">{comparisonResult.adaptive_rag?.crag_verdict || 'CALIBRATED'}</span></div>
                  <div>• Knowledge Graph: <span className="text-blue-700 font-bold">{comparisonResult.adaptive_rag?.graph_evidence?.length > 0 ? 'Active' : 'Standby'}</span></div>
                </div>

                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Grounded Synthesized Output:
                  </div>
                  <div className="p-3.5 rounded-lg bg-blue-50/30 border border-blue-200 text-xs text-slate-800 leading-relaxed font-sans min-h-[140px] whitespace-pre-wrap">
                    {comparisonResult.adaptive_rag?.answer}
                  </div>
                </div>

                {/* Confidence-RAG Retrieved Sources & Graph Evidence */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    Cross-Encoder Verified Evidence:
                  </div>
                  <div className="space-y-1 max-h-36 overflow-y-auto">
                    {comparisonResult.adaptive_rag?.retrieved_sources?.map((s, idx) => (
                      <div key={idx} className="p-2 rounded bg-slate-50 border border-slate-200 text-[11px] space-y-0.5">
                        <div className="flex justify-between font-mono text-slate-700 font-semibold">
                          <span className="text-blue-700">📄 {s.doc_name}</span>
                          <span className="text-emerald-700">Calibrated: {s.score}</span>
                        </div>
                        <p className="text-slate-600 line-clamp-1 font-sans">{s.snippet}</p>
                      </div>
                    ))}
                    {comparisonResult.adaptive_rag?.graph_evidence?.map((g, idx) => (
                      <div key={`g_${idx}`} className="p-2 rounded bg-indigo-50/50 border border-indigo-200 text-[11px] text-indigo-900">
                        {g}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-blue-100 text-[11px] text-blue-700 font-medium">
                ✅ Multi-stage verification: Rejects noise, traverses relational graphs, and falls back to web if internal vault is insufficient.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 5. NORMAL SINGLE-QUERY PRESENTATION WITH STREAMING             */}
      {/* ------------------------------------------------------------- */}
      {!comparisonResult && currentResult && (currentResult.answer || currentResult.retrieved_sources?.length > 0 || !isProcessing) && (
        <div className="space-y-4">
          {/* Telemetry Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl bg-white border border-slate-200 shadow-xs">
            <div className="flex items-center gap-3">
              {getVerdictBadge(currentResult.crag_verdict, currentResult.crag_confidence)}
              <div className="text-xs text-slate-600 font-medium">
                {currentResult.route_name}
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <Clock className="w-3.5 h-3.5" />
                <span>{currentResult.latency_ms?.toFixed(0)} ms</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleCopyAnswer(currentResult.answer)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs text-slate-700 transition cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-500" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={handleDownloadBrief}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 hover:bg-blue-100 text-xs text-blue-700 font-medium transition cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export Brief</span>
                </button>
              </div>
            </div>
          </div>

          {/* Synthesized Answer Card */}
          <div className="p-6 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Synthesized Response
              </h3>
              {isProcessing && (
                <div className="flex items-center gap-1.5 text-xs text-blue-600 font-medium">
                  <span className="w-2 h-2 rounded-full bg-blue-600 animate-ping"></span>
                  <span>Streaming tokens in real-time...</span>
                </div>
              )}
            </div>
            <div className="prose prose-slate prose-sm max-w-none text-slate-800 leading-relaxed whitespace-pre-wrap font-sans">
              {currentResult.answer}
              {isProcessing && (
                <span className="inline-block w-1.5 h-4 ml-1 bg-blue-600 animate-pulse align-middle" />
              )}
            </div>
          </div>

          {/* Knowledge Graph Evidence Lines (if any) */}
          {currentResult.graph_evidence && currentResult.graph_evidence.length > 0 && (
            <div className="p-4 rounded-xl bg-indigo-50/60 border border-indigo-200 space-y-2">
              <div className="text-xs font-semibold text-indigo-950 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-indigo-600" />
                <span>Knowledge Graph Traversal Evidence ({currentResult.graph_evidence.length} Entity Relationships)</span>
              </div>
              <div className="space-y-1">
                {currentResult.graph_evidence.map((line, idx) => (
                  <div key={idx} className="text-xs text-indigo-900 font-mono">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Retrieved Evidence Passages */}
          {currentResult.retrieved_sources && currentResult.retrieved_sources.length > 0 && (
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-blue-600" />
                <span>Retrieved Evidence ({currentResult.retrieved_sources.length})</span>
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {currentResult.retrieved_sources.map((src, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-2 shadow-xs"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-blue-700 font-semibold truncate max-w-[200px]">
                        📄 {src.doc_name}
                      </span>
                      <span className="text-[10px] text-slate-500 px-2 py-0.5 rounded bg-slate-100 border border-slate-200">
                        Page/Row {src.page}
                      </span>
                    </div>
                    <p className="text-xs text-slate-700 leading-normal line-clamp-4 font-mono bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                      {src.snippet || src.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
