import React from 'react';
import { Layers, Activity, Cpu } from 'lucide-react';

export default function Header({ serverStatus = 'online', totalChunks = 0 }) {
  return (
    <header className="h-16 border-b border-slate-200 bg-white/95 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center shadow-xs text-white">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-slate-900">Confidence-RAG</h1>
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
              Adaptive Research System
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Calibrated Evidence Retrieval, Knowledge Graph Reasoning & Verification Pipeline
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 font-mono">
          <Cpu className="w-3.5 h-3.5 text-blue-600" />
          <span>Configured LLM Provider</span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-700 font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>API :8000 {serverStatus}</span>
        </div>
      </div>
    </header>
  );
}
