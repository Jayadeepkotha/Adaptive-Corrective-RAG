import React, { useState } from 'react';
import {
  UploadCloud, FileText, Database, Shield, Globe, Zap, RefreshCw, CheckSquare, Square, Check, Trash2
} from 'lucide-react';

export default function VaultSidebar({
  searchMode,
  setSearchMode,
  vaultDocs = [],
  totalChunks = 0,
  graphNodes = 0,
  graphEdges = 0,
  onUploadFiles,
  isUploading,
  uploadProgress,
  onRefreshVault,
  onClearVault,
  selectedDocs = [],
  onToggleDoc,
  onSelectAllDocs,
  onDeselectAllDocs
}) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUploadFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadFiles(Array.from(e.target.files));
    }
    e.target.value = '';
  };

  const getFormatBadge = (name = '') => {
    const ext = name.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return { label: 'PDF', bg: 'bg-rose-50 text-rose-700 border-rose-200' };
    if (ext === 'csv') return { label: 'CSV', bg: 'bg-emerald-50 text-emerald-700 border-emerald-200' };
    if (ext === 'json') return { label: 'JSON', bg: 'bg-amber-50 text-amber-700 border-amber-200' };
    if (['png', 'jpg', 'jpeg'].includes(ext)) return { label: 'IMG', bg: 'bg-violet-50 text-violet-700 border-violet-200' };
    return { label: 'TXT', bg: 'bg-slate-100 text-slate-700 border-slate-200' };
  };

  const allSelected = vaultDocs.length > 0 && selectedDocs.length === vaultDocs.length;

  return (
    <aside className="w-80 border-r border-slate-200 bg-white p-4 flex flex-col gap-4 overflow-y-auto">
      {/* 1. Retrieval Strategy Selection */}
      <div className="space-y-2">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-blue-600" />
          Retrieval Policy
        </label>
        <div className="space-y-1.5">
          {[
            {
              id: 'autonomous',
              label: 'Autonomous Adaptive',
              icon: Zap,
              desc: 'CRAG evaluates local evidence confidence and triggers web fallback only when needed.'
            },
            {
              id: 'strict_vault',
              label: 'Local Documents Only',
              icon: Shield,
              desc: 'Suppresses web-search fallback. (Note: LLM queries still egress to configured provider unless a local model is used.)'
            },
            {
              id: 'web_augmented',
              label: 'Web-Augmented Search',
              icon: Globe,
              desc: 'Retrieves from external web search sources alongside local index.'
            }
          ].map(m => {
            const Icon = m.icon;
            const active = searchMode === m.id;
            return (
              <button
                key={m.id}
                onClick={() => setSearchMode(m.id)}
                className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all ${
                  active
                    ? 'bg-blue-50/70 border-blue-300 text-slate-900 shadow-xs'
                    : 'bg-slate-50/60 border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300'
                }`}
              >
                <div className="flex items-center gap-2 font-medium">
                  <Icon className={`w-3.5 h-3.5 ${active ? 'text-blue-600' : 'text-slate-400'}`} />
                  <span className={active ? 'text-blue-900 font-semibold' : 'text-slate-800'}>{m.label}</span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1 ml-5 leading-snug">{m.desc}</p>
              </button>
            );
          })}
        </div>
      </div>

      <hr className="border-slate-200" />

      {/* 2. Multi-Modal & Multi-File Document Ingestion */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <UploadCloud className="w-3.5 h-3.5 text-blue-600" />
            Document Ingestion
          </label>
          <span className="text-[10px] text-blue-600 font-medium bg-blue-50 border border-blue-200 px-1.5 py-0.5 rounded">
            Multi-Select Active
          </span>
        </div>

        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-xl p-4 text-center transition-all cursor-pointer ${
            dragActive
              ? 'border-blue-500 bg-blue-50/50'
              : 'border-slate-200 hover:border-blue-400 bg-slate-50/60'
          }`}
        >
          <input
            type="file"
            multiple
            onChange={handleFileChange}
            accept=".pdf,.csv,.tsv,.json,.jsonl,.png,.jpg,.jpeg,.txt,.md"
            className="absolute inset-0 opacity-0 cursor-pointer"
            disabled={isUploading}
          />
          <div className="flex flex-col items-center gap-1.5 pointer-events-none">
            <UploadCloud className={`w-6 h-6 ${isUploading ? 'animate-bounce text-blue-600' : 'text-slate-400'}`} />
            <p className="text-xs font-medium text-slate-700">
              {isUploading ? (uploadProgress || 'Extracting & Indexing...') : 'Drop files here or click to browse'}
            </p>
            <p className="text-[10px] text-slate-500">
              Select multiple files (PDF, CSV, JSON, IMG, TXT)
            </p>
          </div>
        </div>
      </div>

      {/* 3. Index & Graph Telemetry Summary */}
      <div className="grid grid-cols-2 gap-2 bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-center">
        <div>
          <div className="text-base font-bold text-slate-900">{totalChunks}</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Index Chunks</div>
        </div>
        <div>
          <div className="text-base font-bold text-blue-600">{graphNodes}</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Graph Nodes</div>
        </div>
      </div>

      {/* 4. Ingested Documents List with Multi-Select Target Filtering */}
      <div className="space-y-2 flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Indexed Vault
            </label>
            {vaultDocs.length > 0 && (
              <span className="text-[10px] font-medium text-slate-500 bg-slate-100 px-1.5 py-0.2 rounded border border-slate-200">
                {selectedDocs.length}/{vaultDocs.length} active
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {vaultDocs.length > 0 && onClearVault && (
              <button
                onClick={onClearVault}
                className="p-1 hover:bg-rose-50 rounded text-slate-400 hover:text-rose-600 transition flex items-center gap-1 text-[10px]"
                title="Clear all stored documents and knowledge graph (Fresh Start)"
              >
                <Trash2 className="w-3 h-3 text-rose-500" />
                <span className="text-rose-600 font-medium">Reset</span>
              </button>
            )}
            <button
              onClick={onRefreshVault}
              className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-700 transition"
              title="Refresh Index Inventory"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Multi-Select Filter Controls */}
        {vaultDocs.length > 1 && (
          <div className="flex items-center justify-between text-[11px] px-1 text-slate-500">
            <span>Query targeting:</span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={allSelected ? onDeselectAllDocs : onSelectAllDocs}
                className="text-blue-600 hover:text-blue-800 font-medium hover:underline"
              >
                {allSelected ? 'Clear All' : 'Select All'}
              </button>
            </div>
          </div>
        )}

        <div className="space-y-1.5 flex-1 overflow-y-auto pr-1">
          {vaultDocs.length === 0 ? (
            <p className="text-xs text-slate-400 py-3 text-center">No documents indexed yet.</p>
          ) : (
            vaultDocs.map((doc, idx) => {
              const badge = getFormatBadge(doc.name);
              const isSelected = selectedDocs.length === 0 || selectedDocs.includes(doc.name);

              return (
                <div
                  key={idx}
                  onClick={() => onToggleDoc && onToggleDoc(doc.name)}
                  className={`flex items-center justify-between p-2 rounded-lg border text-xs cursor-pointer transition select-none ${
                    isSelected
                      ? 'bg-slate-50 border-slate-200 text-slate-800 hover:border-blue-300'
                      : 'bg-white/40 border-slate-100 text-slate-400 opacity-60 hover:opacity-80'
                  }`}
                  title={isSelected ? 'Included in retrieval (click to exclude)' : 'Excluded from retrieval (click to include)'}
                >
                  <div className="flex items-center gap-2 truncate flex-1 min-w-0">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => onToggleDoc && onToggleDoc(doc.name)}
                      onClick={(e) => e.stopPropagation()}
                      className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 w-3.5 h-3.5 cursor-pointer"
                    />
                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${badge.bg}`}>
                      {badge.label}
                    </span>
                    <span className={`truncate font-mono text-[11px] ${isSelected ? 'text-slate-700 font-medium' : 'text-slate-400'}`}>
                      {doc.name}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400 ml-2 whitespace-nowrap">
                    {doc.chunks} ch
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
