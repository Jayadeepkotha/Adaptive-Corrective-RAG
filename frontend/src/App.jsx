import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VaultSidebar from './components/VaultSidebar';
import IntelligenceStudio from './components/IntelligenceStudio';
import KnowledgeGraphView from './components/KnowledgeGraphView';
import ObservatoryView from './components/ObservatoryView';
import { MessageSquare, Network, BarChart3 } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [searchMode, setSearchMode] = useState('autonomous');
  const [vaultDocs, setVaultDocs] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [graphData, setGraphData] = useState({ nodes: [], edges_count: 0, triples: [] });
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [serverStatus, setServerStatus] = useState('connecting');

  // Fetch initial vault and graph inventory
  const fetchVault = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/vault`);
      if (res.ok) {
        const data = await res.json();
        const docs = data.documents || [];
        setVaultDocs(docs);
        setTotalChunks(data.total_chunks || 0);
        setSelectedDocs(prev => {
          const docNames = docs.map(d => d.name);
          if (prev.length === 0) return docNames;
          const valid = prev.filter(n => docNames.includes(n));
          // if new docs were added, add them to selection
          const newlyAdded = docNames.filter(n => !prev.includes(n));
          return [...valid, ...newlyAdded];
        });
        setServerStatus('online');
      }
    } catch (err) {
      console.warn('API unreachable:', err);
      setServerStatus('offline');
    }
  };

  const fetchGraph = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/graph`);
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
      }
    } catch (err) {
      console.warn('Graph fetch failed:', err);
    }
  };

  useEffect(() => {
    fetchVault();
    fetchGraph();
  }, []);

  // Multi-file upload handler
  const handleUploadFiles = async (files) => {
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setUploadProgress(`Uploading ${files.length} document(s)...`);

    try {
      // Try batch upload endpoint first
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file);
      }

      let res = await fetch(`${API_BASE}/api/v1/upload_batch`, {
        method: 'POST',
        body: formData
      });

      // Fallback to sequential uploads if batch endpoint returns error
      if (!res.ok) {
        for (let i = 0; i < files.length; i++) {
          setUploadProgress(`Indexing ${i + 1} of ${files.length}: ${files[i].name}...`);
          const singleForm = new FormData();
          singleForm.append('file', files[i]);
          await fetch(`${API_BASE}/api/v1/upload`, {
            method: 'POST',
            body: singleForm
          });
        }
      }

      await fetchVault();
      await fetchGraph();
    } catch (err) {
      alert('Error uploading files: ' + err.message);
    } finally {
      setIsUploading(false);
      setUploadProgress('');
    }
  };

  const handleToggleDoc = (docName) => {
    setSelectedDocs(prev =>
      prev.includes(docName)
        ? prev.filter(d => d !== docName)
        : [...prev, docName]
    );
  };

  const handleSelectAllDocs = () => {
    setSelectedDocs(vaultDocs.map(d => d.name));
  };

  const handleDeselectAllDocs = () => {
    setSelectedDocs([]);
  };

  const handleClearVault = async () => {
    if (!window.confirm("Are you sure you want to delete all persistent documents, vectors, and knowledge graph data for a fresh start?")) {
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/v1/vault`, { method: 'DELETE' });
      if (res.ok) {
        await fetchVault();
        await fetchGraph();
        setCurrentResult(null);
      }
    } catch (err) {
      alert("Error clearing vault: " + err.message);
    }
  };

  const handleRunCompare = async (queryText) => {
    if (!queryText.trim()) return;
    setIsProcessing(true);
    setComparisonResult(null);
    setCurrentResult(null);

    try {
      const docFilter = (selectedDocs.length > 0 && selectedDocs.length < vaultDocs.length)
        ? selectedDocs
        : (selectedDocs.length === 0 ? [] : null);

      const res = await fetch(`${API_BASE}/api/v1/query/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          search_mode: searchMode,
          selected_docs: docFilter
        })
      });

      if (res.ok) {
        const data = await res.json();
        setComparisonResult(data);
      } else {
        alert('Comparison evaluation failed: ' + res.statusText);
      }
    } catch (err) {
      alert('Error running comparison: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunQuery = async (queryText) => {
    if (!queryText.trim()) return;
    setIsProcessing(true);

    // Initial placeholder result
    setCurrentResult({
      query: queryText,
      complexity: 'CLASSIFYING...',
      route_name: 'Analyzing retrieval complexity...',
      answer: '',
      retrieved_sources: [],
      web_sources: [],
      graph_evidence: [],
      latency_ms: 0
    });

    try {
      // If user selected a subset of docs, restrict retrieval to those docs
      const docFilter = (selectedDocs.length > 0 && selectedDocs.length < vaultDocs.length)
        ? selectedDocs
        : (selectedDocs.length === 0 ? [] : null);

      const res = await fetch(`${API_BASE}/api/v1/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          search_mode: searchMode,
          selected_docs: docFilter
        })
      });

      if (!res.ok || !res.body) {
        // Fallback to non-streaming endpoint if streaming fails
        const fallbackRes = await fetch(`${API_BASE}/api/v1/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: queryText,
            search_mode: searchMode,
            selected_docs: docFilter
          })
        });
        if (fallbackRes.ok) {
          const data = await fallbackRes.json();
          setCurrentResult(data);
        }
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const ev of events) {
          const trimmed = ev.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const jsonStr = trimmed.slice(6);
          try {
            const parsed = JSON.parse(jsonStr);
            if (parsed.type === 'metadata' && parsed.data) {
              setCurrentResult(prev => ({
                ...prev,
                ...parsed.data,
                answer: prev?.answer || ''
              }));
            } else if (parsed.type === 'token') {
              setCurrentResult(prev => ({
                ...prev,
                answer: (prev?.answer || '') + parsed.content
              }));
            } else if (parsed.type === 'done') {
              if (parsed.latency_ms) {
                setCurrentResult(prev => ({
                  ...prev,
                  latency_ms: parsed.latency_ms
                }));
              }
            }
          } catch (pe) {
            console.warn('SSE chunk parse error:', pe);
          }
        }
      }
    } catch (err) {
      alert('Query execution error: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-50 text-slate-800 overflow-hidden font-sans">
      {/* Top Header */}
      <Header serverStatus={serverStatus} totalChunks={totalChunks} />

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <VaultSidebar
          searchMode={searchMode}
          setSearchMode={setSearchMode}
          vaultDocs={vaultDocs}
          selectedDocs={selectedDocs}
          onToggleDoc={handleToggleDoc}
          onSelectAllDocs={handleSelectAllDocs}
          onDeselectAllDocs={handleDeselectAllDocs}
          totalChunks={totalChunks}
          graphNodes={graphData.nodes?.length || 0}
          graphEdges={graphData.edges_count || 0}
          onUploadFiles={handleUploadFiles}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
          onRefreshVault={() => { fetchVault(); fetchGraph(); }}
          onClearVault={handleClearVault}
        />

        {/* Center Content Workspace */}
        <main className="flex-1 flex flex-col bg-slate-50 overflow-hidden">
          {/* Navigation Tab Bar */}
          <div className="h-12 border-b border-slate-200 bg-white px-6 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              {[
                { id: 'studio', label: 'Query & Synthesis', icon: MessageSquare },
                { id: 'graph', label: 'Knowledge Graph', icon: Network },
                { id: 'observatory', label: 'Evaluation & Metrics', icon: BarChart3 }
              ].map(tab => {
                const Icon = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
                      active
                        ? 'bg-blue-50 text-blue-700 border border-blue-200/80 font-semibold shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tab Content Display */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === 'studio' && (
              <IntelligenceStudio
                onRunQuery={handleRunQuery}
                onRunCompare={handleRunCompare}
                comparisonResult={comparisonResult}
                onClearComparison={() => setComparisonResult(null)}
                isProcessing={isProcessing}
                currentResult={currentResult}
              />
            )}
            {activeTab === 'graph' && (
              <KnowledgeGraphView graphData={graphData} />
            )}
            {activeTab === 'observatory' && (
              <ObservatoryView />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
