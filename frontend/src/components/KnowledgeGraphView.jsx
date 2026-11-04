import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { ZoomIn, ZoomOut, Maximize2, Network, Info } from 'lucide-react';

export default function KnowledgeGraphView({ graphData }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    // Convert nodes and triples to Cytoscape elements
    const elements = [];
    const nodeSet = new Set();

    (graphData.nodes || []).forEach(node => {
      nodeSet.add(node);
      elements.push({
        data: { id: node, label: node }
      });
    });

    (graphData.triples || []).forEach((t, idx) => {
      if (!nodeSet.has(t.subject)) {
        nodeSet.add(t.subject);
        elements.push({ data: { id: t.subject, label: t.subject } });
      }
      if (!nodeSet.has(t.object)) {
        nodeSet.add(t.object);
        elements.push({ data: { id: t.object, label: t.object } });
      }
      elements.push({
        data: {
          id: `edge_${idx}`,
          source: t.subject,
          target: t.object,
          label: t.relation
        }
      });
    });

    // Initialize Cytoscape with clean, restrained light theme
    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#2563eb',
            'label': 'data(label)',
            'color': '#0f172a',
            'font-size': '10px',
            'font-weight': '600',
            'text-valign': 'bottom',
            'text-margin-y': 4,
            'text-halign': 'center',
            'width': '32px',
            'height': '32px',
            'border-width': '2px',
            'border-color': '#1d4ed8',
            'text-background-color': '#ffffff',
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 1.5,
            'line-color': '#94a3b8',
            'target-arrow-color': '#64748b',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '8px',
            'color': '#475569',
            'text-rotation': 'autorotate',
            'text-background-color': '#f8fafc',
            'text-background-opacity': 0.9,
            'text-background-padding': '2px'
          }
        },
        {
          selector: ':selected',
          style: {
            'background-color': '#0284c7',
            'border-color': '#0369a1',
            'border-width': '3px',
            'line-color': '#0284c7',
            'target-arrow-color': '#0284c7'
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: true,
        randomize: false,
        componentSpacing: 100,
        nodeOverlap: 20,
        idealEdgeLength: 80,
        edgeElasticity: 100
      }
    });

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode({
        id: node.id(),
        degree: node.degree(),
        connectedEdges: node.connectedEdges().map(e => ({
          source: e.data('source'),
          target: e.data('target'),
          label: e.data('label')
        }))
      });
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [graphData]);

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current?.fit();

  return (
    <div className="flex-1 flex flex-col p-6 max-w-6xl mx-auto w-full gap-4 h-full">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Network className="w-5 h-5 text-blue-600" />
            Knowledge Graph Topology
          </h2>
          <p className="text-xs text-slate-500">
            Entity-relationship graph extracted automatically from ingested multi-modal documents.
          </p>
        </div>

        <div className="flex items-center gap-1.5 bg-white border border-slate-200 rounded-xl p-1 shadow-xs">
          <button onClick={handleZoomIn} className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-600" title="Zoom in">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={handleZoomOut} className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-600" title="Zoom out">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={handleFit} className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-600" title="Fit to screen">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-[500px]">
        {/* Cytoscape Graph Canvas */}
        <div
          ref={containerRef}
          className="flex-1 rounded-2xl bg-slate-50 border border-slate-200 relative overflow-hidden shadow-xs"
        />

        {/* Selected Entity Inspector Drawer */}
        {selectedNode ? (
          <div className="w-72 rounded-2xl bg-white border border-slate-200 p-4 space-y-3 flex flex-col shadow-xs">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Entity Details
              </span>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-xs text-slate-400 hover:text-slate-700"
              >
                ✕
              </button>
            </div>
            <div>
              <div className="text-sm font-bold text-blue-700 break-words">{selectedNode.id}</div>
              <div className="text-xs text-slate-500 mt-1">Connections: {selectedNode.degree} edges</div>
            </div>

            <hr className="border-slate-200" />

            <div className="flex-1 overflow-y-auto space-y-2">
              <div className="text-[11px] font-semibold text-slate-500 uppercase">Semantic Triples</div>
              {selectedNode.connectedEdges?.map((e, idx) => (
                <div key={idx} className="p-2 rounded-lg bg-slate-50 text-xs border border-slate-200">
                  <span className="text-blue-700 font-semibold">{e.source}</span>
                  <span className="text-slate-400 mx-1.5 font-mono text-[10px]">-[{e.label}]-&gt;</span>
                  <span className="text-emerald-700 font-semibold">{e.target}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="w-72 rounded-2xl bg-white border border-slate-200 p-4 flex flex-col items-center justify-center text-center text-slate-400 shadow-xs">
            <Info className="w-6 h-6 mb-2 opacity-60 text-slate-400" />
            <p className="text-xs">Select any node on the graph canvas to inspect its entity connections.</p>
          </div>
        )}
      </div>
    </div>
  );
}
