import React from 'react'

// A small left-to-right KG path: each entity is a pill (colored dot + name) with
// its type as a caption, joined by labelled arrow connectors.
//   nodes: [{ id, label, type: 'disease'|'drug'|'gene'|'protein'|'other' }]
//   edges: [{ source, target, label }]

const TYPE = {
  disease: { color: 'var(--contra)', bg: 'var(--contra-muted)' },
  drug: { color: 'var(--indication)', bg: 'var(--indication-muted)' },
  gene: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  protein: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  other: { color: 'var(--structure)', bg: 'rgba(91,107,130,0.10)' },
}

export default function GraphPath({ nodes = [], edges = [] }) {
  if (nodes.length === 0) return null
  const edgeBetween = (a, b) =>
    edges.find((e) => e.source === a && e.target === b) ||
    edges.find((e) => e.source === b && e.target === a)

  return (
    <div className="kg-path" role="img" aria-label={`KG path: ${nodes.map((n) => n.label).join(' → ')}`}>
      {nodes.map((node, i) => {
        const t = TYPE[node.type] || TYPE.other
        const next = nodes[i + 1]
        const edge = next ? edgeBetween(node.id, next.id) : null
        return (
          <React.Fragment key={node.id || i}>
            <div className="kg-node">
              <div className="kg-chip" style={{ color: t.color, borderColor: t.color, background: t.bg }}>
                <span className="kg-dot" style={{ background: t.color }} />
                {node.label}
              </div>
              <span className="kg-node-type">{node.type}</span>
            </div>
            {next && (
              <div className="kg-edge">
                {edge?.label && <span className="kg-edge-label" title={edge.label}>{edge.label}</span>}
                <span className="kg-edge-line" />
              </div>
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}
