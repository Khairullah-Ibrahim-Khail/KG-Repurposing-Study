import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

function show(val, std) {
  if (val === null || val === undefined) return '—'
  const base = val.toFixed(3)
  return std === null || std === undefined ? base : `${base} ± ${std.toFixed(3)}`
}

function Mono({ children }) {
  return (
    <span style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>
      {children}
    </span>
  )
}

function Legend2() {
  return (
    <div className="table-legend">
      <div className="table-legend-item"><div className="legend-swatch indication" /><span>Indication</span></div>
      <div className="table-legend-item"><div className="legend-swatch contra" /><span>Contraindication</span></div>
    </div>
  )
}

// Paper-reported cell: a real value, an explicit "no data" pointer for zero-shot
// TxGNN, or an em dash. Never a bare "0".
function PaperValue({ pr, model, split, field, suppl }) {
  const value = pr?.[field]
  if (value) {
    return (<span><Mono>{value}</Mono> <span className="badge badge-paper">paper</span></span>)
  }
  if (split === 'zeroshot' && (model === 'txgnn_two_phase' || model === 'txgnn_attn_on')) {
    return (
      <span className="paper-missing" title={`Absolute zero-shot values in Huang et al. 2024 ${suppl}.`}>
        — <span style={{ fontSize: '0.7rem' }}>[no data — see {suppl}]</span>
      </span>
    )
  }
  return <span style={{ color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>—</span>
}

function Q1Chart({ rows }) {
  const data = ['standard', 'zeroshot'].map((split) => {
    const row = (m) => rows.find((r) => r.model === m && r.split === split) || {}
    return {
      split: split === 'zeroshot' ? 'Zero-Shot' : 'Standard',
      gnn_no_kg_ind: row('gnn_no_kg').auprc_ind,
      gnn_kg_ind: row('gnn_kg').auprc_ind,
      gnn_no_kg_contra: row('gnn_no_kg').auprc_contra,
      gnn_kg_contra: row('gnn_kg').auprc_contra,
    }
  })
  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
          <XAxis dataKey="split" tick={{ fontFamily: 'var(--font-mono)', fontSize: 12 }} />
          <YAxis domain={[0.6, 1]} tickFormatter={(v) => v.toFixed(2)}
                 tick={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
          <Tooltip formatter={(v) => v.toFixed(3)} />
          <Legend wrapperStyle={{ fontFamily: 'var(--font-body)', fontSize: '0.8rem' }} />
          <Bar dataKey="gnn_no_kg_ind" name="GNN no-KG (ind)" fill="var(--indication)" opacity={0.85} />
          <Bar dataKey="gnn_kg_ind" name="GNN KG (ind)" fill="var(--accent)" opacity={0.85} />
          <Bar dataKey="gnn_no_kg_contra" name="GNN no-KG (contra)" fill="var(--indication)" opacity={0.45} />
          <Bar dataKey="gnn_kg_contra" name="GNN KG (contra)" fill="var(--accent)" opacity={0.45} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ResultsView() {
  const { data, loading, error } = useStudyData()
  const [filter, setFilter] = useState('all')

  if (loading) return <div className="page status-loading">Loading…</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const table = data.comparisonTable || []
  const q1Rows = table.filter((r) => ['gnn_no_kg', 'gnn_kg'].includes(r.model))
  const shown = filter === 'all' ? table : table.filter((r) => r.split === filter)
  const note = data.paperReported?.zeroshot?._relativeNote || ''

  return (
    <div className="page">
      <h1>Q1 + Q5: Results</h1>

      <section className="section" aria-labelledby="q1">
        <h2 id="q1">Q1 — Does KG Augmentation Beat No-KG?</h2>
        <p>
          One HGT backbone, two conditions: 2-layer message passing vs. 0-layer (none). Same Phase 1
          pretrain for both. Primary metric: indication AUPRC.
        </p>
        <div className="finding-box">
          <p>
            <strong>Finding:</strong> message passing does not raise indication AUPRC on either
            split. Standard split: no-KG wins (<Mono>0.893</Mono> vs <Mono>0.825</Mono>). Zero-shot
            indication is within noise (<Mono>0.704</Mono> vs <Mono>0.714</Mono>); KG only helps
            zero-shot contraindication (<Mono>0.831</Mono> vs <Mono>0.725</Mono>). Reported as-is.
          </p>
        </div>

        <Q1Chart rows={q1Rows} />
        <SourceLine source="results/gnn/{standard,zeroshot}/seed_{42,0,1}/gnn_baseline.json, gnn_no_kg.json"
                    extra="bars are mean AUPRC across seeds [42, 0, 1]" />

        <h3>Detailed Q1 Table</h3>
        <Legend2 />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Split</th>
                <th className="col-indication">AUPRC ind (mean±std)</th>
                <th className="col-contra">AUPRC contra (mean±std)</th>
                <th className="col-indication">AUROC ind</th>
                <th className="col-contra">AUROC contra</th>
                <th>Wall-clock (s)</th>
              </tr>
            </thead>
            <tbody>
              {q1Rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.displayName}</td>
                  <td className={r.split === 'zeroshot' ? 'tag-zeroshot' : 'tag-standard'}>
                    {r.split === 'zeroshot' ? 'zero-shot' : 'standard'}
                  </td>
                  <td><Mono>{show(r.auprc_ind, r.auprc_ind_std)}</Mono></td>
                  <td><Mono>{show(r.auprc_contra, r.auprc_contra_std)}</Mono></td>
                  <td><Mono>{show(r.auroc_ind, r.auroc_ind_std)}</Mono></td>
                  <td><Mono>{show(r.auroc_contra, r.auroc_contra_std)}</Mono></td>
                  <td><Mono>{r.wall_s !== null ? r.wall_s : '—'}</Mono></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <SourceLine source="results/metrics/comparison_table.csv" />
      </section>

      <section className="section" aria-labelledby="q5">
        <h2 id="q5">Q5 — Full Comparison: GNN vs TxGNN</h2>
        <p>
          Auto-generated from <code>results/metrics/comparison_table.csv</code>. All models, both
          splits, seeds [42, 0, 1], random-negative AUPRC. The{' '}
          <span className="badge badge-paper">paper_reported</span> column (Huang et al. 2024, Suppl.
          S1–S2) is never merged with ours.
        </p>
        {note && (
          <div className="warn-box"><p><strong>Zero-shot paper_reported note:</strong> {note}</p></div>
        )}

        <div className="seg" role="group" aria-label="Filter by split">
          <span className="seg-label">Filter split</span>
          {['all', 'standard', 'zeroshot'].map((s) => (
            <button key={s} className="seg-item" aria-pressed={filter === s} onClick={() => setFilter(s)}>
              {s === 'all' ? 'All' : s === 'zeroshot' ? 'Zero-Shot' : 'Standard'}
            </button>
          ))}
        </div>

        {['standard', 'zeroshot']
          .filter((s) => filter === 'all' || filter === s)
          .map((split) => (
            <div key={split}>
              <h3>{split === 'zeroshot' ? 'Zero-Shot Split' : 'Standard Split'}</h3>
              <Legend2 />
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th><th>Type</th>
                      <th className="col-indication">AUPRC ind (scaled)</th>
                      <th className="col-indication">AUPRC ind (paper)</th>
                      <th className="col-contra">AUPRC contra (scaled)</th>
                      <th className="col-contra">AUPRC contra (paper)</th>
                      <th>Wall-clock (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shown.filter((r) => r.split === split).map((r, i) => (
                      <tr key={i}>
                        <td>{r.displayName}</td>
                        <td>
                          <span className={
                            r.reproduction_type === 'scaled_reproduction' ? 'badge badge-scaled'
                              : r.reproduction_type === 'original_ablation' ? 'badge badge-original'
                                : 'badge'
                          }>{r.reproduction_type}</span>
                        </td>
                        <td><Mono>{show(r.auprc_ind, r.auprc_ind_std)}</Mono></td>
                        <td><PaperValue pr={r.paper_reported} model={r.model} split={split}
                                        field="auprc_ind" suppl="Suppl. S1" /></td>
                        <td><Mono>{show(r.auprc_contra, r.auprc_contra_std)}</Mono></td>
                        <td><PaperValue pr={r.paper_reported} model={r.model} split={split}
                                        field="auprc_contra" suppl="Suppl. S2" /></td>
                        <td><Mono>{r.wall_s !== null ? r.wall_s : '—'}</Mono></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}

        <SourceLine source="results/metrics/comparison_table.csv"
                    extra="paper_reported: Huang et al. 2024, Suppl. Tables S1–S2" />
        <p style={{ fontSize: '0.85rem', color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>
          Note: <code>txgnn_two_phase</code> and <code>txgnn_attn_on</code> are the same model; both
          CSV rows are shown. Rows marked “[NOT YET RUN]” are filtered out.
        </p>
      </section>
    </div>
  )
}
