import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

const MODELS = ['txgnn_two_phase', 'single_stage', 'joint_contrastive']

function show(val, std) {
  if (val === null || val === undefined) return '—'
  return std === null || std === undefined ? val.toFixed(3) : `${val.toFixed(3)} ± ${std.toFixed(3)}`
}

export default function Alternatives() {
  const { data, loading, error } = useStudyData()
  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const rows = (data.comparisonTable || []).filter((r) => MODELS.includes(r.model))
  const zero = rows.filter((r) => r.split === 'zeroshot')
  const std = rows.filter((r) => r.split === 'standard')
  const baseline = zero.find((r) => r.model === 'txgnn_two_phase')

  const chartZero = zero.map((r) => ({
    name: r.displayName, 'AUPRC ind': r.auprc_ind, 'AUPRC contra': r.auprc_contra,
  }))

  return (
    <div className="page">
      <h1>Q2 — Alternatives to Two-Phase Training</h1>
      <p>
        Baseline: two-phase TxGNN (Phase 1 KG pretrain, Phase 2 task fine-tune). Two alternatives on
        the same backbone (hidden_dim=64, 2 layers, 4 heads).
      </p>

      <div className="finding-box">
        <p><strong>Pre-defined criterion:</strong> higher zero-shot indication AUPRC at equal or
          lower wall-clock time.</p>
        <p><strong>Finding:</strong> two-phase wins zero-shot indication (0.736 vs 0.706 vs 0.670)
          and is the fastest (71.6s vs 130.7s vs 97.1s). No alternative beats it.</p>
      </div>

      <h2>Zero-Shot Split</h2>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartZero} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0.6, 0.9]} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip formatter={(v) => v.toFixed(3)} />
            <Legend />
            <Bar dataKey="AUPRC ind" fill="var(--indication)" opacity={0.85} />
            <Bar dataKey="AUPRC contra" fill="var(--contra)" opacity={0.85} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <SourceLine source="results/metrics/comparison_table.csv" extra="zero-shot split, seeds [42, 0, 1]" />

      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Method</th><th>AUPRC ind</th><th>AUPRC contra</th><th>AUROC ind</th><th>Wall-clock (s)</th></tr>
          </thead>
          <tbody>
            {zero.map((r, i) => (
              <tr key={i} style={r.model === 'txgnn_two_phase' ? { fontWeight: 600 } : {}}>
                <td>
                  {r.displayName}
                  {r.model === 'txgnn_two_phase' && (
                    <span className="positive-badge" style={{ marginLeft: '0.5rem' }}>best</span>
                  )}
                </td>
                <td>{show(r.auprc_ind, r.auprc_ind_std)}</td>
                <td>{show(r.auprc_contra, r.auprc_contra_std)}</td>
                <td>{show(r.auroc_ind, r.auroc_ind_std)}</td>
                <td>{r.wall_s !== null ? r.wall_s : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/metrics/comparison_table.csv" />

      <h2>Standard Split</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Method</th><th>AUPRC ind</th><th>AUPRC contra</th><th>Wall-clock (s)</th></tr>
          </thead>
          <tbody>
            {std.map((r, i) => (
              <tr key={i}>
                <td>{r.displayName}</td>
                <td>{show(r.auprc_ind, r.auprc_ind_std)}</td>
                <td>{show(r.auprc_contra, r.auprc_contra_std)}</td>
                <td>{r.wall_s !== null ? r.wall_s : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/metrics/comparison_table.csv" />

      <h2>Interpretation</h2>
      <p>
        Phase 1 (KG link prediction) positions the embeddings so the later therapeutic fine-tune
        generalizes to unseen diseases. Collapsing the phases loses that.
      </p>
      <p>
        The contrastive variant edges ahead on zero-shot contraindication (0.854) but trails on
        indication (0.670 ± 0.038) with more variance — so it loses under the chosen criterion.
      </p>

      <h2>Trade-off Table</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th><th>Zero-shot ind AUPRC</th><th>Wall-clock (s)</th>
              <th>Beats two-phase (ind)?</th><th>Beats two-phase (speed)?</th>
            </tr>
          </thead>
          <tbody>
            {zero.map((r, i) => {
              const betterInd = baseline ? r.auprc_ind > baseline.auprc_ind : false
              const betterSpeed = baseline ? r.wall_s < baseline.wall_s : false
              const isBase = r.model === 'txgnn_two_phase'
              const cell = (ok) => ({
                color: ok ? 'var(--indication)' : 'var(--contra)',
                fontWeight: 600, fontFamily: 'var(--font-mono)',
              })
              return (
                <tr key={i}>
                  <td>{r.displayName}</td>
                  <td>{r.auprc_ind?.toFixed(3)}</td>
                  <td>{r.wall_s}</td>
                  <td style={cell(betterInd)}>{isBase ? '(baseline)' : betterInd ? 'Yes' : 'No'}</td>
                  <td style={cell(betterSpeed)}>{isBase ? '(baseline)' : betterSpeed ? 'Yes' : 'No'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/metrics/comparison_table.csv" extra="zero-shot split" />
    </div>
  )
}
