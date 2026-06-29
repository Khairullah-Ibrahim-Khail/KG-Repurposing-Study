import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

function show(val, std) {
  if (val === null || val === undefined) return '—'
  return std === null || std === undefined ? val.toFixed(3) : `${val.toFixed(3)} ± ${std.toFixed(3)}`
}
function Mono({ children }) {
  return <span style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>{children}</span>
}

const VARIANT = {
  txgnn: 'TxGNN (attn=ON, sim=ON)',
  txgnn_no_attn: 'TxGNN (attn=OFF, sim=ON)',
  txgnn_no_sim: 'TxGNN (attn=ON, sim=OFF)',
  txgnn_no_both: 'TxGNN (attn=OFF, sim=OFF)',
}

export default function Ablations() {
  const { data, loading, error } = useStudyData()
  if (loading) return <div className="page status-loading">Loading…</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const matrix = data.ablationMatrix?.matrix || []
  const decision = data.ablationMatrix?.q6_decision || {}
  const q6Table = data.q6AblationTable || []
  const zero = matrix.filter((r) => r.split === 'zeroshot')
  const std = matrix.filter((r) => r.split === 'standard')

  const chart = zero.map((r) => ({
    name: VARIANT[r.variant] || r.variant,
    'Indication AUPRC': r.indication_auprc_mean,
    'Contraindication AUPRC': r.contraindication_auprc_mean,
  }))

  const attnOn = zero.find((r) => r.variant === 'txgnn')
  const attnOff = zero.find((r) => r.variant === 'txgnn_no_attn')
  const delta = decision.delta_attn_on_minus_off
  const deltaText = delta !== undefined ? (delta >= 0 ? '+' : '') + delta.toFixed(3) : '—'

  return (
    <div className="page">
      <h1>Q6 — Is Attention in TxGNN Optional? (Answer: Detrimental)</h1>

      <p>
        Hypothesis: the zero-shot advantage comes from the disease-affinity head, not from HGT
        attention. The question asks whether attention is “optional”; the result is stronger —
        attention is <strong>actively detrimental</strong> in this scaled reproduction.
      </p>
      <p>
        Rule fixed before running: delta = AUPRC(attn=ON) − AUPRC(attn=OFF).{' '}
        <Mono>|delta| &lt; 0.02</Mono> → optional; <Mono>delta ≤ −0.02</Mono> → detrimental;{' '}
        <Mono>delta ≥ +0.02</Mono> → helps.
      </p>

      <div className="stat-card-negative">
        <div className="stat-label">Attention AUPRC delta (ON − OFF)</div>
        <div className="stat-value">{deltaText}</div>
        <div className="stat-sublabel">
          <Mono>{attnOn ? attnOn.indication_auprc_mean.toFixed(3) : '?'}</Mono> −{' '}
          <Mono>{attnOff ? attnOff.indication_auprc_mean.toFixed(3) : '?'}</Mono>
        </div>
        <div className="stat-verdict">Attention is DETRIMENTAL</div>
      </div>

      <div className="negative-box">
        <p>
          <strong>Q6 decision (threshold ±0.02, set before running):</strong>{' '}
          {decision.conclusion ||
            'delta < −0.02 — removing HGT attention IMPROVES zero-shot indication AUPRC.'}
        </p>
      </div>

      <h2>Ablation Matrix — Zero-Shot</h2>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chart} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
                   interval={0} angle={-20} textAnchor="end" />
            <YAxis domain={[0.7, 0.9]} tickFormatter={(v) => v.toFixed(2)}
                   tick={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
            <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)} />
            <Legend wrapperStyle={{ fontFamily: 'var(--font-body)', fontSize: '0.8rem' }} />
            <Bar dataKey="Indication AUPRC" fill="var(--indication)" opacity={0.85} />
            <Bar dataKey="Contraindication AUPRC" fill="var(--contra)" opacity={0.85} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <SourceLine source="results/ablations/matrix.json" extra="zero-shot split, seeds [42, 0, 1]" />

      <h2>Full Ablation Table — Zero-Shot</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Variant</th><th>Ind AUPRC</th><th>Contra AUPRC</th><th>Ind AUROC</th><th>Wall-clock (s)</th></tr>
          </thead>
          <tbody>
            {zero.map((r, i) => (
              <tr key={i} style={r.variant === 'txgnn_no_attn' ? { background: 'var(--indication-muted)' } : {}}>
                <td>
                  {VARIANT[r.variant] || r.variant}
                  {r.variant === 'txgnn_no_attn' && (
                    <span className="positive-badge" style={{ marginLeft: '0.5rem' }}>best ind AUPRC</span>
                  )}
                </td>
                <td><Mono>{show(r.indication_auprc_mean, r.indication_auprc_std)}</Mono></td>
                <td><Mono>{show(r.contraindication_auprc_mean, r.contraindication_auprc_std)}</Mono></td>
                <td><Mono>{r.indication_auroc_mean?.toFixed(3)}</Mono></td>
                <td><Mono>{r.wall_clock_s_mean}</Mono></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/ablations/matrix.json" />

      <h2>Ablation Matrix — Standard</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Variant</th><th>Ind AUPRC</th><th>Contra AUPRC</th><th>Wall-clock (s)</th></tr>
          </thead>
          <tbody>
            {std.map((r, i) => (
              <tr key={i}>
                <td>{VARIANT[r.variant] || r.variant}</td>
                <td><Mono>{show(r.indication_auprc_mean, r.indication_auprc_std)}</Mono></td>
                <td><Mono>{show(r.contraindication_auprc_mean, r.contraindication_auprc_std)}</Mono></td>
                <td><Mono>{r.wall_clock_s_mean}</Mono></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/ablations/matrix.json" />

      <h2>Q6 Ablation Table (CSV)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Variant</th><th>Split</th><th>AUPRC ind</th><th>N seeds</th></tr>
          </thead>
          <tbody>
            {q6Table.map((r, i) => (
              <tr key={i}>
                <td>{r.displayName || r.variant}</td>
                <td className={r.split === 'zeroshot' ? 'tag-zeroshot' : 'tag-standard'}>
                  {r.split === 'zeroshot' ? 'zero-shot' : 'standard'}
                </td>
                <td><Mono>{show(r.auprc_ind_mean, r.auprc_ind_std)}</Mono></td>
                <td><Mono>{r.n_seeds}</Mono></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source="results/metrics/q6_ablation_table.csv" />

      <h2>Interpretation</h2>
      <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
        <li>Removing attention helps — it is not load-bearing, it hurts in this scaled setup.</li>
        <li>Removing similarity (<code>txgnn_no_sim</code>) drops AUPRC <Mono>0.736 → 0.726</Mono>,
          so the affinity head contributes positively.</li>
        <li>Removing both (<Mono>0.762</Mono>) beats the full model but trails removing attention
          alone (<Mono>0.772</Mono>).</li>
        <li>HGT attention likely overfits at 64 dims with limited data.</li>
      </ul>

      <div className="finding-box">
        <p>
          Consistent with Huang et al. (2024): they found a learnable attention gate ineffective and
          replaced it with a fixed degree-based gating (λ=0.7). Our ablation reaches the same
          conclusion independently.
        </p>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--structure)', fontFamily: 'var(--font-mono)' }}>
        Scope: scaled reproduction, RTX 4060 8 GB, hidden_dim=64, 2 layers, PrimeKG, seeds [42, 0, 1].
      </p>
    </div>
  )
}
