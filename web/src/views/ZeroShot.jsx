import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

const BIN_ORDER = ['0', '1-5', '6-20', '21+']
const BIN_LABEL = { '0': '0 edges', '1-5': '1-5 edges', '6-20': '6-20 edges', '21+': '21+ edges' }
const COLORS = ['var(--accent)', 'var(--indication)', 'var(--structure)', 'var(--contra)']

function buildSeries(binned, models, relation) {
  return BIN_ORDER.map((bin) => {
    const row = { bin: BIN_LABEL[bin] || bin }
    models.forEach((m) => {
      const hit = binned.find((r) => r.model === m && r.bin === bin && r.relation === relation)
      if (hit) row[m] = hit.mean_auprc
    })
    return row
  }).filter((row) => models.some((m) => row[m] !== undefined))
}

function CurveChart({ data, models }) {
  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="bin" />
          <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)}
                 label={{ value: 'Mean AUPRC', angle: -90, position: 'insideLeft', dx: -10 }} />
          <Tooltip formatter={(v) => (v !== undefined ? v.toFixed(4) : '—')} />
          <Legend />
          {models.map((m, i) => <Bar key={m} dataKey={m} name={m} fill={COLORS[i % COLORS.length]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ZeroShot() {
  const { data, loading, error } = useStudyData()
  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const binned = data.degradationCurveData?.binned || []
  const models = [...new Set(binned.map((r) => r.model))]
  const indSeries = buildSeries(binned, models, 'indication')
  const contraSeries = buildSeries(binned, models, 'contraindication')

  return (
    <div className="page">
      <h1>Q3 — Why Is Zero-Shot Prediction Preferred?</h1>

      <section className="section" aria-labelledby="concept">
        <h2 id="concept">Conceptual Answer</h2>
        <p>
          Huang et al. (2024) report that 92% of PrimeKG diseases have no indication edge. On our
          download, 9,388 unique indication pairs span 17,080 diseases — consistent with ~8%
          coverage. A supervised classifier needs labeled pairs per disease; for the other ~92% there
          are none.
        </p>
        <p>
          Disease-similarity zero-shot transfer is the only tractable route for those diseases. A
          standard-split model can make no prediction for a disease with zero training edges.
        </p>
        <div className="finding-box">
          <p>
            <strong>92% of PrimeKG diseases have no indication edge</strong> (Huang et al. 2024,
            main text). Zero-shot is the only approach that covers them.
          </p>
        </div>
      </section>

      <section className="section" aria-labelledby="empirical">
        <h2 id="empirical">Empirical: Degradation Curve</h2>
        <p>
          AUPRC against the number of training treatment edges per disease, binned into four groups
          (mean AUPRC per bin).
        </p>

        <h3>Indication AUPRC by Training-Edge Count</h3>
        {indSeries.length ? (
          <>
            <CurveChart data={indSeries} models={models} />
            <SourceLine source="results/metrics/degradation_curve_data.json"
                        extra="binned by n_train_edges: [0], [1-5], [6-20], [21+]" />
          </>
        ) : (
          <p style={{ color: 'var(--structure)' }}>
            No indication data ({data.degradationCurveData?.raw?.length ?? 0} raw rows).
          </p>
        )}

        <h3>Contraindication AUPRC by Training-Edge Count</h3>
        {contraSeries.length ? (
          <>
            <CurveChart data={contraSeries} models={models} />
            <SourceLine source="results/metrics/degradation_curve_data.json" extra="contraindication, binned" />
          </>
        ) : (
          <p style={{ color: 'var(--structure)' }}>No contraindication data available.</p>
        )}

        <h3>Binned Summary Table</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Relation</th><th>Bin</th><th>Mean AUPRC</th><th>N diseases</th></tr>
            </thead>
            <tbody>
              {binned.map((row, i) => (
                <tr key={i}>
                  <td>{row.model}</td>
                  <td>{row.relation}</td>
                  <td>{BIN_LABEL[row.bin] || row.bin}</td>
                  <td>{row.mean_auprc.toFixed(4)}</td>
                  <td>{row.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <SourceLine source="results/metrics/degradation_curve_data.json" />
      </section>

      <section className="section" aria-labelledby="interp">
        <h2 id="interp">Interpretation</h2>
        <p>
          Models on the zero-shot split (every test disease at 0 training edges) face the hardest
          case. TxGNN’s affinity head reaches these diseases by borrowing from structurally similar
          diseases that do have edges.
        </p>
        <p>
          Scope: PrimeKG, scaled reproduction (hidden_dim=64), seed=42,{' '}
          <code>results/metrics/degradation_curve_data.json</code>.
        </p>
      </section>
    </div>
  )
}
