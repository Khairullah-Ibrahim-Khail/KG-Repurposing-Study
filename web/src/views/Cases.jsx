import React, { useState } from 'react'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

function PredTable({ rows, source }) {
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Rank</th><th>Drug</th><th>DrugBank ID</th><th>Score</th><th>Known positive?</th></tr>
          </thead>
          <tbody>
            {rows.slice(0, 20).map((row, i) => (
              <tr key={i} style={row.is_positive ? { background: 'var(--indication-muted)', fontWeight: 600 } : {}}>
                <td>{row.rank}</td>
                <td>{row.drug_name}</td>
                <td><code>{row.drug_id}</code></td>
                <td style={{ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>
                  {row.score?.toFixed(4)}
                </td>
                <td>
                  {row.is_positive
                    ? <span className="positive-badge">Yes</span>
                    : <span className="negative-badge">No</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source={source} />
    </>
  )
}

function PathTable({ rows, source }) {
  if (!rows || rows.length === 0)
    return <p style={{ color: 'var(--structure)' }}>No path data available.</p>
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Drug ID</th><th>Via entity</th><th>Via type</th>
              <th>Drug → entity</th><th>Entity → disease</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p, i) => (
              <tr key={i}>
                <td><code>{p.drug_id}</code></td>
                <td>{p.via_entity}</td>
                <td>{p.via_type}</td>
                <td><code>{p.relation_drug_to_entity}</code></td>
                <td><code>{p.relation_entity_to_disease}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine source={source} />
    </>
  )
}

export default function Cases() {
  const { data, loading, error } = useStudyData()
  const [active, setActive] = useState('caseA')

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const { caseA, caseB } = data.caseStudies || {}

  return (
    <div className="page">
      <h1>Q4 — Case Studies</h1>
      <p>
        Two diseases fixed before predictions were seen. Drug names, scores, and KG paths all come
        from files in <code>results/predictions/</code>.
      </p>
      <div className="warn-box">
        <p>
          <strong>On the paper’s cases:</strong> Huang et al. (2024) ran their case studies with the
          full TxGNN (512-dim, pre-trained features). Those cannot be reproduced here — the full
          model exceeds 8&thinsp;GB and the features are not released. These are <em>original</em>
          cases on the scaled model, drawn from diseases in our zero-shot test split.
        </p>
      </div>

      <div className="seg" role="group" aria-label="Select case study">
        <button className="seg-item" aria-pressed={active === 'caseA'} onClick={() => setActive('caseA')}>
          Case A · FHC (rare, n_pos=1)
        </button>
        <button className="seg-item" aria-pressed={active === 'caseB'} onClick={() => setActive('caseB')}>
          Case B · S. aureus (n_pos=45)
        </button>
      </div>

      {active === 'caseA' && caseA && (
        <section aria-labelledby="caseA">
          <h2 id="caseA">Case A: {caseA.diseaseName}</h2>
          <table style={{ marginBottom: '1rem' }}>
            <tbody>
              <tr><th scope="row">Disease ID</th><td>{caseA.diseaseId}</td></tr>
              <tr><th scope="row">Known positives</th><td>{caseA.nPos}</td></tr>
              <tr><th scope="row">Known indication drug</th><td>{caseA.knownDrug}</td></tr>
              <tr><th scope="row">In top-20?</th><td><span className="negative-badge">No — model fails</span></td></tr>
              <tr><th scope="row">Per-disease AUPRC</th><td>{caseA.auprc}</td></tr>
              <tr><th scope="row">Split</th><td>Zero-shot (held out from training)</td></tr>
            </tbody>
          </table>
          <p>
            Familial hypertrophic cardiomyopathy is an inherited thickening of the heart muscle
            (sarcomere-gene mutations: MYH7, MYBPC3). The only PrimeKG indication edge is Propranolol
            (DB00571), a non-selective beta-blocker.
          </p>
          <div className="warn-box">
            <p>
              All top-20 indication scores are negative; Propranolol is absent from the top-20.
              AUPRC = {caseA.auprc}. The model fails on this rare disease.
            </p>
          </div>
          <h3>Top-20 Indication Predictions</h3>
          <PredTable rows={caseA.predictions} source={caseA.source} />
          <h3>Top-20 Contraindication Predictions</h3>
          <PredTable rows={caseA.contraindications || []} source={caseA.source} />
          <h3>KG Paths for Known Drugs</h3>
          <p>
            Traced drugs (Milrinone, Amrinone, Dipyridamole) are FHC <em>contraindications</em>, not
            indications. Their paths reach FHC through disease-disease similarity
            (hypertrophic cardiomyopathy → FHC) and through Propranolol via drug-drug edges.
          </p>
          <PathTable rows={caseA.paths} source={caseA.pathsSource} />
        </section>
      )}

      {active === 'caseB' && caseB && (
        <section aria-labelledby="caseB">
          <h2 id="caseB">Case B: {caseB.diseaseName}</h2>
          <table style={{ marginBottom: '1rem' }}>
            <tbody>
              <tr><th scope="row">Disease ID</th><td>{caseB.diseaseId}</td></tr>
              <tr><th scope="row">Known positives</th><td>{caseB.nPos}</td></tr>
              <tr><th scope="row">First positive in top-20</th><td>{caseB.firstPositiveDrug} at rank {caseB.firstPositiveRank}</td></tr>
              <tr><th scope="row">Per-disease AUPRC</th><td>{caseB.auprc}</td></tr>
              <tr><th scope="row">Split</th><td>Zero-shot (held out from training)</td></tr>
            </tbody>
          </table>
          <p>
            Staphylococcus aureus is a common gram-positive pathogen; many antibiotic classes are
            indicated (beta-lactams, tetracyclines, macrolides, topicals). MRSA needs mupirocin,
            daptomycin, or vancomycin.
          </p>
          <div className="finding-box">
            <p>
              Benzylpenicillin appears at rank 18 (positive). Plausible antibiotics in the top-20:
              Mupirocin (3), Doxycycline (5), Minocycline (10), Benzylpenicillin (18). Three
              antineoplastics land in the top-10 (Etoposide 2, Carboplatin 6, Bleomycin 8).
            </p>
          </div>
          <h3>Top-20 Indication Predictions</h3>
          <PredTable rows={caseB.predictions} source={caseB.source} />
          <h3>KG Paths for Known Drugs</h3>
          <p>
            Traced drugs (Cefprozil, Cefdinir, Tazobactam) reach the target via a related node
            (“staphylococcal infection”) and disease-disease similarity — the same mechanism as Case A.
          </p>
          <PathTable rows={caseB.paths} source={caseB.pathsSource} />
        </section>
      )}

      <h2 style={{ marginTop: '2rem' }}>Interpretation Notes</h2>
      <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
        <li>Both cases use the scaled model (64-dim, no pre-trained features); rankings are not clinical advice.</li>
        <li>The disease-disease similarity pathway dominates both, matching the Q6 finding.</li>
        <li>Case A (n_pos=1): zero-shot rare-disease generalization is not confirmed here.</li>
        <li>Case B (n_pos=45): plausible antibiotics surface, but 64-dim embedding noise yields cancer drugs.</li>
      </ul>
    </div>
  )
}
