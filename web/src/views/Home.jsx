import React from 'react'
import { Link } from 'react-router-dom'
import { useStudyData } from '../state/StudyData.jsx'
import GraphPath from '../ui/GraphPath.jsx'
import SourceLine from '../ui/SourceLine.jsx'

const PATH_NODES = [
  { id: 'fhc', label: 'FHC', type: 'disease' },
  { id: 'myh7', label: 'MYH7', type: 'gene' },
  { id: 'propranolol', label: 'Propranolol', type: 'drug' },
]
const PATH_EDGES = [
  { source: 'fhc', target: 'myh7', label: 'gene_associated_with_disease' },
  { source: 'myh7', target: 'propranolol', label: 'treats' },
]

const QUESTIONS = [
  {
    num: 'Q1', to: '/results', tone: 'unexpected',
    title: 'Does KG augmentation beat a no-KG baseline?',
    finding:
      'No improvement on indication AUPRC. On the standard split the no-KG model wins (0.893 vs 0.825); KG helps only zero-shot contraindication.',
  },
  {
    num: 'Q2', to: '/q2-alternatives', tone: 'expected',
    title: 'Is there a better alternative to two-phase training?',
    finding:
      'No. Two-phase TxGNN leads zero-shot indication (0.736 vs 0.706 vs 0.670) and is the fastest of the three.',
  },
  {
    num: 'Q3', to: '/q3-zeroshot', tone: 'confirmed',
    title: 'Why is zero-shot prediction preferred?',
    finding:
      '92% of PrimeKG diseases have no approved therapy. A supervised classifier has no labels for them — zero-shot is a necessity, not a preference.',
  },
  {
    num: 'Q4', to: '/case-studies', tone: 'mixed',
    title: 'What do the two case studies reveal?',
    finding:
      'FHC (n_pos=1): model fails, Propranolol not in top-20. S. aureus (n_pos=45): Benzylpenicillin at rank 18, three cancer drugs in the top-10.',
  },
  {
    num: 'Q5', to: '/results', tone: 'neutral',
    title: 'How does a plain GNN compare to TxGNN?',
    finding:
      'Standard split: gnn_no_kg wins (0.893). Zero-shot: TxGNN two-phase leads (0.736). Table generated straight from CSV.',
  },
  {
    num: 'Q6', to: '/q6-ablations', tone: 'unexpected',
    title: 'Is attention augmentation in TxGNN optional?',
    finding:
      'It is detrimental. Dropping HGT attention raises zero-shot indication AUPRC from 0.736 to 0.772 (delta −0.036).',
  },
]

const TONE_COLOR = {
  expected: 'var(--indication)',
  unexpected: 'var(--contra)',
  confirmed: 'var(--accent)',
  mixed: 'var(--structure)',
  neutral: 'var(--structure)',
}
const TONE_LABEL = {
  expected: 'confirmed hypothesis',
  unexpected: 'unexpected finding',
  confirmed: 'confirmed',
  mixed: 'mixed',
  neutral: 'neutral',
}

function Divider() {
  return (
    <div className="section-divider" aria-hidden="true">
      <svg width="48" height="16" viewBox="0 0 48 16" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="6" cy="8" r="5" stroke="var(--accent)" strokeWidth="1.5" fill="none" />
        <line x1="11" y1="8" x2="37" y2="8" stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx="42" cy="8" r="5" stroke="var(--accent)" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}

const SCOPE = [
  ['Data', 'PrimeKG (Chandak et al., 2023) — 129,375 nodes, 8,100,498 edges, 10 node types'],
  ['Task', 'Drug indication and contraindication prediction'],
  ['Splits', 'Standard (diseases shared) and zero-shot (641 held-out diseases)'],
  ['Metrics', 'AUPRC (primary), AUROC (secondary). Random-negative 1:5 evaluation.'],
  ['Seeds', '[42, 0, 1] — mean ± std reported'],
  ['Hardware', 'NVIDIA GeForce RTX 4060, 8 GB VRAM. CUDA throughout.'],
  ['Model scale', 'hidden_dim=64, 2 layers, 4 heads (published: 512, 3, 8)'],
]

export default function Home() {
  const { loading, error } = useStudyData()
  return (
    <div className="page">
      <div className="overview-hero">
        <h1>TxGNN Drug Repurposing</h1>
        <p className="subtitle">A scaled reproduction on PrimeKG · RTX 4060 8&thinsp;GB</p>
      </div>

      <GraphPath nodes={PATH_NODES} edges={PATH_EDGES} />
      <SourceLine source="results/predictions/case_study_caseA_paths_txgnn.csv" />

      <div className="finding-box">
        <p>
          <strong>Scaled reproduction.</strong> Every model uses <code>hidden_dim=64</code> instead
          of the published <code>512</code> because of the 8&thinsp;GB VRAM budget. Outputs carry the{' '}
          <span className="badge badge-scaled">scaled_reproduction</span> tag; paper figures from
          Huang et al. (2024) <span className="badge badge-paper">paper_reported</span> sit in a
          separate column and are never merged with ours.
        </p>
      </div>

      <p style={{ color: 'var(--structure)', marginBottom: '1.5rem', maxWidth: '72ch' }}>
        Six empirical questions about knowledge-graph augmentation for drug repurposing. Every figure
        traces back to a file in <code>results/</code>.
      </p>

      {loading && <p className="status-loading">Loading data…</p>}
      {error && (
        <div className="status-error">
          <strong>Could not load results.json:</strong> {error}
          <br />
          Run <code>npm run build:data</code> inside <code>web/</code> first.
        </div>
      )}

      <Divider />
      <h2>Six Research Questions</h2>
      <ol className="question-list" style={{ listStyle: 'none' }}>
        {QUESTIONS.map((q) => (
          <li key={q.num} className="question-item">
            <span className="question-num">{q.num}</span>
            <div className="question-body">
              <Link to={q.to} className="question-title">{q.title}</Link>
              <p className="question-finding">{q.finding}</p>
              <span className="outcome-tag" style={{ color: TONE_COLOR[q.tone] }}>
                {TONE_LABEL[q.tone]}
              </span>
            </div>
            <Link to={q.to} className="question-arrow" aria-label={`Open ${q.num}`}>→</Link>
          </li>
        ))}
      </ol>

      <Divider />
      <h2>Project Scope</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            {SCOPE.map(([k, v]) => (
              <tr key={k}>
                <th scope="row">{k}</th>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Divider />
      <h2>Leakage Check</h2>
      <p>
        The zero-shot split passes the audit: no held-out disease appears in any training treatment
        edge, verified for seeds [42, 0, 1]. Files:{' '}
        <code>results/metrics/leakage_check_seed{'{n}'}.json</code> — all PASS.
      </p>

      <h2>Citation</h2>
      <p>
        Huang K, Chandak P, Wang Q, et al. “A foundation model for clinician-centered drug
        repurposing.” <em>Nature Medicine</em>, 2024.{' '}
        <a href="https://doi.org/10.1038/s41591-024-03233-x" target="_blank" rel="noopener noreferrer">
          DOI: 10.1038/s41591-024-03233-x
        </a>
      </p>
    </div>
  )
}
