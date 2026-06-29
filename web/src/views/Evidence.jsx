import React from 'react'

const REFERENCES = [
  {
    id: 'huang2024',
    label: 'Huang et al. 2024 — TxGNN (primary)',
    citation:
      'Huang K, Chandak P, Wang Q, Havaldar S, Vaid A, Leskovec J, Nadkarni GN, Glicksberg BS, Gehlenborg N, Zitnik M. "A foundation model for clinician-centered drug repurposing." Nature Medicine, 2024.',
    doi: 'https://doi.org/10.1038/s41591-024-03233-x',
    pmc: 'PMC11326339 (preprint), PMC11645266 (final)',
    usedFor: 'Q1–Q6 (paper being reproduced)',
    claims: [
      'Standard TxGNN indication AUPRC 0.91 ± 0.02 (Suppl. S1)',
      'Standard HAN indication AUPRC 0.87 ± 0.18 (Suppl. S1)',
      'Standard TxGNN contraindication AUPRC 0.82 ± 0.01 (Suppl. S2)',
      'Standard HAN contraindication AUPRC 0.84 ± 0.00 (Suppl. S2)',
      'Zero-shot TxGNN indication AUPRC 0.90 ± 0.02 (Suppl. S1)',
      'Zero-shot TxGNN contraindication AUPRC 0.80 ± 0.01 (Suppl. S2)',
      'Zero-shot relative gains +19.0% ind / +23.9% contra vs next-best (Fig 2d)',
      '92% of PrimeKG diseases have no indication edge (main text)',
      'Learnable attention found ineffective; replaced by fixed degree gating λ=0.7 (Methods)',
    ],
    notes: [
      'Zero-shot absolute values from Suppl. Tables S1/S2 (MOESM1 ESM)',
      '49.2% / 35.1% abstract headline cannot be tied to a split; not used in tables',
      'Standard contraindication: HAN (0.84) beats TxGNN (0.82); the gain claim is indication-only',
    ],
  },
  {
    id: 'chandak2023',
    label: 'Chandak et al. 2023 — PrimeKG',
    citation:
      'Chandak P, Huang K, Zitnik M. "Building a knowledge graph to enable precision medicine." Scientific Data, 10, 67 (2023).',
    doi: 'https://doi.org/10.1038/s41597-023-01960-3',
    usedFor: 'Data description',
    claims: [
      '129,375 nodes, 8,100,498 edges (our download)',
      'Harvard Dataverse: doi:10.7910/DVN/IXA7BM',
    ],
  },
  {
    id: 'kgbert2019',
    label: 'Yao et al. 2019 — KG-BERT',
    citation: 'Yao L, Mao C, Luo Y. "KG-BERT: BERT for Knowledge Graph Completion." arXiv:1909.03193, 2019.',
    doi: 'https://arxiv.org/abs/1909.03193',
    usedFor: 'Background — LM+KG lineage',
    claims: ['Treats KG triples as text for valid/invalid classification'],
    notes: ['Numbers not confirmed from PDF — background only'],
  },
  {
    id: 'dragon2022',
    label: 'Yasunaga et al. 2022 — DRAGON',
    citation: 'Yasunaga M, Bosselut A, Ren H, et al. "Deep Bidirectional Language-Knowledge Graph Pretraining." NeurIPS 2022.',
    doi: 'https://arxiv.org/abs/2210.09338',
    usedFor: 'Background — joint LM+KG pretraining',
    claims: ['+5% average gain on downstream QA (abstract)'],
    notes: ['ConceptNet/UMLS, not PrimeKG — numbers do not transfer to drug repurposing'],
  },
]

const FILES = [
  ['results/metrics/comparison_table.csv', 'Full model comparison', 'Q1, Q2, Q5'],
  ['results/metrics/q6_ablation_table.csv', 'Attention ablation AUPRC', 'Q6'],
  ['results/ablations/matrix.json', 'Ablation matrix + Q6 decision', 'Q6'],
  ['results/metrics/degradation_curve_data.json', 'Per-disease AUPRC vs edge count', 'Q3'],
  ['results/predictions/case_study_caseA_txgnn.csv', 'FHC top-20 predictions', 'Q4'],
  ['results/predictions/case_study_caseB_txgnn.csv', 'S. aureus top-20 predictions', 'Q4'],
  ['results/metrics/leakage_check_seed{n}.json', 'Leakage audit (PASS, seeds 42/0/1)', 'All zero-shot'],
]

const NOT_USED = [
  ['+49.2% / +35.1% headline (abstract)', 'Cannot be tied to a split in the main text; abstract only.'],
  ['DRAGON +5% / +10% (QA)', 'Different domain (ConceptNet) — does not transfer to PrimeKG.'],
  ['transformer_kg / transformer_nokg results', '[NOT YET RUN] — excluded from the table.'],
]

export default function Evidence() {
  return (
    <div className="page">
      <h1>Evidence and Sources</h1>
      <p>
        Every number traces to a result file in <code>results/</code> or to a primary citation below.
      </p>

      <div className="warn-box">
        <p>
          <strong>Provenance rule:</strong> paper numbers live in the <code>paper_reported</code>
          column with their source; our numbers live in <code>scaled_reproduction</code>. They are
          never merged.
        </p>
      </div>

      <div className="finding-box">
        <p>
          <strong>Scope:</strong> all experiments use GNN-based models (HGT). TxGNN is not a language
          model. KG-BERT and DRAGON are background only — different datasets, not compared here.
        </p>
      </div>

      {REFERENCES.map((ref) => (
        <section key={ref.id} className="section"
                 style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
          <h2>{ref.label}</h2>
          <p><strong>Citation:</strong> {ref.citation}</p>
          {ref.doi && (
            <p><strong>DOI:</strong>{' '}
              <a href={ref.doi} target="_blank" rel="noopener noreferrer">{ref.doi}</a></p>
          )}
          {ref.pmc && <p><strong>PMC:</strong> {ref.pmc}</p>}
          <p><strong>Used for:</strong> {ref.usedFor}</p>
          <h3>Claims used</h3>
          <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            {ref.claims.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
          {ref.notes && (
            <>
              <h3>Notes and caveats</h3>
              <ul style={{ paddingLeft: '1.5rem' }}>
                {ref.notes.map((n, i) => <li key={i} style={{ color: 'var(--structure)' }}>{n}</li>)}
              </ul>
            </>
          )}
        </section>
      ))}

      <section className="section" style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
        <h2>Result Files (scaled reproduction)</h2>
        <table>
          <thead><tr><th>File</th><th>Content</th><th>Serves</th></tr></thead>
          <tbody>
            {FILES.map(([f, c, s], i) => (
              <tr key={i}><td><code>{f}</code></td><td>{c}</td><td>{s}</td></tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="section" style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
        <h2>Claims NOT Used (and Why)</h2>
        <table>
          <thead><tr><th>Claim</th><th>Reason</th></tr></thead>
          <tbody>
            {NOT_USED.map(([claim, why], i) => (
              <tr key={i}><td>{claim}</td><td>{why}</td></tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
