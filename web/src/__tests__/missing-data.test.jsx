/** Zero-shot paper_reported must render explicit text, never a bare "0". */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

const DATA = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  paperReported: {
    zeroshot: {
      _relativeNote: 'relative only: +19.0% ind / +23.9% contra vs next-best baseline (abs. in Suppl. S1/S2)',
      txgnn_two_phase: { auprc_ind: '0.90 ± 0.02', auprc_contra: '0.80 ± 0.01', source: 'Suppl. S1/S2' },
    },
    standard: {},
  },
  comparisonTable: [
    {
      model: 'txgnn_two_phase', displayName: 'TxGNN Two-Phase', split: 'zeroshot',
      n_seeds_run: 3, reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.7357, auprc_ind_std: 0.0108, auroc_ind: 0.9503, auroc_ind_std: 0.0094,
      auprc_contra: 0.8316, auprc_contra_std: 0.0102, auroc_contra: 0.9734, auroc_contra_std: 0.0023,
      wall_s: 71.6,
      paper_reported: { auprc_ind: '0.90 ± 0.02', auprc_contra: '0.80 ± 0.01', source: 'Suppl. S1/S2' },
    },
    {
      model: 'gnn_no_kg', displayName: 'GNN Baseline (no-KG)', split: 'zeroshot',
      n_seeds_run: 3, reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.7044, auprc_ind_std: 0.0272, auroc_ind: 0.9342, auroc_ind_std: 0.0054,
      auprc_contra: 0.7252, auprc_contra_std: 0.0079, auroc_contra: 0.9315, auroc_contra_std: 0.0057,
      wall_s: 0.2, paper_reported: null,
    },
  ],
  ablationMatrix: { q6_decision: {}, matrix: [] },
  q6AblationTable: [],
  degradationCurveData: { raw: [], binned: [] },
  caseStudies: {},
  reportSections: [],
}

beforeEach(() => {
  global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(DATA) }))
})

vi.mock('recharts', () => ({
  BarChart: ({ children }) => React.createElement('div', {}, children),
  Bar: () => null, XAxis: () => null, YAxis: () => null, CartesianGrid: () => null,
  Tooltip: () => null, Legend: () => null,
  ResponsiveContainer: ({ children }) => React.createElement('div', {}, children),
}))

import { StudyDataProvider } from '../state/StudyData.jsx'
import ResultsView from '../views/ResultsView.jsx'

function mount() {
  return render(
    <MemoryRouter>
      <StudyDataProvider>
        <ResultsView />
      </StudyDataProvider>
    </MemoryRouter>
  )
}

describe('paper_reported rendering', () => {
  it('shows the relative-only note, never bare "0"', async () => {
    const { findByText } = mount()
    const el = await findByText(/relative only/i)
    expect(el).toBeTruthy()
    const cells = el.closest('body')?.querySelectorAll('td') || []
    expect(Array.from(cells).some((td) => td.textContent.trim() === '0')).toBe(false)
  })
  it('renders TxGNN paper AUPRC (0.90) for zero-shot', async () => {
    const { findByText } = mount()
    expect(await findByText(/0\.90/)).toBeTruthy()
  })
  it('has no [NOT YET RUN] in any cell', async () => {
    const { findByText, container } = mount()
    await findByText(/TxGNN Two-Phase/i)
    const cells = container.querySelectorAll('td')
    expect(Array.from(cells).some((td) => td.textContent.includes('[NOT YET RUN]'))).toBe(false)
  })
})
