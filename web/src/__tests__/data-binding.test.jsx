/** ResultsView renders values straight from results.json (nothing hardcoded). */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

const SAMPLE = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  comparisonTable: [
    {
      model: 'gnn_no_kg', displayName: 'GNN Baseline (no-KG)', split: 'standard',
      n_seeds_run: 3, reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.8926, auprc_ind_std: 0.0061, auroc_ind: 0.9719, auroc_ind_std: 0.0024,
      auprc_contra: 0.9453, auprc_contra_std: 0.0023, auroc_contra: 0.9913, auroc_contra_std: 0.0003,
      wall_s: 0.7, paper_reported: null,
    },
    {
      model: 'gnn_kg', displayName: 'GNN Baseline (KG)', split: 'standard',
      n_seeds_run: 3, reproduction_type: 'scaled_reproduction',
      auprc_ind: 0.8245, auprc_ind_std: 0.022, auroc_ind: 0.9716, auroc_ind_std: 0.0055,
      auprc_contra: 0.895, auprc_contra_std: 0.0262, auroc_contra: 0.9836, auroc_contra_std: 0.0034,
      wall_s: 145.5, paper_reported: null,
    },
  ],
  ablationMatrix: { q6_decision: {}, matrix: [] },
  q6AblationTable: [],
  degradationCurveData: { raw: [], binned: [] },
  caseStudies: {},
  reportSections: [],
  paperReported: { zeroshot: { _relativeNote: 'relative only: +19.0% ind / +23.9% contra vs next-best' } },
}

beforeEach(() => {
  global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(SAMPLE) }))
})

vi.mock('recharts', () => ({
  BarChart: ({ children }) => React.createElement('div', { 'data-testid': 'bar-chart' }, children),
  Bar: () => null, XAxis: () => null, YAxis: () => null, CartesianGrid: () => null,
  Tooltip: () => null, Legend: () => null,
  ResponsiveContainer: ({ children }) => React.createElement('div', {}, children),
}))

import { StudyDataProvider } from '../state/StudyData.jsx'
import ResultsView from '../views/ResultsView.jsx'

function mount() {
  render(
    <MemoryRouter>
      <StudyDataProvider>
        <ResultsView />
      </StudyDataProvider>
    </MemoryRouter>
  )
}

describe('data-binding', () => {
  it('renders gnn_no_kg AUPRC from JSON', async () => {
    mount()
    expect((await screen.findAllByText(/0\.893/)).length).toBeGreaterThan(0)
  })
  it('renders gnn_kg AUPRC from JSON', async () => {
    mount()
    expect((await screen.findAllByText(/0\.825/)).length).toBeGreaterThan(0)
  })
  it('shows the zero-shot relative-only note', async () => {
    mount()
    expect((await screen.findAllByText(/relative only|19\.0%/i)).length).toBeGreaterThan(0)
  })
  it('shows the display name from JSON', async () => {
    mount()
    expect((await screen.findAllByText(/GNN Baseline \(no-KG\)/i)).length).toBeGreaterThan(0)
  })
})
