/** FullReport renders all 5 report sections as tabbed markdown. */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'

const SECTIONS = [
  { filename: '01_introduction.md', title: 'Introduction',
    content: '# 1. Introduction\n\nDrug repurposing — finding new uses for approved drugs.',
    source: 'report/sections/01_introduction.md' },
  { filename: '02_methods.md', title: 'Methods',
    content: '# 2. Methods\n\nPrimeKG from Harvard Dataverse.', source: 'report/sections/02_methods.md' },
  { filename: '03_results.md', title: 'Results',
    content: '# 3. Results\n\nNumbers link to `results/`.', source: 'report/sections/03_results.md' },
  { filename: '04_case_studies.md', title: 'Case Studies',
    content: '# 4. Case Studies\n\nFixed before predictions.', source: 'report/sections/04_case_studies.md' },
  { filename: '05_discussion.md', title: 'Discussion',
    content: '# 5. Discussion\n\nKG does not consistently help.', source: 'report/sections/05_discussion.md' },
]

const DATA = {
  generatedAt: '2026-06-24T00:00:00.000Z',
  comparisonTable: [], ablationMatrix: { q6_decision: {}, matrix: [] },
  q6AblationTable: [], degradationCurveData: { raw: [], binned: [] },
  caseStudies: {}, reportSections: SECTIONS,
}

beforeEach(() => {
  global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(DATA) }))
})

import { StudyDataProvider } from '../state/StudyData.jsx'
import FullReport from '../views/FullReport.jsx'

function mount() {
  return render(
    <MemoryRouter>
      <StudyDataProvider>
        <FullReport />
      </StudyDataProvider>
    </MemoryRouter>
  )
}

describe('report sections', () => {
  it('renders all 5 tabs', async () => {
    const { findByText } = mount()
    for (const s of SECTIONS) expect(await findByText(s.title)).toBeTruthy()
  })
  it('renders the first section content', async () => {
    const { findByText } = mount()
    expect(await findByText(/Drug repurposing/i)).toBeTruthy()
  })
  it('shows a provenance caption with report/sections/', async () => {
    const { container, findByText } = mount()
    await findByText('Introduction')
    const caps = container.querySelectorAll('.provenance')
    expect(caps.length).toBeGreaterThan(0)
    expect(caps[0].textContent).toContain('report/sections/')
  })
  it('renders exactly 5 tab buttons', async () => {
    const { container, findByText } = mount()
    await findByText('Introduction')
    expect(container.querySelectorAll('button').length).toBe(5)
  })
})
