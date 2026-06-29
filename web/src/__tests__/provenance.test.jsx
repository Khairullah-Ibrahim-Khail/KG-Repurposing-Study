/** SourceLine must render a non-empty "source:" caption for any real path. */
import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import React from 'react'
import SourceLine from '../ui/SourceLine.jsx'

describe('SourceLine', () => {
  it('renders the source path', () => {
    const { container } = render(<SourceLine source="results/metrics/comparison_table.csv" />)
    const el = container.querySelector('.provenance')
    expect(el).not.toBeNull()
    expect(el.textContent).toContain('results/metrics/comparison_table.csv')
  })

  it('exposes the source via aria-label', () => {
    const { container } = render(<SourceLine source="results/ablations/matrix.json" />)
    const el = container.querySelector('.provenance')
    expect(el.getAttribute('aria-label')).toContain('Source:')
    expect(el.getAttribute('aria-label')).toContain('results/ablations/matrix.json')
  })

  it('renders extra text', () => {
    const { container } = render(
      <SourceLine source="results/metrics/comparison_table.csv" extra="zero-shot split" />
    )
    expect(container.querySelector('.provenance').textContent).toContain('zero-shot split')
  })

  it('renders nothing without a source', () => {
    expect(render(<SourceLine source="" />).container.firstChild).toBeNull()
    expect(render(<SourceLine />).container.firstChild).toBeNull()
  })

  it('is non-empty for every real path', () => {
    const paths = [
      'results/metrics/comparison_table.csv',
      'results/ablations/matrix.json',
      'results/metrics/degradation_curve_data.json',
      'results/predictions/case_study_caseA_txgnn.csv',
      'results/predictions/case_study_caseB_txgnn.csv',
    ]
    paths.forEach((p) => {
      const { container } = render(<SourceLine source={p} />)
      expect(container.querySelector('.provenance').textContent.trim().length).toBeGreaterThan(0)
    })
  })
})
