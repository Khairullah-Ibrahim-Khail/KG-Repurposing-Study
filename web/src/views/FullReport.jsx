import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useStudyData } from '../state/StudyData.jsx'
import SourceLine from '../ui/SourceLine.jsx'

export default function FullReport() {
  const { data, loading, error } = useStudyData()
  const [active, setActive] = useState(0)

  if (loading) return <div className="page status-loading">Loading...</div>
  if (error) return <div className="page"><div className="status-error">{error}</div></div>

  const sections = data.reportSections || []
  const current = sections[active]

  return (
    <div className="page">
      <h1>Full Report</h1>
      <p>
        The five report sections, rendered from <code>report/sections/</code>. Content is loaded from
        files, not typed into this page.
      </p>

      <div className="report-tabs" role="tablist">
        {sections.map((s, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={active === i}
            className={`report-tab${active === i ? ' is-active' : ''}`}
            onClick={() => setActive(i)}
          >
            {s.title}
          </button>
        ))}
      </div>

      {sections.length === 0 && (
        <p style={{ color: 'var(--structure)' }}>
          No report sections found. Run <code>npm run build:data</code> first.
        </p>
      )}

      {current && (
        <div className="report-card">
          <div className="report-body">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: (props) => (
                  <div className="table-scroll">
                    <table {...props} />
                  </div>
                ),
              }}
            >
              {current.content}
            </ReactMarkdown>
          </div>
          <SourceLine source={current.source} />
        </div>
      )}
    </div>
  )
}
