import React from 'react'

// Provenance caption shown under every chart and table. The "source:" prefix
// is rendered via the .provenance ::before rule; the aria-label restates it.
export default function SourceLine({ source, extra }) {
  if (!source) return null
  return (
    <p className="provenance" aria-label={`Source: ${source}`}>
      {source}
      {extra && (
        <span style={{ fontFamily: 'inherit', fontWeight: 'normal' }}> — {extra}</span>
      )}
    </p>
  )
}
