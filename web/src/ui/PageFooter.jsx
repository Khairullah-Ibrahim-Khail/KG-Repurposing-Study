import React from 'react'
import { useStudyData } from '../state/StudyData.jsx'

export default function PageFooter() {
  const { data } = useStudyData()
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="site-footer-inner">
        <span>
          Built from <code>results/</code> at compile time — no values hardcoded in the UI
        </span>
        {data?.generatedAt && (
          <span>
            Built: <time dateTime={data.generatedAt}>{data.generatedAt}</time>
          </span>
        )}
      </div>
    </footer>
  )
}
