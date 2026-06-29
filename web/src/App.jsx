import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { StudyDataProvider } from './state/StudyData.jsx'
import SideRail from './ui/SideRail.jsx'
import PageFooter from './ui/PageFooter.jsx'
import Home from './views/Home.jsx'
import Methods from './views/Methods.jsx'
import ResultsView from './views/ResultsView.jsx'
import Alternatives from './views/Alternatives.jsx'
import ZeroShot from './views/ZeroShot.jsx'
import Ablations from './views/Ablations.jsx'
import Cases from './views/Cases.jsx'
import FullReport from './views/FullReport.jsx'
import Evidence from './views/Evidence.jsx'

const ROUTES = [
  ['/', Home],
  ['/methods', Methods],
  ['/results', ResultsView],
  ['/q2-alternatives', Alternatives],
  ['/q3-zeroshot', ZeroShot],
  ['/q6-ablations', Ablations],
  ['/case-studies', Cases],
  ['/report', FullReport],
  ['/evidence', Evidence],
]

export default function App() {
  return (
    <BrowserRouter>
      <StudyDataProvider>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <SideRail />
        <main id="main-content" className="site-main">
          <Routes>
            {ROUTES.map(([path, View]) => (
              <Route key={path} path={path} element={<View />} />
            ))}
          </Routes>
        </main>
        <PageFooter />
      </StudyDataProvider>
    </BrowserRouter>
  )
}
