# Web — Drug Repurposing Study Site

Static Vite + React site that presents the scaled-TxGNN results. It reads a
single JSON blob compiled from `../results/` — no metric is hardcoded in a
component.

## Commands (from `web/`)

```bash
npm install
npm run build:data   # compile public/data/results.json from ../results + ../report
npm run dev          # dev server
npm run build        # production build (runs build:data first)
npm run preview      # preview the build
npm run test         # vitest
```

## How data flows
1. `scripts/compile-data.mjs` reads `../results/` and `../report/sections/`.
2. It writes `public/data/results.json`.
3. `state/StudyData.jsx` fetches that once and shares it via context.

`compile-data.mjs` is the **only** place `paper_reported` values live.

## Structure
- `ui/` — SideRail, PageFooter, SourceLine, GraphPath, TopNav
- `views/` — Home, Methods, ResultsView (Q1+Q5), Alternatives (Q2),
  ZeroShot (Q3), Cases (Q4), Ablations (Q6), FullReport, Evidence
- `state/StudyData.jsx` — single fetch + context

## Routes
`/` Overview · `/methods` · `/results` (Q1+Q5) · `/q2-alternatives` ·
`/q3-zeroshot` · `/q6-ablations` · `/case-studies` · `/report` · `/evidence`

## Provenance
Every chart and table ends with a `SourceLine` pointing at its result file.

## Deploy (Vercel)
Build command `npm run build:data && npm run build`, output `dist`, framework
`vite`, root directory `web/`.
