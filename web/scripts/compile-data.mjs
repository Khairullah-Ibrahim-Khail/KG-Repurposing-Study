/**
 * compile-data.mjs
 * Gathers every result file from ../../results and report markdown from
 * ../../report/sections into a single web/public/data/results.json.
 *
 * This file is the ONLY place paper-reported numbers are written; no React
 * component may hardcode a metric.
 * Source: Huang et al. (2024), Nature Medicine, Suppl. Tables S1/S2 (MOESM1 ESM).
 */

import { existsSync, readFileSync, writeFileSync } from 'fs'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, '../../')
const resultsDir = resolve(repoRoot, 'results')
const reportDir = resolve(repoRoot, 'report/sections')
const outPath = resolve(here, '../public/data/results.json')

// --- paper-reported values (single source of truth) ------------------------
const PAPER_REPORTED = {
  standard: {
    txgnn_two_phase: { auprc_ind: '0.91 ± 0.02', auprc_contra: '0.82 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2' },
    txgnn_attn_on: { auprc_ind: '0.91 ± 0.02', auprc_contra: '0.82 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2' },
    han_paper: { auprc_ind: '0.87 ± 0.18', auprc_contra: '0.84 ± 0.00',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2 (HAN row)' },
  },
  zeroshot: {
    txgnn_two_phase: { auprc_ind: '0.90 ± 0.02', auprc_contra: '0.80 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2' },
    txgnn_attn_on: { auprc_ind: '0.90 ± 0.02', auprc_contra: '0.80 ± 0.01',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1/S2' },
    _relativeNote:
      'relative only: +19.0% ind / +23.9% contra vs next-best baseline (abs. in Suppl. S1/S2)',
    bioBERT_paper: { auprc_ind: '0.76 ± 0.03', auprc_contra: null,
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S1 (BioBERT row)' },
    rgcn_paper: { auprc_ind: null, auprc_contra: '0.64 ± 0.03',
      source: 'Huang et al. 2024, Nat Med, Suppl. Table S2 (RGCN row)' },
  },
}

const DISPLAY = {
  gnn_no_kg: 'GNN Baseline (no-KG)',
  gnn_kg: 'GNN Baseline (KG)',
  txgnn_two_phase: 'TxGNN Two-Phase',
  single_stage: 'Single-Stage',
  joint_contrastive: 'Joint Contrastive',
  txgnn_attn_on: 'TxGNN (attn=ON)',
  txgnn_attn_off: 'TxGNN (attn=OFF)',
}

// --- tiny readers ----------------------------------------------------------
const num = (v) => {
  const n = parseFloat(v)
  return Number.isNaN(n) ? null : n
}

function parseCSV(text) {
  const lines = text.trim().split('\n')
  const headers = lines[0].split(',').map((h) => h.trim())
  return lines.slice(1).map((line) => {
    const cells = []
    let buf = ''
    let quoted = false
    for (const ch of line) {
      if (ch === '"') quoted = !quoted
      else if (ch === ',' && !quoted) { cells.push(buf.trim()); buf = '' }
      else buf += ch
    }
    cells.push(buf.trim())
    return Object.fromEntries(headers.map((h, i) => [h, cells[i] ?? '']))
  })
}

const readJSON = (p) => (existsSync(p) ? JSON.parse(readFileSync(p, 'utf-8')) : null)
const readCSV = (p) => (existsSync(p) ? parseCSV(readFileSync(p, 'utf-8')) : [])
const readText = (p) => (existsSync(p) ? readFileSync(p, 'utf-8') : '')

// --- builders --------------------------------------------------------------
function comparisonTable() {
  return readCSV(resolve(resultsDir, 'metrics/comparison_table.csv'))
    .filter((r) => r.auprc_ind && r.auprc_ind !== '[NOT YET RUN]')
    .map((r) => ({
      model: r.model,
      displayName: DISPLAY[r.model] || r.model,
      split: r.split,
      n_seeds_run: parseInt(r.n_seeds_run) || 0,
      reproduction_type: r.reproduction_type,
      auprc_ind: num(r.auprc_ind),
      auprc_ind_std: num(r.auprc_ind_std),
      auroc_ind: num(r.auroc_ind),
      auroc_ind_std: num(r.auroc_ind_std),
      auprc_contra: num(r.auprc_contra),
      auprc_contra_std: num(r.auprc_contra_std),
      auroc_contra: num(r.auroc_contra),
      auroc_contra_std: num(r.auroc_contra_std),
      wall_s: num(r.wall_s),
      paper_reported: PAPER_REPORTED[r.split]?.[r.model] || null,
    }))
}

function q6Table() {
  return readCSV(resolve(resultsDir, 'metrics/q6_ablation_table.csv')).map((r) => ({
    variant: r.variant,
    displayName: DISPLAY[r.variant] || r.variant,
    split: r.split,
    auprc_ind_mean: parseFloat(r.auprc_ind_mean),
    auprc_ind_std: parseFloat(r.auprc_ind_std),
    n_seeds: parseInt(r.n_seeds),
  }))
}

function degradation() {
  const raw = readJSON(resolve(resultsDir, 'metrics/degradation_curve_data.json'))
  if (!raw) return { raw: [], binned: [] }

  const order = ['0', '1-5', '6-20', '21+']
  const bucket = (n) => (n === 0 ? '0' : n <= 5 ? '1-5' : n <= 20 ? '6-20' : '21+')

  const groups = {}
  for (const row of raw) {
    const key = `${row.model}|||${row.relation}|||${bucket(row.n_train_edges)}`
    ;(groups[key] ||= []).push(row.auprc)
  }
  const binned = Object.entries(groups).map(([key, vals]) => {
    const [model, relation, bin] = key.split('|||')
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    return { model, relation, bin, mean_auprc: parseFloat(mean.toFixed(4)), n: vals.length }
  })
  binned.sort((a, b) =>
    a.model !== b.model ? a.model.localeCompare(b.model)
      : a.relation !== b.relation ? a.relation.localeCompare(b.relation)
        : order.indexOf(a.bin) - order.indexOf(b.bin)
  )

  let sample = raw
  if (raw.length > 500) {
    const step = Math.ceil(raw.length / 500)
    sample = raw.filter((_, i) => i % step === 0)
  }
  return { raw: sample, binned }
}

function caseStudies() {
  const pred = (tag) => readCSV(resolve(resultsDir, `predictions/case_study_${tag}_txgnn.csv`))
  const pathRows = (tag) =>
    readCSV(resolve(resultsDir, `predictions/case_study_${tag}_paths_txgnn.csv`)).map((r) => ({
      drug_id: r.drug_id,
      via_entity: r.via_entity,
      via_type: r.via_type,
      relation_drug_to_entity: r.relation_drug_to_entity,
      relation_entity_to_disease: r.relation_entity_to_disease,
      disease_id: r.disease_id,
    }))

  const toPred = (r) => ({
    disease_id: r.disease_id,
    relation: r.relation,
    rank: parseInt(r.rank),
    drug_id: r.drug_id,
    drug_name: r.drug_name,
    score: parseFloat(r.score),
    is_positive: r.is_positive === 'True' || r.is_positive === 'true',
  })

  const a = pred('caseA')
  const b = pred('caseB')
  return {
    caseA: {
      diseaseId: '24573',
      diseaseName: 'Familial Hypertrophic Cardiomyopathy',
      shortName: 'FHC',
      nPos: 1,
      relation: 'indication',
      auprc: 0.025,
      source: 'results/predictions/case_study_caseA_txgnn.csv',
      predictions: a.filter((r) => r.relation === 'indication').map(toPred),
      contraindications: a.filter((r) => r.relation === 'contraindication').map(toPred),
      paths: pathRows('caseA'),
      pathsSource: 'results/predictions/case_study_caseA_paths_txgnn.csv',
      knownDrug: 'Propranolol (DB00571)',
      knownDrugInTop20: false,
      notes:
        'All top-20 scores negative. Propranolol (the one known indication) is not in the top-20. The model fails on this rare disease.',
    },
    caseB: {
      diseaseId: '5545',
      diseaseName: 'Staphylococcus Aureus Infection',
      shortName: 'S. aureus',
      nPos: 45,
      relation: 'indication',
      auprc: 0.088,
      source: 'results/predictions/case_study_caseB_txgnn.csv',
      predictions: b.map(toPred),
      paths: pathRows('caseB'),
      pathsSource: 'results/predictions/case_study_caseB_paths_txgnn.csv',
      firstPositiveRank: 18,
      firstPositiveDrug: 'Benzylpenicillin (DB01053)',
      notes:
        'Benzylpenicillin at rank 18 (positive). Mupirocin (rank 3) and Doxycycline (rank 5) are plausible; three cancer drugs land in the top-10.',
    },
  }
}

function reportSections() {
  const order = [
    ['01_introduction.md', 'Introduction'],
    ['02_methods.md', 'Methods'],
    ['03_results.md', 'Results'],
    ['04_case_studies.md', 'Case Studies'],
    ['05_discussion.md', 'Discussion'],
  ]
  return order.map(([filename, title]) => ({
    filename,
    title,
    content: readText(resolve(reportDir, filename)),
    source: `report/sections/${filename}`,
  }))
}

// --- emit ------------------------------------------------------------------
const payload = {
  generatedAt: new Date().toISOString(),
  paperReportedNote:
    'paper_reported values from Huang et al. (2024), Nature Medicine, Suppl. Tables S1/S2 (MOESM1 ESM). DOI: 10.1038/s41591-024-03233-x',
  paperReported: PAPER_REPORTED,
  comparisonTable: comparisonTable(),
  ablationMatrix: readJSON(resolve(resultsDir, 'ablations/matrix.json')),
  q6AblationTable: q6Table(),
  degradationCurveData: degradation(),
  caseStudies: caseStudies(),
  reportSections: reportSections(),
}

writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf-8')
console.log(`[compile-data] wrote ${outPath}`)
console.log(`[compile-data] comparison rows: ${payload.comparisonTable.length}`)
console.log(`[compile-data] report sections: ${payload.reportSections.length}`)
