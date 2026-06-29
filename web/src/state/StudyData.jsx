import React, { createContext, useContext, useEffect, useState } from 'react'

// The whole site reads one JSON blob (built from ../../results by
// scripts/compile-data.mjs). It is fetched once here and shared via context.

const StudyDataContext = createContext(null)

export function StudyDataProvider({ children }) {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  useEffect(() => {
    let alive = true
    fetch('/data/results.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
        return res.json()
      })
      .then((json) => alive && setState({ data: json, loading: false, error: null }))
      .catch((err) => alive && setState({ data: null, loading: false, error: err.message }))
    return () => {
      alive = false
    }
  }, [])

  return <StudyDataContext.Provider value={state}>{children}</StudyDataContext.Provider>
}

export function useStudyData() {
  return useContext(StudyDataContext)
}
