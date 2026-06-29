import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'

// Horizontal nav fallback (hidden on desktop; the SideRail is the primary nav).
const ITEMS = [
  { to: '/', label: 'Overview', exact: true },
  { to: '/methods', label: 'Methods' },
  { to: '/results', label: 'Q1 + Q5' },
  { to: '/q2-alternatives', label: 'Q2' },
  { to: '/q3-zeroshot', label: 'Q3' },
  { to: '/q6-ablations', label: 'Q6' },
  { to: '/case-studies', label: 'Q4' },
  { to: '/report', label: 'Report' },
  { to: '/evidence', label: 'Evidence' },
]

export default function TopNav() {
  const { pathname } = useLocation()
  return (
    <nav aria-label="Secondary navigation" className="site-nav">
      <div className="site-nav-inner">
        <NavLink to="/" className="site-nav-brand">
          TxGNN·KG
        </NavLink>
        <ul className="site-nav-links" role="list">
          {ITEMS.map(({ to, label, exact }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={exact}
                className="site-nav-link"
                aria-current={(exact ? pathname === to : pathname.startsWith(to)) ? 'page' : undefined}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}
