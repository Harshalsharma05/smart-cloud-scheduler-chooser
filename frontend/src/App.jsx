import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Timeline from './Timeline.jsx'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// [key, label] for each results column; lower is better for all of them.
const COLUMNS = [
  ['late_pct', 'Late %'],
  ['avg_wait_min', 'Avg wait (min)'],
  ['avg_bounded_slowdown', 'Slowdown'],
  ['cloud_cost', 'Cloud cost'],
  ['cloud_instance_hours', 'Instance hours'],
  ['score', 'Score J'],
]
const CHARTS = [
  ['late_pct', 'Late deadline jobs (%)'],
  ['avg_wait_min', 'Average wait (min)'],
  ['cloud_cost', 'Cloud cost (units)'],
]

async function api(path, body) {
  let res
  try {
    res = await fetch(API + path, body && {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error(`Cannot reach the backend at ${API}. Is it running? (cd backend, then: uvicorn api.main:app --reload)`)
  }
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText)
    throw new Error(`Backend error ${res.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`)
  }
  return res.json()
}

export default function App() {
  const [meta, setMeta] = useState(null) // { scenarios, rulesets, profiles }
  const [form, setForm] = useState({ scenario: 'quiet', profile: 'balanced', seed: 1, timeline_for: 'B4' })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function loadMeta() {
    Promise.all([api('/scenarios'), api('/rulesets'), api('/profiles')])
      .then(([scenarios, rulesets, profiles]) => setMeta({ scenarios, rulesets, profiles }))
      .catch((e) => setError(e.message))
  }
  useEffect(loadMeta, [])

  async function run() {
    setLoading(true)
    setError('')
    try {
      const body = { ...form, seed: Number(form.seed), rulesets: meta.rulesets.map((r) => r.id) }
      setResult(await api('/run', body))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value })

  return (
    <main>
      <header>
        <h1>Smart Scheduler Chooser</h1>
        <p className="sub">Same jobs, different scheduling rule sets: who wins depends on the situation.</p>
      </header>

      {error && (
        <div className="error">
          {error} {!meta && <button onClick={() => { setError(''); loadMeta() }}>Retry</button>}
        </div>
      )}

      {meta && (
        <section className="controls">
          <label>Scenario
            <select value={form.scenario} onChange={set('scenario')}>
              {meta.scenarios.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </label>
          <label>Profile
            <select value={form.profile} onChange={set('profile')}>
              {meta.profiles.map((p) => (
                <option key={p.id} value={p.id}>{p.id} ({p.w_miss} / {p.w_slowdown} / {p.w_cost})</option>
              ))}
            </select>
          </label>
          <label>Seed
            <input type="number" min="1" value={form.seed} onChange={set('seed')} />
          </label>
          <label>Timeline for
            <select value={form.timeline_for} onChange={set('timeline_for')}>
              {meta.rulesets.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
            </select>
          </label>
          <button className="run" onClick={run} disabled={loading}>{loading ? 'Running…' : 'Run'}</button>
          <p className="hint">
            {meta.scenarios.find((s) => s.id === form.scenario)?.description}
            {' '}Profile weights are miss / slowdown / cost.
          </p>
        </section>
      )}

      {loading && <div className="loading">Simulating 24 hours under every rule set…</div>}

      {result && !loading && <Results result={result} rulesets={meta.rulesets} />}
    </main>
  )
}

function Results({ result, rulesets }) {
  const rows = result.results
  const best = Math.min(...rows.map((r) => r.score))
  const winners = rows.filter((r) => r.score === best).map((r) => r.ruleset)
  const bestOf = Object.fromEntries(COLUMNS.map(([k]) => [k, Math.min(...rows.map((r) => r[k]))]))
  const label = Object.fromEntries(rulesets.map((r) => [r.id, r.label]))

  return (
    <>
      <div className="banner">
        Best under <b>{result.profile}</b>: <b>{winners.join(' = ')}</b>
        {winners.length > 1 && ' (tie)'}
        <span className="banner-sub">{result.scenario}, seed {result.seed}, {result.n_jobs} jobs · lower J is better</span>
      </div>

      <table>
        <thead>
          <tr>
            <th>Rule set</th>
            {COLUMNS.map(([k, name]) => <th key={k}>{name}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.ruleset} className={winners.includes(r.ruleset) ? 'winner' : ''}>
              <td><b>{r.ruleset}</b> <span className="rs-label">{label[r.ruleset]}</span></td>
              {COLUMNS.map(([k]) => (
                <td key={k} className={r[k] === bestOf[k] ? 'best' : ''}>{r[k]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="charts">
        {CHARTS.map(([k, title]) => (
          <div className="chart" key={k}>
            <h3>{title}</h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={rows} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="ruleset" />
                <YAxis />
                <Tooltip />
                <Bar dataKey={k} fill="#4f6bd8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      <Timeline key={`${result.scenario}-${result.seed}-${result.timeline.ruleset}`} timeline={result.timeline} />
    </>
  )
}
