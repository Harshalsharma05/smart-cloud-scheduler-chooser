import { useState } from 'react'

const WINDOW_H = 6
const MAX_JOBS = 300
const W = 1000 // SVG user units
const LANE = 7
const GAP = 1
const HEAD = 22 // space for the hour axis and section titles

/** Greedy lane packing: each item goes in the first lane that is free at its start. */
function pack(items) {
  const laneEnds = []
  return items.map((it) => {
    let lane = laneEnds.findIndex((end) => end <= it.s)
    if (lane === -1) {
      lane = laneEnds.length
      laneEnds.push(0)
    }
    laneEnds[lane] = it.e
    return { ...it, lane }
  })
}

const laneCount = (packed) => packed.reduce((m, p) => Math.max(m, p.lane + 1), 0)

export default function Timeline({ timeline }) {
  const lastH = Math.ceil(Math.max(...timeline.jobs.map((j) => j.end)) / 3600)
  const [fromH, setFromH] = useState(0)
  const t0 = fromH * 3600
  const t1 = (fromH + WINDOW_H) * 3600
  const x = (t) => ((Math.min(Math.max(t, t0), t1) - t0) / (t1 - t0)) * W

  // Jobs that start inside the window (capped), clipped to the window for drawing.
  const starting = timeline.jobs.filter((j) => j.start >= t0 && j.start < t1).sort((a, b) => a.start - b.start)
  const jobs = pack(starting.slice(0, MAX_JOBS).map((j) => ({ ...j, s: j.start, e: Math.min(j.end, t1) })))
  // Cloud instances alive at any point in the window.
  const insts = pack(
    timeline.cloud
      .filter((c) => c.requested < t1 && c.released > t0)
      .map((c) => ({ ...c, s: Math.max(c.requested, t0), e: Math.min(c.released, t1) })),
  )

  const jobsTop = HEAD
  const jobsH = Math.max(1, laneCount(jobs)) * (LANE + GAP)
  const instTop = jobsTop + jobsH + HEAD
  const instH = Math.max(1, laneCount(insts)) * (LANE + GAP)
  const height = instTop + instH + 4
  const hours = Array.from({ length: WINDOW_H + 1 }, (_, i) => fromH + i)

  return (
    <section className="timeline">
      <h3>Timeline: {timeline.ruleset}</h3>
      <div className="tl-controls">
        <label>
          Window {fromH} h – {fromH + WINDOW_H} h
          <input type="range" min="0" max={Math.max(0, lastH - WINDOW_H)} value={fromH}
            onChange={(e) => setFromH(Number(e.target.value))} />
        </label>
        <span className="hint">
          {Math.min(starting.length, MAX_JOBS)} of {starting.length} jobs starting in this window
          {starting.length > MAX_JOBS && ' (first 300 shown)'} · {insts.length} cloud instances
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${height}`} width="100%">
        {hours.map((h) => (
          <g key={h}>
            <line x1={x(h * 3600)} x2={x(h * 3600)} y1={12} y2={height} className="tl-grid" />
            <text x={x(h * 3600) + 2} y={10} className="tl-axis">{h} h</text>
          </g>
        ))}
        <text x={2} y={jobsTop - 2} className="tl-title">Jobs (lanes are packed, not nodes)</text>
        {jobs.map((j) => (
          <rect key={j.id} x={x(j.s)} y={jobsTop + j.lane * (LANE + GAP)} height={LANE}
            width={Math.max(1, x(j.e) - x(j.s))}
            className={`${j.on_cloud ? 'tl-cloud-job' : 'tl-cluster-job'}${j.late ? ' tl-late' : ''}`}>
            <title>{`job ${j.id} on ${j.where}\n${(j.start / 3600).toFixed(2)} h → ${(j.end / 3600).toFixed(2)} h${j.late ? '\nLATE' : ''}`}</title>
          </rect>
        ))}

        <text x={2} y={instTop - 2} className="tl-title">Cloud instances</text>
        {insts.map((c) => (
          <g key={c.id}>
            <title>{`${c.id} (${c.type})\nrequested ${(c.requested / 3600).toFixed(2)} h, ready ${(c.ready / 3600).toFixed(2)} h, released ${(c.released / 3600).toFixed(2)} h`}</title>
            <rect x={x(c.s)} y={instTop + c.lane * (LANE + GAP)} height={LANE}
              width={Math.max(1, x(c.ready) - x(c.s))} className="tl-boot" />
            <rect x={x(c.ready)} y={instTop + c.lane * (LANE + GAP)} height={LANE}
              width={Math.max(0, x(c.e) - x(c.ready))} className="tl-billed" />
          </g>
        ))}
        {insts.length === 0 && <text x={W / 2} y={instTop + 6} className="tl-axis" textAnchor="middle">no cloud instances in this window</text>}
      </svg>

      <div className="legend">
        <span><i className="sw tl-cluster-job" /> job on cluster</span>
        <span><i className="sw tl-cloud-job" /> job on cloud</span>
        <span><i className="sw tl-late-sw" /> late (missed deadline)</span>
        <span><i className="sw tl-boot" /> instance booting</span>
        <span><i className="sw tl-billed" /> instance billed</span>
      </div>
    </section>
  )
}
