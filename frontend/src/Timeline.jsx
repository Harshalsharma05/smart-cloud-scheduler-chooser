import { useEffect, useMemo, useRef, useState } from 'react'

const DEFAULT_SPAN = 6 * 3600
const MIN_SPAN = 5 * 60
const MAX_JOBS = 300
const GUTTER = 92 // left label column (px)
const RIGHT = 12
const AXIS_H = 28
const SECTION_H = 24
const LANE = 10
const GAP = 3
const OVERVIEW_H = 52
const BUCKET = 15 * 60 // overview density bucket
const TICK_STEPS = [300, 600, 900, 1800, 3600, 7200, 10800, 21600]

const clock = (t) => `${Math.floor(t / 3600)}:${String(Math.floor((t % 3600) / 60)).padStart(2, '0')}`
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi)

function dur(s) {
  s = Math.round(s)
  if (s < 60) return `${s} s`
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h) return m ? `${h} h ${m} m` : `${h} h`
  return r && m < 10 ? `${m} m ${r} s` : `${m} m`
}

/** Greedy lane packing on the clipped [s, e) interval: first lane free at the item's start. */
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
  const wrapRef = useRef(null)
  const svgRef = useRef(null)
  const [width, setWidth] = useState(900)
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(360, Math.floor(e.contentRect.width))))
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [])

  const domainEnd = useMemo(
    () => Math.max(...timeline.jobs.map((j) => j.end), ...timeline.cloud.map((c) => c.released), 3600),
    [timeline],
  )
  const fit = (t0, span, instant = false) => {
    span = clamp(span, Math.min(MIN_SPAN, domainEnd), domainEnd)
    t0 = clamp(t0, 0, domainEnd - span)
    return { t0, t1: t0 + span, instant }
  }

  // `view` is the target window; `shown` eases toward it and is what gets drawn.
  const [view, setView] = useState(() => fit(0, DEFAULT_SPAN))
  const [history, setHistory] = useState([])
  const [shown, setShown] = useState(view)
  const shownRef = useRef(view)
  useEffect(() => {
    const from = shownRef.current
    const ms = view.instant ? 0 : 240
    const start = performance.now()
    const frame = { id: 0 }
    const step = (now) => {
      const k = ms ? Math.min(1, (now - start) / ms) : 1
      const e = 1 - (1 - k) ** 3
      const v = { t0: from.t0 + (view.t0 - from.t0) * e, t1: from.t1 + (view.t1 - from.t1) * e }
      shownRef.current = v
      setShown(v)
      if (k < 1) frame.id = requestAnimationFrame(step)
    }
    frame.id = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame.id)
  }, [view])

  const [hidden, setHidden] = useState(() => new Set()) // 'cluster' | 'cloud' | 'boot' | 'billed'
  const [lateFocus, setLateFocus] = useState(false)
  const [hover, setHover] = useState(null) // { kind: 'job' | 'inst', item, cx, cy }
  const [cursor, setCursor] = useState(null) // px
  const [drag, setDrag] = useState(null) // { a, b } px

  const toggle = (key) => setHidden((h) => {
    const n = new Set(h)
    if (n.has(key)) n.delete(key)
    else n.add(key)
    return n
  })

  // ----- what is in view (packed against the target view so lanes don't jump while animating)
  const { t0, t1 } = view
  const inView = timeline.jobs
    .filter((j) => j.start < t1 && j.end > t0)
    .sort((a, b) => a.start - b.start || a.id - b.id)
  const drawable = inView.filter((j) => !hidden.has(j.on_cloud ? 'cloud' : 'cluster'))
  const jobs = pack(drawable.slice(0, MAX_JOBS).map((j) => ({ ...j, s: Math.max(j.start, t0), e: Math.min(j.end, t1) })))
  const instInView = timeline.cloud.filter((c) => c.requested < t1 && c.released > t0)
  const insts = hidden.has('boot') && hidden.has('billed')
    ? []
    : pack(instInView.map((c) => ({ ...c, s: Math.max(c.requested, t0), e: Math.min(c.released, t1) })))
  const counts = {
    cluster: inView.filter((j) => !j.on_cloud).length,
    cloud: inView.filter((j) => j.on_cloud).length,
    late: inView.filter((j) => j.late).length,
    inst: instInView.length,
  }

  // ----- geometry
  const plotW = width - GUTTER - RIGHT
  const span = shown.t1 - shown.t0
  const x = (t) => GUTTER + ((t - shown.t0) / span) * plotW
  const tAt = (px) => shown.t0 + ((px - GUTTER) / plotW) * span
  const jobsTop = AXIS_H + SECTION_H
  const jobLanes = Math.max(1, laneCount(jobs))
  const instTop = jobsTop + jobLanes * (LANE + GAP) + SECTION_H + 10
  const instLanes = Math.max(1, laneCount(insts))
  const height = instTop + instLanes * (LANE + GAP) + 10
  const laneY = (top, lane) => top + lane * (LANE + GAP)

  const step = TICK_STEPS.find((s) => (s / span) * plotW >= 80) ?? TICK_STEPS.at(-1)
  const ticks = []
  for (let t = Math.ceil(shown.t0 / (step / 2)) * (step / 2); t <= shown.t1; t += step / 2) ticks.push(t)

  // ----- linking: hovering a cloud job highlights its instance and vice versa
  const hotJob = hover?.kind === 'job' ? hover.item.id : hover?.kind === 'inst' ? hover.item.job_id : null
  const hotInst = hover?.kind === 'inst' ? hover.item.id : hover?.kind === 'job' && hover.item.on_cloud ? hover.item.where : null

  // ----- interactions
  const localX = (e) => e.clientX - svgRef.current.getBoundingClientRect().left
  const inPlot = (px) => px >= GUTTER && px <= GUTTER + plotW
  function showTip(e, kind, item) {
    if (drag) return
    const r = wrapRef.current.getBoundingClientRect()
    setHover({ kind, item, cx: e.clientX - r.left, cy: e.clientY - r.top })
  }
  function zoomTo(a, b) {
    setHistory((h) => [...h, view])
    setView(fit(a, b - a))
  }
  function back() {
    setView({ ...history.at(-1), instant: false })
    setHistory((h) => h.slice(0, -1))
  }
  function reset() {
    setHistory([])
    setView(fit((view.t0 + view.t1) / 2 - DEFAULT_SPAN / 2, DEFAULT_SPAN))
  }

  const svgHandlers = {
    onPointerDown: (e) => {
      if (e.button !== 0) return
      const px = clamp(localX(e), GUTTER, GUTTER + plotW)
      setHover(null)
      setDrag({ a: px, b: px })
      e.currentTarget.setPointerCapture(e.pointerId)
    },
    onPointerMove: (e) => {
      const px = localX(e)
      setCursor(inPlot(px) ? px : null)
      if (drag) setDrag({ ...drag, b: clamp(px, GUTTER, GUTTER + plotW) })
    },
    onPointerUp: () => {
      if (drag && Math.abs(drag.b - drag.a) > 6) {
        const a = tAt(Math.min(drag.a, drag.b))
        const b = tAt(Math.max(drag.a, drag.b))
        zoomTo(a, b)
      }
      setDrag(null)
    },
    onPointerLeave: () => {
      setCursor(null)
      setHover(null)
    },
  }

  const zoomed = history.length > 0

  return (
    <section className="timeline" ref={wrapRef}>
      <div className="tl-head">
        <h3>Timeline: {timeline.ruleset}</h3>
        <div className="tl-tools">
          <span className="tl-range">{clock(view.t0)} – {clock(view.t1)} <span className="hint">({dur(view.t1 - view.t0)})</span></span>
          <button onClick={back} disabled={!zoomed}>← Back</button>
          <button onClick={reset} disabled={!zoomed && view.t1 - view.t0 === Math.min(DEFAULT_SPAN, domainEnd)}>Reset zoom</button>
        </div>
      </div>

      <div className="legend">
        <LegendChip cls="sw-cluster" label="Cluster jobs" count={counts.cluster} off={hidden.has('cluster')} onClick={() => toggle('cluster')} />
        <LegendChip cls="sw-cloud" label="Cloud jobs" count={counts.cloud} off={hidden.has('cloud')} onClick={() => toggle('cloud')} />
        <LegendChip cls="sw-late" label={lateFocus ? 'Late (highlighted)' : 'Late'} count={counts.late} active={lateFocus} onClick={() => setLateFocus(!lateFocus)} />
        <LegendChip cls="sw-boot" label="Instance booting" count={counts.inst} off={hidden.has('boot')} onClick={() => toggle('boot')} />
        <LegendChip cls="sw-billed" label="Instance billed" off={hidden.has('billed')} onClick={() => toggle('billed')} />
        <span className="hint tl-help">Click a chip to show/hide · drag on the chart to zoom · hover a bar for details</span>
      </div>

      <Overview timeline={timeline} domainEnd={domainEnd} shown={shown} width={width}
        onPan={(start) => setView(fit(start, view.t1 - view.t0, true))} />

      {drawable.length > MAX_JOBS && (
        <div className="tl-cap">Showing the first {MAX_JOBS} of {drawable.length} jobs in view. Drag to zoom in to see the rest.</div>
      )}

      <svg ref={svgRef} width={width} height={height} {...svgHandlers}
        className={`tl-svg${hover ? ' has-hover' : ''}${lateFocus ? ' late-focus' : ''}`}>
        <defs>
          <clipPath id="tl-plot"><rect x={GUTTER} y={0} width={plotW} height={height} /></clipPath>
        </defs>

        {/* lane stripes */}
        {Array.from({ length: jobLanes }, (_, i) => i % 2 === 0 && (
          <rect key={`sj${i}`} x={GUTTER} y={laneY(jobsTop, i) - GAP / 2} width={plotW} height={LANE + GAP} className="tl-stripe" />
        ))}
        {Array.from({ length: instLanes }, (_, i) => i % 2 === 0 && (
          <rect key={`si${i}`} x={GUTTER} y={laneY(instTop, i) - GAP / 2} width={plotW} height={LANE + GAP} className="tl-stripe" />
        ))}

        {/* axis + grid */}
        <g clipPath="url(#tl-plot)">
          {ticks.map((t) => {
            const major = Math.round(t) % step === 0
            return (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={AXIS_H - 6} y2={height} className={major ? 'tl-grid' : 'tl-grid-minor'} />
                {major && <text x={x(t) + 3} y={AXIS_H - 10} className="tl-axis">{step >= 3600 ? `${t / 3600} h` : clock(t)}</text>}
              </g>
            )
          })}
        </g>

        {/* section labels */}
        <text x={8} y={jobsTop - 8} className="tl-section">Jobs</text>
        <text x={8} y={jobsTop + 6} className="tl-section-sub">packed rows</text>
        <line x1={0} x2={width - RIGHT} y1={instTop - SECTION_H + 2} y2={instTop - SECTION_H + 2} className="tl-sep" />
        <text x={8} y={instTop - 8} className="tl-section">Cloud</text>
        <text x={8} y={instTop + 6} className="tl-section-sub">instances</text>

        <g clipPath="url(#tl-plot)">
          {/* jobs */}
          {jobs.map((j) => {
            const cls = ['bar', j.on_cloud ? 'tl-cloud-job' : 'tl-cluster-job']
            if (j.late) cls.push('tl-late')
            if (j.id === hotJob) cls.push('hovered')
            return (
              <rect key={j.id} x={x(j.s)} y={laneY(jobsTop, j.lane)} height={LANE} rx={2}
                width={Math.max(1.5, x(j.e) - x(j.s))} className={cls.join(' ')}
                onPointerMove={(e) => showTip(e, 'job', j)} onPointerLeave={() => setHover(null)} />
            )
          })}
          {/* deadline markers: late jobs, plus the hovered job */}
          {jobs.filter((j) => j.deadline != null && (j.late || j.id === hotJob)).map((j) => (
            <line key={`d${j.id}`} x1={x(j.deadline)} x2={x(j.deadline)}
              y1={laneY(jobsTop, j.lane) - 2} y2={laneY(jobsTop, j.lane) + LANE + 2}
              className={j.late ? 'tl-deadline' : 'tl-deadline ok'} />
          ))}

          {/* cloud instances: booting then billed */}
          {insts.map((c) => {
            const hot = c.id === hotInst ? ' hovered' : ''
            const y = laneY(instTop, c.lane)
            return (
              <g key={c.id} onPointerMove={(e) => showTip(e, 'inst', c)} onPointerLeave={() => setHover(null)}>
                {!hidden.has('boot') && (
                  <rect x={x(c.s)} y={y} height={LANE} rx={2} width={Math.max(1.5, x(c.ready) - x(c.s))} className={`bar tl-boot${hot}`} />
                )}
                {!hidden.has('billed') && c.released > c.ready && (
                  <rect x={x(Math.max(c.ready, c.s))} y={y} height={LANE} rx={2}
                    width={Math.max(1.5, x(c.e) - x(Math.max(c.ready, c.s)))} className={`bar tl-billed${hot}`} />
                )}
              </g>
            )
          })}
          {insts.length === 0 && (
            <text x={GUTTER + plotW / 2} y={instTop + 8} className="tl-axis" textAnchor="middle">no cloud instances in this window</text>
          )}

          {/* drag selection */}
          {drag && Math.abs(drag.b - drag.a) > 2 && (
            <g>
              <rect x={Math.min(drag.a, drag.b)} y={AXIS_H - 6} width={Math.abs(drag.b - drag.a)} height={height - AXIS_H + 6} className="tl-select" />
              <text x={Math.min(drag.a, drag.b) + 4} y={AXIS_H + 8} className="tl-select-label">
                {clock(tAt(Math.min(drag.a, drag.b)))} – {clock(tAt(Math.max(drag.a, drag.b)))}
              </text>
            </g>
          )}

          {/* crosshair */}
          {cursor != null && !drag && (
            <g pointerEvents="none">
              <line x1={cursor} x2={cursor} y1={AXIS_H - 6} y2={height} className="tl-cross" />
              <rect x={cursor - 22} y={2} width={44} height={15} rx={3} className="tl-cross-box" />
              <text x={cursor} y={13} textAnchor="middle" className="tl-cross-text">{clock(tAt(cursor))}</text>
            </g>
          )}
        </g>
      </svg>

      {hover && <Tooltip hover={hover} wrapWidth={width} />}
    </section>
  )
}

function LegendChip({ cls, label, count, off, active, onClick }) {
  return (
    <button className={`chip${off ? ' off' : ''}${active ? ' active' : ''}`} onClick={onClick} title={off ? 'Show' : 'Hide / highlight'}>
      <i className={`sw ${cls}`} />
      {label}
      {count != null && <span className="chip-count">{count}</span>}
    </button>
  )
}

function Tooltip({ hover, wrapWidth }) {
  const { kind, item: d, cx, cy } = hover
  const left = cx > wrapWidth - 280 ? cx - 270 : cx + 16
  return (
    <div className="tl-tip" style={{ left, top: cy + 16 }}>
      {kind === 'job' ? (
        <>
          <div className="tip-title">
            Job #{d.id}
            <span className={`badge ${d.late ? 'late' : d.deadline != null ? 'ok' : ''}`}>
              {d.late ? 'LATE' : d.deadline != null ? 'on time' : 'no deadline'}
            </span>
          </div>
          <Row k="Ran on" v={`${d.where} (${d.on_cloud ? 'cloud' : 'cluster'})`} />
          <Row k="Size" v={`${d.cores} cores · ${d.mem_gb} GB${d.gpus ? ` · ${d.gpus} GPU` : ''}`} />
          <Row k="Arrived" v={`${clock(d.arrival)} · waited ${dur(d.start - d.arrival)}`} />
          <Row k="Ran" v={`${clock(d.start)} → ${clock(d.end)} (${dur(d.end - d.start)})`} />
          <Row k="Estimate" v={dur(d.est_runtime)} />
          {d.deadline != null && (
            <Row k="Deadline" v={`${clock(d.deadline)} · ${d.late ? 'missed by' : 'met with'} ${dur(Math.abs(d.deadline - d.end))}${d.late ? '' : ' to spare'}`} />
          )}
        </>
      ) : (
        <>
          <div className="tip-title">Instance {d.id}<span className="badge">{d.type}</span></div>
          <Row k="For job" v={`#${d.job_id}`} />
          <Row k="Requested" v={clock(d.requested)} />
          <Row k="Ready" v={`${clock(d.ready)} (boot ${dur(d.ready - d.requested)})`} />
          <Row k="Released" v={`${clock(d.released)} · billed ${dur(d.released - d.ready)}`} />
          <Row k="Cost" v={`${d.cost} units (${d.price_per_h}/h)`} />
        </>
      )}
    </div>
  )
}

const Row = ({ k, v }) => <div className="tip-row"><span>{k}</span><span>{v}</span></div>

/** Whole-run strip: running jobs per 15 min (cluster / cloud stacked); drag or click to pan the view. */
function Overview({ timeline, domainEnd, shown, width, onPan }) {
  const svgRef = useRef(null)
  const grab = useRef(null)
  const buckets = useMemo(() => {
    const n = Math.ceil(domainEnd / BUCKET)
    const cl = new Array(n).fill(0)
    const cd = new Array(n).fill(0)
    for (const j of timeline.jobs) {
      const arr = j.on_cloud ? cd : cl
      for (let b = Math.floor(j.start / BUCKET); b <= Math.floor((j.end - 1) / BUCKET) && b < n; b++) arr[b]++
    }
    return { cl, cd, max: Math.max(1, ...cl.map((v, i) => v + cd[i])) }
  }, [timeline, domainEnd])

  const W = width - GUTTER - RIGHT
  const barH = OVERVIEW_H - 16
  const ox = (t) => GUTTER + (t / domainEnd) * W
  const tAt = (px) => ((px - GUTTER) / W) * domainEnd
  const bw = Math.max(1, (BUCKET / domainEnd) * W - 0.5)
  const hourStep = [1, 2, 3, 4, 6, 12].find((h) => (h * 3600 / domainEnd) * W >= 50) ?? 12
  const hours = []
  for (let h = 0; h * 3600 <= domainEnd; h += hourStep) hours.push(h)
  const span = shown.t1 - shown.t0

  const handlers = {
    onPointerDown: (e) => {
      const t = tAt(e.clientX - svgRef.current.getBoundingClientRect().left)
      grab.current = t >= shown.t0 && t <= shown.t1 ? t - shown.t0 : span / 2
      onPan(t - grab.current)
      e.currentTarget.setPointerCapture(e.pointerId)
    },
    onPointerMove: (e) => {
      if (grab.current == null) return
      onPan(tAt(e.clientX - svgRef.current.getBoundingClientRect().left) - grab.current)
    },
    onPointerUp: () => { grab.current = null },
  }

  return (
    <svg ref={svgRef} width={width} height={OVERVIEW_H} className="tl-overview" {...handlers}>
      <text x={8} y={16} className="tl-section">Overview</text>
      <text x={8} y={30} className="tl-section-sub">drag to pan</text>
      <rect x={GUTTER} y={2} width={W} height={barH} className="ov-bg" rx={3} />
      {buckets.cl.map((v, i) => {
        const hc = (v / buckets.max) * (barH - 4)
        const hd = (buckets.cd[i] / buckets.max) * (barH - 4)
        const bx = ox(i * BUCKET)
        return (
          <g key={i}>
            {v > 0 && <rect x={bx} y={barH + 1 - hc} width={bw} height={hc} className="ov-cluster" />}
            {hd > 0 && <rect x={bx} y={barH + 1 - hc - hd} width={bw} height={hd} className="ov-cloud" />}
          </g>
        )
      })}
      {hours.map((h) => (
        <text key={h} x={ox(h * 3600)} y={OVERVIEW_H - 3} className="tl-axis" textAnchor={h === 0 ? 'start' : 'middle'}>{h} h</text>
      ))}
      {/* dim outside the current view, outline the view */}
      <rect x={GUTTER} y={2} width={Math.max(0, ox(shown.t0) - GUTTER)} height={barH} className="ov-dim" />
      <rect x={ox(shown.t1)} y={2} width={Math.max(0, GUTTER + W - ox(shown.t1))} height={barH} className="ov-dim" />
      <rect x={ox(shown.t0)} y={1} width={Math.max(3, ox(shown.t1) - ox(shown.t0))} height={barH + 2} rx={3} className="ov-window" />
    </svg>
  )
}
