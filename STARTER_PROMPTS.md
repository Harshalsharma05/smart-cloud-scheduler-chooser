# Starter prompts for Claude Code

Open Claude Code in the `smart-scheduler-chooser` folder. Paste the **first prompt** once, then use the **next-phase prompt** each time a phase is done and you have checked it.

---

## 1. First prompt (paste once)

```
Read CLAUDE.md and IMPLEMENTATION.md in this folder fully before doing anything.

Context: this is my academic minor project, Smart Scheduler Chooser. I am building it phase by phase
for a progress-check demo. IMPLEMENTATION.md is the source of truth. The ../v0.3/ folder is background
only; do not build its full scope.

Rules:
- Do only Phase 0 now. Do not start Phase 1.
- Keep it simple: this is an academic prototype, not an industrial system. No extra features,
  no extra abstractions, no extra tests beyond what the phase's check asks for.
- I am on Windows. Give me any commands I need to run myself.
- If anything in the plan is unclear or looks wrong, ask me before changing the design.

When Phase 0 is done:
1. Run the phase's check.
2. Tell me in a few lines what you built, which files changed, and how I can verify it myself.
3. Stop and wait for me.
```

---

## 2. Next-phase prompt (reuse for Phases 1 to 7)

Replace `N` with the phase number.

```
Phase N-1 looks good. Now do Phase N from IMPLEMENTATION.md, and only Phase N.

Same rules as before: keep it simple, follow the data model, rule sets and API contract in
IMPLEMENTATION.md exactly, add only the tests that Phase N's check asks for.

When done: run the check, commit with message "Phase N: <short summary>", tell me in a few lines
what you built and how to verify it, then stop.
```

---

## 3. Helper prompts (use when needed)

**Something broke:**
```
The check for Phase N fails with this error: <paste error>.
Find the cause and fix it with the smallest change. Do not refactor other code.
```

**The four rule sets give nearly the same results (Phase 5):**
```
In Phase 5 the rule sets give almost identical results. Tune only the scenario parameters in
workloads/scenarios.py (arrival rate, burst size, GPU share, deadline share, slack) until:
B5 has the lowest wait but highest cloud cost, B2/B3 have zero cloud cost, and B4 has the fewest
late jobs in deadline_heavy. Show me the before/after table. Do not change the simulator logic.
```

**Before the progress check:**
```
Read the "Progress-check demo script" in IMPLEMENTATION.md. Check that every step works end to end
(backend running, frontend running, each scenario runs). List anything that fails and fix it.
Do not add new features.
```

**Stretch (only after Phase 7 works):**
```
Phase 7 is done. Now do Phase 8 (simple switching chooser) from IMPLEMENTATION.md, and only Phase 8.
Same rules as before.
```

---

## Phase checklist

| Phase | What | Done? |
|---|---|---|
| 0 | Setup: venv, git, Vite app | ☐ |
| 1 | Models + workload generator | ☐ |
| 2 | Engine + cluster + safety checker | ☐ |
| 3 | Cloud pool | ☐ |
| 4 | Rule sets B2–B5 | ☐ |
| 5 | Metrics + runner (backend demo) | ☐ |
| 6 | FastAPI | ☐ |
| 7 | React page (visible demo) | ☐ |
| 8 | Stretch: switching chooser | ☐ |
| 9 | Stretch: save/replay + oracle | ☐ |

Tips:
- Review each phase's check yourself before moving on; bugs in Phases 2–4 are the hardest to find later.
- Start a fresh Claude Code session if a conversation gets long; the first prompt brings it back up to speed.
