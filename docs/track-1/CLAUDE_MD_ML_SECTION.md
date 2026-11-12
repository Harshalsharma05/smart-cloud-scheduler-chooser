# Text to add to CLAUDE.md (Track 1)

Paste the block below into `CLAUDE.md` as a new section. The Track 2 / main chat will rewrite the "What this project is / Current goal" header for the paper work. Until then, the first line of the block tells Claude Code which plan wins inside `ml/`.

---

```markdown
## ML track (folder `ml/`): Track 1 of the CCGrid 2027 paper

**Source of truth for anything in `ml/`: `docs/track-1/ML_PLAN.md`.** It overrides IMPLEMENTATION.md for ML work. Prompts are in `docs/track-1/STARTER_PROMPTS_ML.md`.

### How to work
- One phase from ML_PLAN.md at a time (T0, T1, T2, T2b, T3a, T3b, T4, T5). At the end: run the phase's done-check, commit, print the report in the format from STARTER_PROMPTS_ML.md, and stop.
- If the data or the plan forces a choice the plan doesn't cover, stop and ask. Don't redesign.
- Don't touch `backend/`, `frontend/` or `paper/` from an ML phase.

### Interface contract (do not change silently)
- Track 1 writes `data/processed/philly_jobs.parquet`, `data/processed/philly_attempts.parquet` and `predictions/<name>.csv` (`job_id, elapsed_s, rt_p10_s, rt_p50_s, rt_p90_s, p_fail, p_cancel`; names: oracle, last2, gbdt, lightgbm, transformer).
- `rt_*` = remaining runtime in seconds at `elapsed_s`, floored at 60. Elapsed grid: 0, 60, 600, 3600, 14400, 43200 s. Rows only while the job runs.
- Every prediction file must pass `python ml/check_predictions.py <file>`.
- Only `oracle` may use ground truth.

### Leakage rules (never relax them)
- Observation time: `t_obs = submit_time` at elapsed 0, else `first_start_time + elapsed`.
- History = the same user's jobs with `end_time <= t_obs`, at most 32, never the job itself.
- Time split from `ml/config.py` (train < val < test by submit time). No random split. No test metrics before T4; tune on val only.
- No user-ID embedding (test users can be new).

### Code rules
- Python 3.11 in `ml/.venv` (separate from `backend/.venv`). Run scripts from the repo root: `python ml/<script>.py`.
- Plain functions and dataclasses; plain PyTorch (no Lightning, Hydra, W&B, MLflow). Constants live in `ml/config.py`.
- GTX 1650: fp32 only. `--amp` (mixed precision) only on Kaggle T4.
- Same seed + same data → same outputs. Every training run writes `runs/<name>/summary.json`.
- Tests: only those listed in ML_PLAN.md (about 6). Never commit data, checkpoints or prediction CSVs (git-ignored).

### Commands
    py -3.11 -m venv ml\.venv && ml\.venv\Scripts\activate
    pip install -r ml/requirements.txt          # torch: CUDA wheel matching nvidia-smi
    python ml/philly_clean.py                   # T1
    python ml/examples.py                       # T2
    python ml/export_simple.py                  # T2b
    python ml/check_predictions.py predictions/last2.csv
    python ml/train.py --name tf_v1             # T3 (add --amp only on Kaggle)
    python ml/evaluate.py                       # T4
    pytest ml/tests -q
```
