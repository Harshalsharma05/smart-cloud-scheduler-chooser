# Proof-of-work demo

`demo.ipynb` re-runs small parts of the project live and compares them with the stored results the paper uses.
It only reads data, the model checkpoint, prediction files, results and paper files; it never changes them.
`demo.html` is the same notebook with its outputs, for reading without Python.

| Section | What it shows |
|---|---|
| 1. Data | Philly job log: total jobs, Pass/Failed/Killed split, test-window count, jobs and GPU-hours by status |
| 2. Transformer live | `runs/tf_v1/best.pt` predicts 5 test examples (seed from `ml/config.py`), next to the true outcome and the stored `last2` / `lightgbm` rows; T4 test metrics |
| 3. Simulator live | first 2 h / 20% event of E1: baseline, delay-only, burst-only, mix-p90 (transformer, hold-beyond-grid) from one snapshot; power plot; table compared with `results/e1/runs.csv` (stops if they differ) |
| 4. Checks | backend and ml test suites, `ml/check_predictions.py` on all five files, Table I baseline check |
| 5. Reproducibility | first lines of `paper/numbers.tex` and the commands that regenerate it |

## What it needs (not in git)

`data/processed/*.parquet`, `predictions/*.csv`, `runs/tf_v1/best.pt`, `results/e1/runs.csv`,
`results/philly_baseline/validation.csv`, and both environments: `ml/.venv` (runs the notebook) and `backend/.venv` (runs the
backend tests).

## Run it (Windows, from the repo root)

One-time: add the notebook tools to the ML environment (it already has torch, pandas and lightgbm):

    ml\.venv\Scripts\python -m pip install -r demo\requirements.txt

Run top to bottom, save the outputs in place, and export the HTML (about 1.5 min on the GTX 1650 laptop; the model runs on the CPU):

    ml\.venv\Scripts\python -m nbconvert --to notebook --execute --inplace demo\demo.ipynb
    ml\.venv\Scripts\python -m nbconvert --to html demo\demo.ipynb

Or open it interactively: `ml\.venv\Scripts\python -m pip install notebook`, then
`ml\.venv\Scripts\python -m notebook demo\demo.ipynb` and pick *Run All*.
