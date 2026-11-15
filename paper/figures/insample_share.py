"""W4 leakage check: share of running jobs and GPUs at the event starts (baseline engine) that belong to jobs submitted
before TRAIN_END (2017-11-01, in the learned models' training set). One baseline run, no policies.
Writes results/robust_insample/share.csv.   python paper/figures/insample_share.py   (about 1 min)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from flex.engine import Engine  # noqa: E402
from flex.models import load_config  # noqa: E402
from flex.paired import baseline_snapshots  # noqa: E402
from traces.philly import load_philly, ml_config, utc_epoch  # noqa: E402


def main():
    cfg = load_config()
    L = load_philly(cfg)
    mc = ml_config()
    cut_s = utc_epoch(mc.TRAIN_END) - utc_epoch(mc.SIM_START)          # simulated seconds of TRAIN_END
    starts = sorted({e.t_start for e in L["events"]})
    eng = Engine(L["jobs"], L["G"], check_every=cfg["check_every"])
    rows = []
    for t_s, snap in baseline_snapshots(eng, starts):
        run = [snap.jobs[i] for i in snap.running]
        old = [j for j in run if (j.submit if j.submit_real is None else j.submit_real) < cut_s]
        rows.append(dict(t_start=t_s, running_jobs=len(run), running_gpus=sum(j.gpus for j in run),
                         insample_jobs=len(old), insample_gpus=sum(j.gpus for j in old)))
    d = pd.DataFrame(rows)
    d["insample_jobs_pct"] = 100 * d.insample_jobs / d.running_jobs
    d["insample_gpus_pct"] = 100 * d.insample_gpus / d.running_gpus
    out = ROOT / "results" / "robust_insample"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "share.csv", index=False)
    print(d.round(1).to_string(index=False))
    print("median over the", len(d), "event start times: jobs", round(float(np.median(d.insample_jobs_pct)), 1),
          "% GPUs", round(float(np.median(d.insample_gpus_pct)), 1), "%")


if __name__ == "__main__":
    main()
