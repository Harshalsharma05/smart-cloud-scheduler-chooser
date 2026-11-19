"""T3a: load a checkpoint and write predictions for a split in the contract CSV format.

    python ml/predict.py --run tf_v1 --split val --out runs/tf_v1/val_predictions.csv
Writes predictions only; no metrics.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataset as D  # noqa: E402
import model as M  # noqa: E402
from export_simple import COLS  # noqa: E402


@torch.no_grad()
def predict_frame(net, ex, jobs, vc_map, device, bs=2048, feat=None):
    """ex: example rows (any split) -> DataFrame with the contract columns."""
    d = M.build_tensors(ex, jobs, vc_map, device, feat)
    net.eval()
    qs, ps = [], []
    for i in range(0, len(ex), bs):
        idx = torch.arange(i, min(i + bs, len(ex)), device=device)
        q, logits = net(M.get_batch(d, idx))
        qs.append(q.cpu())
        ps.append(torch.softmax(logits, dim=1).cpu())
    q, p = torch.cat(qs).numpy(), torch.cat(ps).numpy()
    rt = np.maximum(config.MIN_RUNTIME_S, np.expm1(q))  # quantiles are already ordered; the floor keeps the order
    return pd.DataFrame({"job_id": ex["job_id"].to_numpy(), "elapsed_s": ex["elapsed_s"].to_numpy(),
                         "rt_p10_s": rt[:, 0], "rt_p50_s": rt[:, 1], "rt_p90_s": rt[:, 2],
                         "p_fail": p[:, 1], "p_cancel": p[:, 2]})[COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(config.DATA_PROCESSED))
    ap.add_argument("--runs", default=str(config.RUNS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.runs) / args.run / args.ckpt, map_location=device)
    feat = ckpt.get("feat", M.PHILLY_FEAT)
    net = M.make_net(ckpt["n_vc"], feat).to(device)
    net.load_state_dict(ckpt["state_dict"])
    ds = D.get(args.dataset)
    ex = pd.read_parquet(D.examples_path(ds, args.data))
    jobs = D.load_jobs(ds, args.data)
    ex = ex[ex["split"] == args.split]
    df = predict_frame(net, ex, jobs, ckpt["vc_map"], device, feat=feat)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["job_id", "elapsed_s"]).to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows ({args.split}) to {args.out}")


if __name__ == "__main__":
    main()
