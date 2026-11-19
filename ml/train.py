"""T3a: train the history transformer. Uses the train split; the val split only for early stopping.

    python ml/train.py --name smoke --max-rows 20000 --epochs 2
    python ml/train.py --name tf_v1                      (fp32; add --amp only on Kaggle)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataset as D  # noqa: E402
import model as M  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate(net, d, bs, amp):
    """Mean (total, pinball, ce) over a tensor dict."""
    net.eval()
    n = len(d["cur_num"])
    sums = torch.zeros(3, device=d["cur_num"].device)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=d["cur_num"].device)
        with torch.autocast(device_type=d["cur_num"].device.type, dtype=torch.float16, enabled=amp):
            q, logits = net(M.get_batch(d, idx))
        tot, pin, ce = M.loss_fn(q.float(), logits.float(), d["y_rt"][idx], d["y_status"][idx])
        sums += torch.stack([tot, pin, ce]) * len(idx)
    return (sums / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: subsample train to N rows (val to N/4)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (Kaggle T4 only)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--data", default=str(config.DATA_PROCESSED))
    ap.add_argument("--out", default=str(config.RUNS))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = D.get(args.dataset)
    ex = pd.read_parquet(D.examples_path(ds, args.data))
    jobs = D.load_jobs(ds, args.data)
    vc_map = M.vc_index(jobs)
    feat = M.feat_for(ds, jobs)
    tr, va = ex[ex["split"] == "train"], ex[ex["split"] == "val"]
    if args.max_rows:
        tr = tr.sample(n=min(args.max_rows, len(tr)), random_state=args.seed)
        va = va.sample(n=min(max(args.max_rows // 4, 1), len(va)), random_state=args.seed)
    dtr, dva = M.build_tensors(tr, jobs, vc_map, device, feat), M.build_tensors(va, jobs, vc_map, device, feat)

    net = M.make_net(len(vc_map), feat).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp = args.amp and device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device {gpu} | train rows {len(tr):,} val rows {len(va):,} | params {n_params:,} | amp {amp}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    best, bad, history, epoch_times = float("inf"), 0, [], []
    n = len(tr)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n, generator=gen).to(device)
        run = torch.zeros((), device=device)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                q, logits = net(M.get_batch(dtr, idx))
            loss, _, _ = M.loss_fn(q.float(), logits.float(), dtr["y_rt"][idx], dtr["y_status"][idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.detach() * len(idx)
        train_loss = run.item() / n
        v_tot, v_pin, v_ce = evaluate(net, dva, 2048, amp)
        epoch_times.append(time.time() - t0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v_tot, "val_pinball": v_pin, "val_ce": v_ce,
                        "seconds": epoch_times[-1]})
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {v_tot:.4f} (pinball {v_pin:.4f}, ce {v_ce:.4f}) | {epoch_times[-1]:.1f}s", flush=True)

        ckpt = {"state_dict": net.state_dict(), "n_vc": len(vc_map), "vc_map": vc_map, "feat": feat, "epoch": epoch, "val_loss": v_tot}
        torch.save(ckpt, run_dir / f"epoch_{epoch:02d}.pt")
        if v_tot < best:
            best, bad = v_tot, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad += 1
        summary = {"name": args.name, "config": vars(args) | {"H": config.H, "status_loss_weight": config.STATUS_LOSS_WEIGHT},
                   "params": n_params, "epochs_run": epoch, "best_val_loss": best,
                   "seconds_per_epoch": float(np.mean(epoch_times)), "gpu": gpu,
                   "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else 0,
                   "train_rows": len(tr), "val_rows": len(va), "git_commit": git_commit(), "history": history}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if bad >= args.patience:
            print(f"early stopping (no val improvement for {args.patience} epochs)")
            break
    print(f"best val loss {best:.4f} | {summary['seconds_per_epoch']:.1f} s/epoch | peak GPU mem {summary['peak_gpu_mem_mb']} MB")


if __name__ == "__main__":
    main()
