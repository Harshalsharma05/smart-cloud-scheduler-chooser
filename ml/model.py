"""T3a: features -> tensors, the small history transformer, and its loss.

Input per example: up to H history tokens (the user's earlier finished jobs) and one [CUR] token
(the current job at its observation time). No user-ID embedding: the history is the user signal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

STATUSES = ["Pass", "Failed", "Killed"]  # class ids 0, 1, 2
QUANTILES = (0.1, 0.5, 0.9)
N_HIST_NUM = 7
N_CUR_NUM = 8


# ------------------------------------------------------------------ features
def vc_index(jobs):
    """vc string -> id (0..n-1) from the full jobs table (the same table for every split)."""
    return {v: i for i, v in enumerate(sorted(jobs["vc"].unique()))}


def build_tensors(ex, jobs, vc_map, device):
    """Turn the example table into device tensors. Everything is known at the observation time."""
    f = lambda c: jobs[c].astype("float64").fillna(0).clip(lower=0).to_numpy()
    wall, run, start, sub, end = f("wall_s"), f("run_s"), f("first_start_time"), f("submit_time"), f("end_time")
    qwait = np.where(start > 0, np.maximum(start - sub, 0), 0)
    jt = np.stack([np.log1p(wall) / 10, np.log1p(run) / 10, np.log1p(jobs["num_gpus"].to_numpy()) / 5,
                   np.log1p(jobs["num_attempts"].to_numpy()) / 2, np.log1p(qwait) / 10], axis=1)
    job_status = jobs["status"].map({s: i for i, s in enumerate(STATUSES)}).to_numpy()
    job_vc = jobs["vc"].map(vc_map).to_numpy()

    n, H = len(ex), config.H
    hist = np.full((n, H), -1, dtype=np.int32)
    for i, h in enumerate(ex["hist_idx"].to_numpy()):
        hist[i, :len(h)] = h
    hour, dow = ex["submit_hour"].to_numpy(), ex["submit_dow"].to_numpy()
    cur = np.stack([np.log1p(ex["num_gpus"].to_numpy()) / 5,
                    np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    np.log1p(ex["queue_wait_s"].to_numpy()) / 10, np.log1p(ex["elapsed_s"].to_numpy()) / 12,
                    np.minimum(ex["attempts_so_far"].to_numpy(), 5) / 5], axis=1)
    jr = ex["job_row"].to_numpy()
    t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    d = {
        "job_num": t(jt, torch.float32), "job_status": t(job_status, torch.long), "job_vc": t(job_vc, torch.long),
        "job_end": t(end, torch.float64),
        "hist": t(hist, torch.int32), "cur_num": t(cur, torch.float32), "cur_vc": t(job_vc[jr], torch.long),
        "t_obs": t(ex["t_obs"].to_numpy(), torch.float64),
    }
    if "remaining_s" in ex:  # targets (not present in prediction-only use)
        d["y_rt"] = t(np.log1p(ex["remaining_s"].to_numpy()), torch.float32)
        d["y_status"] = t(ex["status"].map({s: i for i, s in enumerate(STATUSES)}).to_numpy(), torch.long)
    return d


def get_batch(d, idx):
    """Gather one batch of model inputs for example indices `idx` (LongTensor on the same device)."""
    h = d["hist"][idx].long()
    pad = h < 0
    h = h.clamp(min=0)
    ago = torch.log1p((d["t_obs"][idx].unsqueeze(1) - d["job_end"][h]).clamp(min=0)).float() / 10
    same_vc = (d["job_vc"][h] == d["cur_vc"][idx].unsqueeze(1)).float()
    hist_num = torch.cat([d["job_num"][h], ago.unsqueeze(-1), same_vc.unsqueeze(-1)], dim=-1)
    return {"hist_num": hist_num, "hist_status": d["job_status"][h], "hist_vc": d["job_vc"][h], "pad": pad,
            "cur_num": d["cur_num"][idx], "cur_vc": d["cur_vc"][idx]}


# --------------------------------------------------------------------- model
class HistoryTransformer(nn.Module):
    def __init__(self, n_vc, d_model=64, heads=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.hist_proj = nn.Linear(N_HIST_NUM, d_model)
        self.cur_proj = nn.Linear(N_CUR_NUM, d_model)
        self.status_emb = nn.Embedding(len(STATUSES), d_model)
        self.vc_emb = nn.Embedding(n_vc, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0 = history, 1 = current
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.rt_head = nn.Linear(d_model, 3)
        self.status_head = nn.Linear(d_model, len(STATUSES))

    def forward(self, b):
        hist = self.hist_proj(b["hist_num"]) + self.status_emb(b["hist_status"]) + self.vc_emb(b["hist_vc"]) + self.type_emb.weight[0]
        cur = (self.cur_proj(b["cur_num"]) + self.vc_emb(b["cur_vc"]) + self.type_emb.weight[1]).unsqueeze(1)
        x = torch.cat([cur, hist], dim=1)  # [CUR] first
        mask = torch.cat([torch.zeros_like(b["pad"][:, :1]), b["pad"]], dim=1)  # True = padding
        h = self.encoder(x, src_key_padding_mask=mask)[:, 0]
        r = self.rt_head(h)
        q50 = r[:, 0]
        q = torch.stack([q50 - F.softplus(r[:, 1]), q50, q50 + F.softplus(r[:, 2])], dim=1)  # q10 <= q50 <= q90
        return q, self.status_head(h)


def loss_fn(q, logits, y_rt, y_status, status_weight=config.STATUS_LOSS_WEIGHT):
    """Pinball loss in log1p space (mean over the 3 quantiles) + weight * cross-entropy. Returns (total, pinball, ce)."""
    taus = torch.tensor(QUANTILES, device=q.device, dtype=q.dtype)
    err = y_rt.unsqueeze(1) - q
    pinball = torch.maximum(taus * err, (taus - 1) * err).mean()
    ce = F.cross_entropy(logits, y_status)
    return pinball + status_weight * ce, pinball, ce
