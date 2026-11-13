"""T3a test: shapes, non-crossing quantiles, and the loss falls when overfitting 256 examples (CPU)."""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import model as M  # noqa: E402
from tests.test_t2_leakage import BOUNDS, make_jobs  # noqa: E402
import examples  # noqa: E402


def test_model_shapes_and_training():
    torch.manual_seed(0)
    jobs, attempts = make_jobs(n=300)
    ex = examples.build_examples(jobs, attempts, bounds=BOUNDS)
    ex = ex[ex["split"].notna()].head(256).reset_index(drop=True)
    jobs = jobs.assign(wall_s=jobs["wall_s"], run_s=jobs["run_s"])
    d = M.build_tensors(ex, jobs, M.vc_index(jobs), torch.device("cpu"))
    net = M.HistoryTransformer(n_vc=len(M.vc_index(jobs)))

    idx = torch.arange(len(ex))
    q, logits = net(M.get_batch(d, idx))
    assert q.shape == (256, 3) and logits.shape == (256, 3)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    first = None
    for step in range(200):
        net.train()
        q, logits = net(M.get_batch(d, idx))
        loss, _, _ = M.loss_fn(q, logits, d["y_rt"], d["y_status"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.6 * first, (first, loss.item())
    net.eval()
    q, _ = net(M.get_batch(d, idx))
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
