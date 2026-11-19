"""T4: evaluate every predictor.

    python ml/evaluate.py

Headline table: TEST split only. Second table: the export rows split by train (in-sample), val (early stopping)
and test (clean) for the learned models. Predictions are built in memory; nothing is tuned here.
Writes ml/results/predictor_metrics.csv, predictor_metrics.md and predictor_metrics_by_split.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

STATUSES = ["Pass", "Failed", "Killed"]
QS = (0.1, 0.5, 0.9)
EPS = 1e-15
EXTRA = ["mae_log", "auroc_cancel"]  # added for Acme (A2); Philly's T4 tables do not have them
METRICS = ["n", "pinball", "mae_p50_min", "coverage_p10_p90", "logloss", "auroc_failed", "brier_pfail"]


def target_rt(remaining_s):
    """Remaining runtime as exported: floored at 60 s (the oracle then matches exactly)."""
    return np.maximum(config.MIN_RUNTIME_S, np.asarray(remaining_s, dtype="float64"))


def pinball_log(y, p10, p50, p90):
    """Mean over the 3 quantiles of the pinball loss in log1p space (seconds in, seconds out)."""
    ly = np.log1p(y)
    tot = 0.0
    for tau, q in zip(QS, (p10, p50, p90)):
        e = ly - np.log1p(q)
        tot += np.mean(np.maximum(tau * e, (tau - 1) * e))
    return float(tot / 3)


def mae_p50_min(y, p50):
    return float(np.median(np.abs(np.asarray(p50) - y)) / 60)


def coverage(y, p10, p90):
    return float(np.mean((y >= np.asarray(p10) - 1e-9) & (y <= np.asarray(p90) + 1e-9)))


def logloss3(status, p_fail, p_cancel):
    """3-class log-loss; probabilities [1 - pf - pc, pf, pc] clipped to [EPS, 1]."""
    p = np.column_stack([1 - np.asarray(p_fail) - np.asarray(p_cancel), p_fail, p_cancel])
    p = np.clip(p, EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(p)), status]))) + 0.0


def auroc_failed(status, p_fail):
    y = (np.asarray(status) == 1).astype(int)
    return float(roc_auc_score(y, p_fail)) if 0 < y.sum() < len(y) else float("nan")


def brier_pfail(status, p_fail):
    return float(np.mean((np.asarray(p_fail) - (np.asarray(status) == 1)) ** 2))


def mae_log(y, p50):
    """Mean absolute error of the p50 in log1p space."""
    return float(np.mean(np.abs(np.log1p(np.asarray(p50)) - np.log1p(y))))


def auroc_cancel(status, p_cancel):
    y = (np.asarray(status) == 2).astype(int)
    return float(roc_auc_score(y, p_cancel)) if 0 < y.sum() < len(y) else float("nan")


def all_metrics(remaining_s, status, pred, runtime=True, extra=False):
    """remaining_s, status (0/1/2 ids): arrays; pred: DataFrame with the contract columns. Runtime metrics skipped if not runtime."""
    y = target_rt(remaining_s)
    out = {"n": len(y)}
    if runtime:
        out["pinball"] = pinball_log(y, pred["rt_p10_s"].to_numpy(), pred["rt_p50_s"].to_numpy(), pred["rt_p90_s"].to_numpy())
        out["mae_p50_min"] = mae_p50_min(y, pred["rt_p50_s"].to_numpy())
        out["coverage_p10_p90"] = coverage(y, pred["rt_p10_s"].to_numpy(), pred["rt_p90_s"].to_numpy())
    else:
        out.update(pinball=np.nan, mae_p50_min=np.nan, coverage_p10_p90=np.nan)
    out["logloss"] = logloss3(status, pred["p_fail"].to_numpy(), pred["p_cancel"].to_numpy())
    out["auroc_failed"] = auroc_failed(status, pred["p_fail"].to_numpy())
    out["brier_pfail"] = brier_pfail(status, pred["p_fail"].to_numpy())
    if extra:
        out["mae_log"] = mae_log(y, pred["rt_p50_s"].to_numpy()) if runtime else np.nan
        out["auroc_cancel"] = auroc_cancel(status, pred["p_cancel"].to_numpy())
    return out


def prior_frame(train, target):
    """Status prior: class frequencies of the train rows at each grid point, applied to `target` rows."""
    st = train["status"].map({s: i for i, s in enumerate(STATUSES)})
    freq = pd.crosstab(train["elapsed_s"], st, normalize="index").reindex(columns=[0, 1, 2], fill_value=0.0)
    f = freq.reindex(target["elapsed_s"].to_numpy()).to_numpy()
    return pd.DataFrame({"p_fail": f[:, 1], "p_cancel": f[:, 2]})


def table(rows):
    return pd.DataFrame(rows)


def md_table(df, floats=3):
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else (f"{v:,.0f}" if c == "n" else f"{v:.{floats}f}"))
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def build_predictions(ex_export, jobs, ex_all, ds=None):
    """name -> contract frame aligned with ex_export rows, for the 5 predictors (+ Acme's no-type lightgbm ablation)."""
    import torch

    import baselines as B
    import dataset as D
    import export_simple as S
    import model as M
    import predict as P

    ds = ds or D.PHILLY
    vc_map = {v: i for i, v in enumerate(sorted(jobs["vc"].unique()))}
    preds = {"oracle": S.oracle_frame(ex_export)}
    fw, ff, fc = S.train_fallbacks(ex_all)
    preds["last2"] = S.last2_frame(ex_export, fw, ff, fc)
    names = ["gbdt", "lightgbm"] + (["lightgbm_notype"] if ds.name == "acme" else [])
    for name in names:
        run = ds.prefix + name
        feats, types = B.load_spec(run)
        preds[name] = B.predict_frame(run, B.load_models(run), ex_export, vc_map, feats, types)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(config.RUNS / (ds.prefix + "tf_v1") / "best.pt", map_location=device)
    feat = ckpt.get("feat", M.PHILLY_FEAT)
    net = M.make_net(ckpt["n_vc"], feat).to(device)
    net.load_state_dict(ckpt["state_dict"])
    preds["transformer"] = P.predict_frame(net, ex_export, jobs, ckpt["vc_map"], device, feat=feat)
    for k, df in preds.items():
        assert (df["job_id"].to_numpy() == ex_export["job_id"].to_numpy()).all(), k
    return preds


def main():
    import argparse

    import dataset as D
    import export_simple as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ds = D.get(ap.parse_args().dataset)
    acme = ds.name == "acme"
    ex = pd.read_parquet(D.examples_path(ds))
    jobs = D.load_jobs(ds)
    exp = S.export_keys(jobs, ex, ds).reset_index(drop=True)
    print(f"export rows {len(exp):,}: {exp['split'].value_counts().to_dict()}")
    preds = build_predictions(exp, jobs, ex, ds)
    status = exp["status"].map({s: i for i, s in enumerate(STATUSES)}).to_numpy()
    rem = exp["remaining_s"].to_numpy()
    train = ex[ex["split"] == "train"]
    prior = prior_frame(train, exp)
    order = ["last2", "gbdt", "lightgbm", "transformer", "oracle", "prior"] + (["lightgbm_notype"] if acme else [])
    frames = {**preds, "prior": prior}

    def run(mask, name):
        return all_metrics(rem[mask], status[mask], frames[name][mask] if name != "prior" else prior[mask], runtime=name != "prior",
                           extra=acme)

    # headline: test split, overall and per grid point
    is_test = (exp["split"] == "test").to_numpy()
    rows = []
    for name in order:
        rows.append({"predictor": name, "elapsed_s": "all", **run(is_test, name)})
        for e in config.GRID:
            m = is_test & (exp["elapsed_s"] == e).to_numpy()
            if m.any():
                rows.append({"predictor": name, "elapsed_s": str(e), **run(m, name)})
    head = table(rows)

    # per split on the export rows, learned models (+ references)
    rows2 = []
    for name in [n for n in order if n not in ("oracle",)]:
        for sp in ("train", "val", "test"):
            m = (exp["split"] == sp).to_numpy()
            if m.any():
                rows2.append({"predictor": name, "split": sp, **run(m, name)})
    by_split = table(rows2)

    out = config.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    pre = ds.prefix
    head.to_csv(out / f"{pre}predictor_metrics.csv", index=False)
    by_split.to_csv(out / f"{pre}predictor_metrics_by_split.csv", index=False)
    overall = head[head["elapsed_s"] == "all"].drop(columns="elapsed_s")
    window = ("Acme Seren, test split only (submitted 2023-06-29 to 2023-07-29, China time)" if acme
              else "Test split only (submitted 2017-11-15 to 2017-12-15)")
    L = [f"# Predictor metrics ({'A2, Acme' if acme else 'T4'})", "",
         f"{window}. Runtime metrics are on remaining runtime floored at 60 s; "
         "pinball is the mean over q10/q50/q90 in log1p space; coverage target is 0.80. `prior` is the train-split class "
         "frequency per grid point (status metrics only). `oracle` uses ground truth. Nothing was tuned on test.", "",
         "## Overall (test)", "", md_table(overall), "",
         "## Per grid point (test)", ""]
    for name in order:
        L += [f"### {name}", "", md_table(head[(head["predictor"] == name) & (head["elapsed_s"] != "all")].drop(columns="predictor")), ""]
    L += ["## Export rows by split (train = in-sample, val = used for early stopping, test = clean)", "",
          "The train and val rows flatter the learned models; only test is a fair estimate. `last2` and `prior` use no trained "
          "parameters beyond the train-split fallbacks.", "", md_table(by_split), ""]
    (out / f"{pre}predictor_metrics.md").write_text("\n".join(L), encoding="utf-8")
    print(md_table(overall))
    print()
    print(md_table(by_split))


if __name__ == "__main__":
    main()
