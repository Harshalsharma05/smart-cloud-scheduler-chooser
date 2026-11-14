"""T4: gradient-boosted-tree baselines on one flat feature table.

    python ml/baselines.py            # trains gbdt (sklearn) and lightgbm, early stopping on val only

Both models predict log1p(remaining_s) at quantiles 0.1 / 0.5 / 0.9 plus a 3-class status
(Pass / Failed / Killed), from the current-job features and the aggregated history features
already in ml_examples (nothing new, nothing from the future).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from export_simple import COLS  # noqa: E402

STATUSES = ["Pass", "Failed", "Killed"]
QUANTILES = (0.1, 0.5, 0.9)
FEATS = ["elapsed_s", "queue_wait_s", "attempts_so_far", "num_gpus", "submit_hour", "submit_dow", "vc_code",
         "hist_count", "hist_mean_log_wall", "hist_median_log_wall", "hist_last1_wall", "hist_last2_wall",
         "hist_fail_rate", "hist_cancel_rate"]
CAT = ["vc_code"]
HGB_MAX_ITER, LGB_MAX_ROUNDS, LR, PATIENCE = 400, 2000, 0.1, 30


def feature_table(ex, vc_map):
    """Flat features (float64) for a slice of ml_examples."""
    X = ex[[c for c in FEATS if c != "vc_code"]].astype("float64").copy()
    X["vc_code"] = ex["vc"].astype(str).map(vc_map).astype("float64").to_numpy()
    return X[FEATS].reset_index(drop=True)


def targets(ex):
    y = np.log1p(ex["remaining_s"].to_numpy().astype("float64"))
    s = ex["status"].map({s: i for i, s in enumerate(STATUSES)}).to_numpy()
    return y, s


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def to_frame(ex, q, p):
    """q: (n,3) log1p quantiles, p: (n,3) class probabilities -> contract frame (sorted quantiles, floor 60 s)."""
    q = np.sort(q, axis=1)
    rt = np.maximum(config.MIN_RUNTIME_S, np.expm1(q))
    p = np.clip(p, 0.0, 1.0)
    pf, pc = p[:, 1], p[:, 2]
    tot = pf + pc
    scale = np.where(tot > 1, 1 / np.where(tot > 0, tot, 1), 1.0)
    return pd.DataFrame({"job_id": ex["job_id"].to_numpy(), "elapsed_s": ex["elapsed_s"].to_numpy(),
                         "rt_p10_s": rt[:, 0], "rt_p50_s": rt[:, 1], "rt_p90_s": rt[:, 2],
                         "p_fail": pf * scale, "p_cancel": pc * scale})[COLS]


# ---------------------------------------------------------------- sklearn gbdt
def _best_iter(model, Xv, yv, tau=None, status=False):
    """Pick the number of iterations with the best val score from staged predictions."""
    if status:
        scores = [log_loss(yv, p, labels=[0, 1, 2]) for p in model.staged_predict_proba(Xv)]
    else:
        scores = [pinball(yv, q, tau) for q in model.staged_predict(Xv)]
    return int(np.argmin(scores)) + 1, float(np.min(scores))


def train_gbdt(Xtr, ytr, str_, Xv, yv, sv, seed=config.SEED):
    """One HistGradientBoosting model per quantile + one classifier. Early stopping on val: fit with a fixed
    maximum, take the best staged iteration on val, refit with that many iterations."""
    common = dict(learning_rate=LR, max_leaf_nodes=31, min_samples_leaf=50, random_state=seed,
                  categorical_features=[FEATS.index(c) for c in CAT])
    models, info = {}, {}
    for tau in QUANTILES:
        mk = lambda n: HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=n, early_stopping=False, **common)
        n, sc = _best_iter(mk(HGB_MAX_ITER).fit(Xtr, ytr), Xv, yv, tau=tau)
        models[f"q{tau}"] = mk(n).fit(Xtr, ytr)
        info[f"q{tau}"] = {"best_iter": n, "val_pinball": sc}
    mk = lambda n: HistGradientBoostingClassifier(max_iter=n, early_stopping=False, **common)
    n, sc = _best_iter(mk(HGB_MAX_ITER).fit(Xtr, str_), Xv, sv, status=True)
    models["status"] = mk(n).fit(Xtr, str_)
    info["status"] = {"best_iter": n, "val_logloss": sc}
    return models, info


def predict_gbdt(models, X):
    q = np.column_stack([models[f"q{t}"].predict(X) for t in QUANTILES])
    return q, models["status"].predict_proba(X)


# --------------------------------------------------------------------- lightgbm
def train_lightgbm(Xtr, ytr, str_, Xv, yv, sv, seed=config.SEED):
    """lgb.train with early stopping on the val set (one model per quantile + multiclass)."""
    base = dict(learning_rate=LR, num_leaves=31, min_data_in_leaf=50, seed=seed, verbose=-1, deterministic=True,
                num_threads=4)
    models, info = {}, {}
    cb = lambda: [lgb.early_stopping(PATIENCE, verbose=False)]
    for tau in QUANTILES:
        p = dict(base, objective="quantile", alpha=tau, metric="quantile")
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CAT)
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        m = lgb.train(p, dtr, LGB_MAX_ROUNDS, valid_sets=[dv], callbacks=cb())
        models[f"q{tau}"] = m
        info[f"q{tau}"] = {"best_iter": m.best_iteration, "val_pinball": pinball(yv, m.predict(Xv, num_iteration=m.best_iteration), tau)}
    p = dict(base, objective="multiclass", num_class=3, metric="multi_logloss")
    dtr = lgb.Dataset(Xtr, str_, categorical_feature=CAT)
    dv = lgb.Dataset(Xv, sv, reference=dtr)
    m = lgb.train(p, dtr, LGB_MAX_ROUNDS, valid_sets=[dv], callbacks=cb())
    models["status"] = m
    info["status"] = {"best_iter": m.best_iteration, "val_logloss": float(log_loss(sv, m.predict(Xv, num_iteration=m.best_iteration), labels=[0, 1, 2]))}
    return models, info


def predict_lightgbm(models, X):
    q = np.column_stack([models[f"q{t}"].predict(X, num_iteration=models[f"q{t}"].best_iteration) for t in QUANTILES])
    return q, models["status"].predict(X, num_iteration=models["status"].best_iteration)


# ---------------------------------------------------------------- save / load
def save_models(name, models, out_dir):
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    for k, m in models.items():
        if name == "gbdt":
            joblib.dump(m, d / f"{k}.joblib")
        else:
            m.save_model(str(d / f"{k}.txt"))


def load_models(name, runs_dir=config.RUNS):
    d = Path(runs_dir) / name
    keys = [f"q{t}" for t in QUANTILES] + ["status"]
    if name == "gbdt":
        return {k: joblib.load(d / f"{k}.joblib") for k in keys}
    return {k: lgb.Booster(model_file=str(d / f"{k}.txt")) for k in keys}


def predict_frame(name, models, ex, vc_map):
    """Contract frame for a slice of ml_examples, from a trained baseline."""
    X = feature_table(ex, vc_map)
    q, p = predict_gbdt(models, X) if name == "gbdt" else predict_lightgbm(models, X)
    return to_frame(ex, q, p)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="debug: subsample train rows")
    ap.add_argument("--runs", default=str(config.RUNS))
    ap.add_argument("--only", choices=["gbdt", "lightgbm"], default=None)
    args = ap.parse_args()

    ex = pd.read_parquet(config.DATA_PROCESSED / "ml_examples.parquet")
    jobs = pd.read_parquet(config.DATA_PROCESSED / "philly_jobs.parquet")
    vc_map = {v: i for i, v in enumerate(sorted(jobs["vc"].unique()))}
    tr, va = ex[ex["split"] == "train"], ex[ex["split"] == "val"]  # test is never touched here
    if args.max_rows:
        tr = tr.sample(min(args.max_rows, len(tr)), random_state=config.SEED)
    Xtr, Xv = feature_table(tr, vc_map), feature_table(va, vc_map)
    (ytr, str_), (yv, sv) = targets(tr), targets(va)
    print(f"train rows {len(tr):,}  val rows {len(va):,}  features {len(FEATS)}")

    for name, fit in [("gbdt", train_gbdt), ("lightgbm", train_lightgbm)]:
        if args.only and args.only != name:
            continue
        t0 = time.time()
        models, info = fit(Xtr, ytr, str_, Xv, yv, sv)
        secs = time.time() - t0
        save_models(name, models, args.runs)
        out = Path(args.runs) / name
        (out / "summary.json").write_text(json.dumps({
            "name": name, "seed": config.SEED, "features": FEATS, "train_rows": len(tr), "val_rows": len(va),
            "learning_rate": LR, "seconds": secs, "val": info, "git_commit": git_commit()}, indent=2))
        print(name, f"{secs:.0f}s", json.dumps(info))


if __name__ == "__main__":
    main()
