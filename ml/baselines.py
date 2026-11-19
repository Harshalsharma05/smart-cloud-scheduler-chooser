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
ACME_FEATS = [c for c in FEATS if c not in ("attempts_so_far", "vc_code")] + ["type_code"]  # Acme: no attempts, one vc
ACME_CAT = ["type_code"]
HGB_MAX_ITER, LGB_MAX_ROUNDS, LR, PATIENCE = 400, 2000, 0.1, 30


def feature_spec(ds, use_type=True):
    """(feature list, categorical list) for a dataset. Philly is the original spec; Acme drops attempts_so_far and the
    constant vc, and adds the workload type unless use_type is False (ablation)."""
    if ds is None or ds.name == "philly":
        return FEATS, CAT
    feats = [c for c in ACME_FEATS if use_type or c != "type_code"]
    return feats, [c for c in ACME_CAT if use_type]


def feature_table(ex, vc_map, feats=FEATS, types=None):
    """Flat features (float64) for a slice of ml_examples."""
    X = ex[[c for c in feats if c not in ("vc_code", "type_code")]].astype("float64").copy()
    if "vc_code" in feats:
        X["vc_code"] = ex["vc"].astype(str).map(vc_map).astype("float64").to_numpy()
    if "type_code" in feats:
        X["type_code"] = ex["type"].astype(str).map({v: i for i, v in enumerate(types)}).astype("float64").to_numpy()
    return X[feats].reset_index(drop=True)


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


def train_gbdt(Xtr, ytr, str_, Xv, yv, sv, seed=config.SEED, cat=CAT):
    """One HistGradientBoosting model per quantile + one classifier. Early stopping on val: fit with a fixed
    maximum, take the best staged iteration on val, refit with that many iterations."""
    common = dict(learning_rate=LR, max_leaf_nodes=31, min_samples_leaf=50, random_state=seed,
                  categorical_features=[list(Xtr.columns).index(c) for c in cat])
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
def train_lightgbm(Xtr, ytr, str_, Xv, yv, sv, seed=config.SEED, cat=CAT):
    """lgb.train with early stopping on the val set (one model per quantile + multiclass)."""
    base = dict(learning_rate=LR, num_leaves=31, min_data_in_leaf=50, seed=seed, verbose=-1, deterministic=True,
                num_threads=4)
    models, info = {}, {}
    cb = lambda: [lgb.early_stopping(PATIENCE, verbose=False)]
    for tau in QUANTILES:
        p = dict(base, objective="quantile", alpha=tau, metric="quantile")
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat)
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        m = lgb.train(p, dtr, LGB_MAX_ROUNDS, valid_sets=[dv], callbacks=cb())
        models[f"q{tau}"] = m
        info[f"q{tau}"] = {"best_iter": m.best_iteration, "val_pinball": pinball(yv, m.predict(Xv, num_iteration=m.best_iteration), tau)}
    p = dict(base, objective="multiclass", num_class=3, metric="multi_logloss")
    dtr = lgb.Dataset(Xtr, str_, categorical_feature=cat)
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
        if "lightgbm" not in name:
            joblib.dump(m, d / f"{k}.joblib")
        else:
            m.save_model(str(d / f"{k}.txt"))


def load_models(name, runs_dir=config.RUNS):
    d = Path(runs_dir) / name
    keys = [f"q{t}" for t in QUANTILES] + ["status"]
    if "lightgbm" not in name:
        return {k: joblib.load(d / f"{k}.joblib") for k in keys}
    return {k: lgb.Booster(model_file=str(d / f"{k}.txt")) for k in keys}


def load_spec(name, runs_dir=config.RUNS):
    """(features, types) a saved baseline was trained with (Philly summaries predate `types`)."""
    sm = json.loads((Path(runs_dir) / name / "summary.json").read_text())
    return sm["features"], sm.get("types")


def predict_frame(name, models, ex, vc_map, feats=FEATS, types=None):
    """Contract frame for a slice of ml_examples, from a trained baseline (`name` is gbdt or lightgbm, any run prefix)."""
    X = feature_table(ex, vc_map, feats, types)
    q, p = predict_gbdt(models, X) if "lightgbm" not in name else predict_lightgbm(models, X)
    return to_frame(ex, q, p)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    import dataset as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="debug: subsample train rows")
    ap.add_argument("--runs", default=str(config.RUNS))
    ap.add_argument("--only", choices=["gbdt", "lightgbm"], default=None)
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ap.add_argument("--no-type", action="store_true", help="Acme ablation: leave the workload type out (run suffix _notype)")
    args = ap.parse_args()
    ds = D.get(args.dataset)

    ex = pd.read_parquet(D.examples_path(ds))
    jobs = D.load_jobs(ds)
    vc_map = {v: i for i, v in enumerate(sorted(jobs["vc"].unique()))}
    feats, cat = feature_spec(ds, not args.no_type)
    types = sorted(jobs["type"].astype(str).unique()) if "type_code" in feats else None
    tr, va = ex[ex["split"] == "train"], ex[ex["split"] == "val"]  # test is never touched here
    if args.max_rows:
        tr = tr.sample(min(args.max_rows, len(tr)), random_state=config.SEED)
    Xtr, Xv = feature_table(tr, vc_map, feats, types), feature_table(va, vc_map, feats, types)
    (ytr, str_), (yv, sv) = targets(tr), targets(va)
    print(f"{ds.name}: train rows {len(tr):,}  val rows {len(va):,}  features {len(feats)}")

    for base, fit in [("gbdt", train_gbdt), ("lightgbm", train_lightgbm)]:
        if args.only and args.only != base:
            continue
        name = ds.prefix + base + ("_notype" if args.no_type else "")
        t0 = time.time()
        models, info = fit(Xtr, ytr, str_, Xv, yv, sv, cat=cat)
        secs = time.time() - t0
        save_models(name, models, args.runs)
        out = Path(args.runs) / name
        (out / "summary.json").write_text(json.dumps({
            "name": name, "seed": config.SEED, "features": feats, "categorical": cat, "types": types, "train_rows": len(tr),
            "val_rows": len(va), "learning_rate": LR, "seconds": secs, "val": info, "git_commit": git_commit()}, indent=2))
        print(name, f"{secs:.0f}s", json.dumps(info))


if __name__ == "__main__":
    main()
