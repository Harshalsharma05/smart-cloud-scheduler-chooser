"""Dataset switch for the Track 1 pipeline: "philly" (default, the paper's data) or "acme" (Seren only, A2).

The same scripts run on both; only the file names, split bounds, window and feature set differ. Philly behaviour is unchanged.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def utc_epoch(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


@dataclass(frozen=True)
class Dataset:
    name: str
    prefix: str          # file-name prefix for examples, predictions and runs ("" for Philly)
    jobs_file: str
    sim_start: str
    test_end: str
    bounds: dict         # submit_time split bounds (epoch seconds)
    has_attempts: bool   # Philly has attempts and run_s; Acme has neither
    use_type: bool       # Acme `type` as a categorical feature


PHILLY = Dataset("philly", "", "philly_jobs.parquet", config.SIM_START, config.TEST_END, {
    "train_end": utc_epoch(config.TRAIN_END), "val_start": utc_epoch(config.VAL_START), "val_end": utc_epoch(config.VAL_END),
    "test_start": utc_epoch(config.TEST_START), "test_end": utc_epoch(config.TEST_END)}, True, False)

ACME = Dataset("acme", "acme_", "acme_jobs.parquet", config.ACME_SIM_START, config.ACME_TEST_END, {
    "train_end": utc_epoch(config.ACME_TRAIN_END), "val_start": utc_epoch(config.ACME_VAL_START),
    "val_end": utc_epoch(config.ACME_VAL_END), "test_start": utc_epoch(config.ACME_TEST_START),
    "test_end": utc_epoch(config.ACME_TEST_END)}, False, True)


def get(name):
    return {"philly": PHILLY, "acme": ACME}[name]


def from_path(path):
    """Prediction files named acme_<name>.csv belong to Acme; everything else is Philly."""
    return ACME if Path(path).stem.startswith("acme_") else PHILLY


def load_jobs(ds, data_dir=None):
    """The jobs table the example row indices (hist_idx, job_row) refer to. Acme: Seren only, rows renumbered."""
    jobs = pd.read_parquet(Path(data_dir or config.DATA_PROCESSED) / ds.jobs_file)
    if ds.name == "acme":
        jobs = jobs[jobs["cluster"] == "seren"].reset_index(drop=True)
    return jobs


def load_attempts(ds, data_dir=None):
    if ds.has_attempts:
        return pd.read_parquet(Path(data_dir or config.DATA_PROCESSED) / "philly_attempts.parquet")
    return pd.DataFrame({"job_id": pd.Series([], dtype=str), "start_time": pd.Series([], dtype="float64")})


def examples_path(ds, data_dir=None):
    return Path(data_dir or config.DATA_PROCESSED) / f"{ds.prefix}ml_examples.parquet"


def pred_path(ds, name):
    return config.PREDICTIONS / f"{ds.prefix}{name}.csv"
