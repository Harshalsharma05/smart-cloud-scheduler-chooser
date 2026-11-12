"""Shared constants for Track 1 (ML_PLAN section 2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PREDICTIONS = ROOT / "predictions"
RUNS = ROOT / "runs"
REPORTS = ROOT / "ml" / "reports"
RESULTS = ROOT / "ml" / "results"

GRID = [0, 60, 600, 3600, 14400, 43200]  # elapsed seconds since first start
H = 32                                    # max history jobs per example
MIN_RUNTIME_S = 60                        # floor for exported remaining runtime

# Time split by submit_time; all bounds half-open: start <= submit_time < end
TRAIN_END = "2017-11-01"     # train: submitted before this
VAL_START = "2017-11-01"
VAL_END = "2017-11-15"       # val: 2017-11-01 to 2017-11-14
TEST_START = "2017-11-15"
TEST_END = "2017-12-16"      # test: 2017-11-15 to 2017-12-15

SEED = 0
STATUS_LOSS_WEIGHT = 1.0     # total loss = pinball + weight * CE
