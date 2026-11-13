"""Job record and config loader for the paper engine (flex)."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Job states. PENDING = not submitted yet.
PENDING, QUEUED, RUNNING, SUSPENDED, CLOUD_BOOTING, CLOUD_RUNNING, DONE = (
    "PENDING", "QUEUED", "RUNNING", "SUSPENDED", "CLOUD_BOOTING", "CLOUD_RUNNING", "DONE")


@dataclass
class FJob:
    """One job. Ground truth (wall_s, status) is for the engine and the oracle predictor only."""
    job_id: str
    user: str
    submit: int
    gpus: int
    wall_s: int
    status: str                 # "Pass" | "Killed" | "Failed"
    util: float = 0.5           # mean GPU utilisation, 0..1
    deadline: int | None = None
    warmup: bool = False
    carry: bool = False         # S8b: already running at the simulation start (pseudo-job submitted at t = 0)
    elapsed0: int = 0           # carry-over: seconds already run before t = 0 (added to work_done for the prediction lookup)
    submit_real: int | None = None   # carry-over: trace submit time relative to t = 0 (negative); used only for tie-breaks
    # runtime fields
    state: str = PENDING
    work_done: float = 0.0      # nominal seconds of progress (valid as of `mark`)
    extra_work: int = 0         # overheads added to the remaining work (S2)
    speed: float = 1.0
    version: int = 0            # bumped when a scheduled finish becomes stale
    mark: int = 0               # time of the last state change / work_done update
    n_starts: int = 0           # first starts from QUEUED (must stay <= 1)
    n_suspends: int = 0
    n_resumes: int = 0
    burst_rem: float | None = None   # remaining work (nominal s) when the job was burst; for cloud-waste metrics
    first_start: int | None = None
    end: int | None = None
    on_cloud: bool = False
    segments: list = field(default_factory=list)   # (start, end, where, speed)


def load_config(path=None) -> dict:
    """Read configs/flex.yaml (default location: backend/configs/flex.yaml)."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "flex.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
