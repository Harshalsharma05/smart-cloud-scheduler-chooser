"""Plain dataclasses: Job, Node, InstanceType, CloudInstance, plus the config loader.
See IMPLEMENTATION.md, section 'Data model'."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Job states
PENDING, RUNNING, DONE = "PENDING", "RUNNING", "DONE"
# Cloud instance states
BOOTING, READY, RELEASED = "BOOTING", "READY", "RELEASED"


@dataclass
class Job:
    id: int
    arrival: int
    cores: int
    mem_gb: int
    gpus: int
    true_runtime: int            # engine only; schedulers must use est_runtime
    est_runtime: int
    deadline: int | None = None
    owner: str = ""
    name: str = ""
    # runtime fields
    state: str = PENDING
    start: int | None = None
    end: int | None = None
    node_id: str | None = None
    instance_id: str | None = None
    on_cloud: bool = False


@dataclass
class Node:
    id: str
    cls: str
    cores: int
    mem_gb: int
    gpus: int
    free_cores: int = 0
    free_mem: int = 0
    free_gpus: int = 0
    running: set = field(default_factory=set)   # job ids

    def __post_init__(self):
        self.free_cores, self.free_mem, self.free_gpus = self.cores, self.mem_gb, self.gpus


@dataclass
class InstanceType:
    name: str
    cores: int
    mem_gb: int
    gpus: int
    price_per_h: float


@dataclass
class CloudInstance:
    id: str
    type: InstanceType
    requested_at: int
    ready_at: int | None = None
    released_at: int | None = None
    job_id: int | None = None
    state: str = BOOTING


def load_config(path=None) -> dict:
    """Load the world settings (default: backend/configs/default.yaml)."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
