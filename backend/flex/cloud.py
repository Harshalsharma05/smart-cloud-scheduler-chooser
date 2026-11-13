"""Cloud GPU instances for the paper engine (D8): per-GPU price, boot delay, per-second billing, GPU cap.

Plain data plus functions. The engine decides when to request, start and release; this module keeps the books.
"""
from dataclasses import dataclass, field


@dataclass
class CloudParams:
    price_per_gpu_h: float
    boot_delay_s: int       # request -> READY
    migrate_s: int          # extra nominal seconds added when a RUNNING job is moved to the cloud
    min_billed_s: int       # minimum billed time per instance
    max_gpus: int           # cap on cloud GPUs in use (booting + running)


@dataclass
class Instance:
    job_id: str
    gpus: int
    requested: int
    ready: int | None = None
    released: int | None = None


@dataclass
class Cloud:
    params: CloudParams
    instances: list = field(default_factory=list)   # every instance ever requested (one per burst job)
    active: dict = field(default_factory=dict)      # job_id -> Instance, not yet released


def gpus_in_use(cloud):
    return sum(i.gpus for i in cloud.active.values())


def can_request(cloud, gpus):
    return gpus_in_use(cloud) + gpus <= cloud.params.max_gpus


def request(cloud, job_id, gpus, t):
    inst = Instance(job_id, gpus, t)
    cloud.instances.append(inst)
    cloud.active[job_id] = inst
    return inst


def set_ready(cloud, job_id, t):
    cloud.active[job_id].ready = t


def release(cloud, job_id, t):
    inst = cloud.active.pop(job_id)
    inst.released = t


def instance_cost(params, inst, now):
    """Bill from READY to RELEASED (or to `now` if still running), per second, with a minimum."""
    if inst.ready is None:
        return 0.0
    end = inst.released if inst.released is not None else now
    billed_s = max(params.min_billed_s, end - inst.ready)
    return inst.gpus * params.price_per_gpu_h * billed_s / 3600.0


def total_cost(cloud, now):
    return sum(instance_cost(cloud.params, i, now) for i in cloud.instances)
