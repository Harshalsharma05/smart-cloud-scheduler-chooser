"""Cloud pool: instance catalogue, request -> booting -> ready -> busy -> released lifecycle, cap, billing."""
from sim.models import BOOTING, READY, RELEASED, CloudInstance, InstanceType


class CloudPool:
    """One job per instance. Billing per second from READY to RELEASED, with a minimum."""

    def __init__(self, cfg):
        c = cfg["cloud"]
        self.types = [InstanceType(**t) for t in c["instance_types"]]
        self.boot_delay = c["boot_delay_s"]
        self.max_instances = c["max_instances"]
        self.min_billed = c["min_billed_s"]
        self.instances = {}          # id -> CloudInstance, in request order
        self.n_live = 0              # BOOTING or READY

    def cheapest_fit(self, job):
        fitting = [t for t in self.types
                   if t.cores >= job.cores and t.mem_gb >= job.mem_gb and t.gpus >= job.gpus]
        return min(fitting, key=lambda t: t.price_per_h, default=None)

    def request(self, job, now):
        """Rent the cheapest fitting type for this job. None if at the cap or nothing fits."""
        itype = self.cheapest_fit(job)
        if itype is None or self.n_live >= self.max_instances:
            return None
        inst = CloudInstance(id=f"c{len(self.instances) + 1}", type=itype, requested_at=now, job_id=job.id)
        self.instances[inst.id] = inst
        self.n_live += 1
        return inst

    def mark_ready(self, inst, now):
        inst.state, inst.ready_at = READY, now

    def release(self, inst, now):
        inst.state, inst.released_at = RELEASED, now
        self.n_live -= 1

    def billed_seconds(self, inst, now):
        if inst.ready_at is None:
            return 0
        end = inst.released_at if inst.released_at is not None else now
        return max(self.min_billed, end - inst.ready_at)

    def cost(self, inst, now):
        return self.billed_seconds(inst, now) * inst.type.price_per_h / 3600

    def total_cost(self, now):
        return sum(self.cost(i, now) for i in self.instances.values())
