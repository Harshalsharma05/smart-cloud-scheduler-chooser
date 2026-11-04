"""Safety checker: asserts capacity is never exceeded, each job runs once, cloud is billed only while live, clock never goes back."""
from sim.models import RUNNING


class SafetyChecker:
    def __init__(self):
        self.last_time = 0
        self.started = set()      # job ids that have started (a job may start only once)

    def job_started(self, job):
        assert job.id not in self.started, f"job {job.id} started twice"
        self.started.add(job.id)

    def check(self, now, nodes, jobs_by_id):
        """Called after every event."""
        assert now >= self.last_time, "clock went backwards"
        self.last_time = now
        running_ids = set()
        for n in nodes:
            used_cores = sum(jobs_by_id[j].cores for j in n.running)
            used_mem = sum(jobs_by_id[j].mem_gb for j in n.running)
            used_gpus = sum(jobs_by_id[j].gpus for j in n.running)
            assert used_cores <= n.cores and used_mem <= n.mem_gb and used_gpus <= n.gpus, \
                f"node {n.id} over capacity"
            assert (n.free_cores, n.free_mem, n.free_gpus) == \
                (n.cores - used_cores, n.mem_gb - used_mem, n.gpus - used_gpus), f"node {n.id} bookkeeping"
            assert not (running_ids & n.running), "job on two nodes"
            running_ids |= n.running
        # cluster-side check only for now; cloud billing checks are added in Phase 3
        cluster_running = {j.id for j in jobs_by_id.values() if j.state == RUNNING and not j.on_cloud}
        assert running_ids == cluster_running, "running jobs and node state disagree"
