"""Safety checker: asserts capacity is never exceeded, each job runs once, cloud is billed only while live, clock never goes back."""
from collections import Counter

from sim.models import BOOTING, DONE, PENDING, READY, RUNNING


class SafetyChecker:
    def __init__(self):
        self.last_time = 0
        self.started = set()      # job ids that have started (a job may start only once)

    def job_started(self, job):
        assert job.id not in self.started, f"job {job.id} started twice"
        self.started.add(job.id)

    def check(self, now, nodes, jobs_by_id, pool):
        """Called after every event."""
        assert now >= self.last_time, "clock went backwards"
        self.last_time = now
        self._check_cluster(nodes, jobs_by_id)
        self._check_cloud(now, pool, jobs_by_id)

    def _check_cluster(self, nodes, jobs_by_id):
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
        cluster_running = {j.id for j in jobs_by_id.values() if j.state == RUNNING and not j.on_cloud}
        assert running_ids == cluster_running, "running jobs and node state disagree"

    def _check_cloud(self, now, pool, jobs_by_id):
        insts = pool.instances.values()
        live = [i for i in insts if i.state in (BOOTING, READY)]
        assert len(live) <= pool.max_instances, "too many live cloud instances"
        assert len(live) == pool.n_live, "live instance count bookkeeping"
        job_ids = [i.job_id for i in insts]
        assert len(job_ids) == len(set(job_ids)), "a job was given two cloud instances"

        for i in insts:
            job = jobs_by_id[i.job_id]
            if i.state == BOOTING:
                assert i.ready_at is None and pool.billed_seconds(i, now) == 0, f"{i.id} billed while booting"
                assert job.state == PENDING and job.node_id is None, f"job {job.id} left cloud queue"
            elif i.state == READY:
                assert i.ready_at <= now and i.released_at is None, f"{i.id} bad READY times"
                assert job.state == RUNNING and job.on_cloud and job.instance_id == i.id, f"{i.id} job mismatch"
            else:  # RELEASED
                assert i.ready_at is not None and i.ready_at <= i.released_at, f"{i.id} bad RELEASED times"
                assert pool.billed_seconds(i, now) >= pool.min_billed, f"{i.id} under minimum billing"
                assert job.state == DONE, f"{i.id} released before its job finished"

        ready_jobs = Counter(i.job_id for i in insts if i.state == READY)
        for j in jobs_by_id.values():
            if j.state == RUNNING and j.on_cloud:
                assert ready_jobs[j.id] == 1 and j.node_id is None, f"cloud job {j.id} not on exactly one instance"
