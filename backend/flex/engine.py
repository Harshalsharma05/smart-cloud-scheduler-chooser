"""GPU-pool discrete-event engine: FCFS with skip-ahead, integer seconds, no wall clock.

Running jobs can be suspended, resumed, power-capped or burst to the cloud (SIM_PLAN D8-D9).
Progress is `work_done` in nominal seconds. A job's finish event carries the job's `version`;
any change of speed or state bumps the version, so old finish events are ignored (D15).
"""
import bisect
import copy
import heapq
import math
from collections import Counter

from . import cloud as cloudlib
from . import safety
from .models import (CLOUD_BOOTING, CLOUD_RUNNING, DONE, QUEUED, RUNNING, SUSPENDED)

FINISH, READY, SUBMIT = 0, 1, 2   # at equal times: finishes, then cloud-ready, then submits


class Engine:
    def __init__(self, jobs, G, check_every=1000, on_event=None, cloud=None):
        """jobs: list of FJob. check_every: full safety check every N events (0 = never).
        on_event(engine, t) is called after every handled event (e.g. for power tracking).
        cloud: a flex.cloud.Cloud, needed only for burst()."""
        self.G, self.free, self.now = G, G, 0
        self.jobs = {j.job_id: j for j in jobs}
        if len(self.jobs) != len(jobs):
            raise ValueError("duplicate job_id")
        self.queue = []              # QUEUED jobs, sorted by (submit, job_id)
        self.running = set()         # job_ids on the on-prem pool
        self._qsizes = Counter()     # gpus -> number of queued jobs, to skip useless scans
        self.heap, self._seq = [], 0
        self.n_events = 0
        self.check_every, self.on_event = check_every, on_event
        self.cloud = cloud
        self.cap_f, self.speed_fn = 1.0, None   # cluster-wide power cap level and its speed function
        for j in sorted(jobs, key=lambda j: (j.submit, j.job_id)):
            self._push(j.submit, SUBMIT, j.job_id, 0)

    # ---- event loop ----
    def _push(self, t, kind, job_id, version):
        heapq.heappush(self.heap, (t, kind, self._seq, job_id, version))
        self._seq += 1

    def _step(self):
        t, kind, _, job_id, version = heapq.heappop(self.heap)
        job = self.jobs[job_id]
        if kind == FINISH and (version != job.version or job.state not in (RUNNING, CLOUD_RUNNING)):
            return                   # stale finish, ignore
        if kind == READY and (version != job.version or job.state != CLOUD_BOOTING):
            return
        self.now = t
        if kind == FINISH:
            self._finish(job)
            self._schedule()
        elif kind == READY:
            self._cloud_ready(job)
        else:
            job.state = QUEUED
            self.queue.append(job)
            self._qsizes[job.gpus] += 1
            self._schedule(new=job)
        self.n_events += 1
        if self.check_every and self.n_events % self.check_every == 0:
            safety.check(self)
        if self.on_event:
            self.on_event(self, t)

    def run_until(self, t):
        """Handle every event with time <= t, then set the clock to t."""
        while self.heap and self.heap[0][0] <= t:
            self._step()
        self.now = max(self.now, t)
        self._notify()

    def run_to_drain(self):
        """Handle all remaining events."""
        while self.heap:
            self._step()

    def snapshot(self):
        """Independent copy of the whole state (the on_event hook is not copied)."""
        hook, self.on_event = self.on_event, None
        try:
            return copy.deepcopy(self)
        finally:
            self.on_event = hook

    def _notify(self):
        """Tell the on_event hook the state may have changed at `now` (after clock moves and job moves)."""
        if self.on_event:
            self.on_event(self, self.now)

    # ---- progress and segments ----
    def _account(self, job, where):
        """Bring work_done up to `now` and close the current segment (recorded as `where`)."""
        dt = self.now - job.mark
        if where in ("onprem", "cloud"):
            job.work_done += job.speed * dt
        if dt > 0:
            job.segments.append((job.mark, self.now, where, job.speed))
        job.mark = self.now

    def _schedule_finish(self, job):
        """Bump the version (drops older finish events) and queue the finish for the current speed."""
        job.version += 1
        remaining = job.wall_s + job.extra_work - job.work_done
        dt = max(0, math.ceil(round(remaining / job.speed, 6)))
        self._push(self.now + dt, FINISH, job.job_id, job.version)

    def _speed(self, job):
        return 1.0 if self.cap_f >= 1.0 else self.speed_fn(job, self.cap_f)

    # ---- scheduling ----
    def _schedule(self, new=None):
        """FCFS with skip-ahead: start every queued job that fits, in queue order."""
        if not self.queue or self.free < min(self._qsizes):
            return
        if new is not None:          # the queue was stable before `new`, so only `new` can start
            if new.gpus <= self.free:
                self.queue.pop()
                self._start(new)
            return
        rest = []
        for i, j in enumerate(self.queue):
            if self.free == 0:
                rest.extend(self.queue[i:])
                break
            if j.gpus <= self.free:
                self._start(j)
            else:
                rest.append(j)
        self.queue = rest

    def _dequeue_size(self, job):
        self._qsizes[job.gpus] -= 1
        if self._qsizes[job.gpus] == 0:
            del self._qsizes[job.gpus]

    def _start(self, job):
        """Start (or restart after a suspend) a job that has already left the queue list."""
        assert job.state == QUEUED, f"{job.job_id} started from {job.state}"
        self._dequeue_size(job)
        self.free -= job.gpus
        job.state, job.mark, job.speed = RUNNING, self.now, self._speed(job)
        if job.first_start is None:
            job.first_start = self.now
            job.n_starts += 1
        self.running.add(job.job_id)
        self._schedule_finish(job)

    def _finish(self, job):
        on_cloud = job.state == CLOUD_RUNNING
        self._account(job, "cloud" if on_cloud else "onprem")
        job.work_done = job.wall_s + job.extra_work
        job.state, job.end = DONE, self.now
        if on_cloud:
            cloudlib.release(self.cloud, job.job_id, self.now)
        else:
            self.free += job.gpus
            self.running.discard(job.job_id)

    # ---- running-job moves ----
    def suspend(self, job):
        """Free the job's GPUs at once and keep its progress (checkpoint assumed instant)."""
        if job.state != RUNNING:
            raise ValueError(f"cannot suspend {job.job_id} in state {job.state}")
        self._account(job, "onprem")
        job.version += 1                      # drops the pending finish
        job.state, job.speed = SUSPENDED, 0.0
        job.n_suspends += 1
        self.free += job.gpus
        self.running.discard(job.job_id)
        self._schedule()
        self._notify()

    def resume(self, job, restore_s=0):
        """Put a suspended job back in the queue (FCFS position); restore_s is added to its remaining work."""
        if job.state != SUSPENDED:
            raise ValueError(f"cannot resume {job.job_id} in state {job.state}")
        self._account(job, "suspended")
        job.extra_work += restore_s
        job.n_resumes += 1
        job.state = QUEUED
        bisect.insort(self.queue, job, key=lambda j: (j.submit, j.job_id))
        self._qsizes[job.gpus] += 1
        self._schedule()
        self._notify()

    def set_cap(self, f, speed_fn=None):
        """Cap every on-prem GPU at level f (1.0 = no cap). speed_fn(job, f) -> speed in (0, 1].
        Running jobs change speed now; their finish events are rescheduled."""
        if speed_fn is not None:
            self.speed_fn = speed_fn
        if f < 1.0 and self.speed_fn is None:
            raise ValueError("set_cap needs a speed_fn")
        self.cap_f = f
        for jid in sorted(self.running):
            job = self.jobs[jid]
            new_speed = self._speed(job)
            if new_speed != job.speed:
                self._account(job, "onprem")
                job.speed = new_speed
                self._schedule_finish(job)
        self._notify()

    def burst(self, job):
        """Move a RUNNING or QUEUED job to the cloud; returns False (job untouched) if the cloud GPU cap blocks it.
        A running job's on-prem GPUs are freed at once and migrate_s is added to its work."""
        if self.cloud is None:
            raise ValueError("engine has no cloud")
        if job.state not in (RUNNING, QUEUED):
            raise ValueError(f"cannot burst {job.job_id} in state {job.state}")
        if not cloudlib.can_request(self.cloud, job.gpus):
            return False
        if job.state == RUNNING:
            self._account(job, "onprem")
            self.free += job.gpus
            self.running.discard(job.job_id)
            job.extra_work += self.cloud.params.migrate_s
        else:
            idx = next(i for i, q in enumerate(self.queue) if q is job)
            del self.queue[idx]
            self._dequeue_size(job)
        job.state, job.on_cloud, job.speed, job.mark = CLOUD_BOOTING, True, 0.0, self.now
        job.version += 1
        cloudlib.request(self.cloud, job.job_id, job.gpus, self.now)
        self._push(self.now + self.cloud.params.boot_delay_s, READY, job.job_id, job.version)
        self._schedule()
        self._notify()
        return True

    def _cloud_ready(self, job):
        self._account(job, "boot")
        cloudlib.set_ready(self.cloud, job.job_id, self.now)
        job.state, job.speed = CLOUD_RUNNING, 1.0
        self._schedule_finish(job)
