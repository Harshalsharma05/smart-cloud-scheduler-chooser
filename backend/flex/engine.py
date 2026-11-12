"""GPU-pool discrete-event engine: FCFS with skip-ahead, integer seconds, no wall clock."""
import copy
import heapq
import math
from collections import Counter

from . import safety
from .models import DONE, QUEUED, RUNNING

FINISH, SUBMIT = 0, 1   # at equal times, finishes are handled before submits


class Engine:
    def __init__(self, jobs, G, check_every=1000, on_event=None):
        """jobs: list of FJob. check_every: full safety check every N events (0 = never).
        on_event(engine, t) is called after every handled event (e.g. for power tracking)."""
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
        for j in sorted(jobs, key=lambda j: (j.submit, j.job_id)):
            self._push(j.submit, SUBMIT, j.job_id, 0)

    # ---- event loop ----
    def _push(self, t, kind, job_id, version):
        heapq.heappush(self.heap, (t, kind, self._seq, job_id, version))
        self._seq += 1

    def _step(self):
        t, kind, _, job_id, version = heapq.heappop(self.heap)
        job = self.jobs[job_id]
        if kind == FINISH and (version != job.version or job.state != RUNNING):
            return                   # stale finish, ignore
        self.now = t
        if kind == FINISH:
            self._finish(job)
            self._schedule()
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

    def _start(self, job):
        assert job.state == QUEUED, f"{job.job_id} started from {job.state}"
        self._qsizes[job.gpus] -= 1
        if self._qsizes[job.gpus] == 0:
            del self._qsizes[job.gpus]
        self.free -= job.gpus
        job.state, job.mark, job.speed = RUNNING, self.now, 1.0
        job.n_starts += 1
        if job.first_start is None:
            job.first_start = self.now
        self.running.add(job.job_id)
        remaining = job.wall_s + job.extra_work - job.work_done
        self._push(self.now + math.ceil(remaining / job.speed), FINISH, job.job_id, job.version)

    def _finish(self, job):
        job.segments.append((job.mark, self.now, "onprem", job.speed))
        job.work_done = job.wall_s + job.extra_work
        job.state, job.end, job.mark = DONE, self.now, self.now
        self.free += job.gpus
        self.running.discard(job.job_id)
