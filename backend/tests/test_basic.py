from rules.base import EASY, EDF, NEVER, RuleSet
from rules.library import RULESETS
from sim.engine import simulate
from sim.models import Job

H = 3600
CPU_SMALL_PRICE = 8


def world(cores=1):
    """A one-node cluster plus a one-type cloud."""
    return {
        "cluster": {"node_classes": [{"name": "cpu", "count": 1, "cores": cores, "mem_gb": 64, "gpus": 0}]},
        "cloud": {"boot_delay_s": 90, "max_instances": 22, "min_billed_s": 60, "instance_types": [
            {"name": "cpu_small", "cores": 8, "mem_gb": 32, "gpus": 0, "price_per_h": CPU_SMALL_PRICE}]},
        "burst": {"deadline_margin_s": 600},
    }


def job(i, runtime, deadline=None, cores=1):
    return Job(id=i, arrival=0, cores=cores, mem_gb=1, gpus=0, true_runtime=runtime,
               est_runtime=runtime, deadline=deadline)


def slide_jobs():
    """One free node; A (10 h), B (1 h, deadline 2 h), C (1 h, deadline 2 h), all at t=0."""
    return [job(1, 10 * H), job(2, H, 2 * H), job(3, H, 2 * H)]


def test_slide_example_fcfs():
    """FCFS (B2) -> B and C late."""
    (a, b, c), _ = simulate(slide_jobs(), world(), RULESETS["B2"])
    assert (a.start, a.end) == (0, 10 * H)
    assert b.end > b.deadline and c.end > c.deadline


def test_slide_example_edf():
    """EDF without cloud -> no late jobs (B 0-1 h, C 1-2 h, A last)."""
    edf_only = RuleSet("EDF", EDF, EASY, NEVER)
    (a, b, c), pool = simulate(slide_jobs(), world(), edf_only)
    assert b.end <= b.deadline and c.end <= c.deadline
    assert a.start == 2 * H and not pool.instances


def test_cloud_immediate_vs_never():
    """Node busy for 10 h, a 1 h job arrives. IMMEDIATE (B5) -> starts at 90 s on cloud,
    cost = price x 1 h. NEVER (B2) -> waits 10 h."""
    (_, b), pool = simulate([job(1, 10 * H), job(2, H)], world(), RULESETS["B5"])
    assert b.on_cloud and b.start == 90 and b.end == 90 + H
    assert pool.total_cost(b.end) == CPU_SMALL_PRICE * 1.0

    (_, b), pool = simulate([job(1, 10 * H), job(2, H)], world(), RULESETS["B2"])
    assert not b.on_cloud and b.start == 10 * H
    assert pool.total_cost(b.end) == 0


def test_easy_backfill():
    """4-core node. A (2 cores, 10 h) runs; big B (4 cores) waits for its reservation at 10 h.
    Small short C (1 core, 1 h) backfills now; long D (1 core, 20 h) would delay B, so it waits."""
    a, b, c, d = job(1, 10 * H, cores=2), job(2, H, cores=4), job(3, H), job(4, 20 * H)
    (a, b, c, d), _ = simulate([a, b, c, d], world(cores=4), RULESETS["B2"])
    assert c.start == 0
    assert b.start == 10 * H
    assert d.start >= b.end
