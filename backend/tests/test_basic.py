from sim.engine import IMMEDIATE, NEVER, simulate
from sim.models import Job

H = 3600
CPU_SMALL_PRICE = 8
ONE_NODE = {
    "cluster": {"node_classes": [{"name": "cpu", "count": 1, "cores": 1, "mem_gb": 4, "gpus": 0}]},
    "cloud": {"boot_delay_s": 90, "max_instances": 22, "min_billed_s": 60, "instance_types": [
        {"name": "cpu_small", "cores": 8, "mem_gb": 32, "gpus": 0, "price_per_h": CPU_SMALL_PRICE}]},
}


def job(i, runtime, deadline=None):
    return Job(id=i, arrival=0, cores=1, mem_gb=1, gpus=0, true_runtime=runtime,
               est_runtime=runtime, deadline=deadline)


def test_slide_example_fcfs():
    """One free node; A (10 h), B (1 h, deadline 2 h), C (1 h, deadline 2 h). FCFS -> B and C late."""
    (a, b, c), _ = simulate([job(1, 10 * H), job(2, H, 2 * H), job(3, H, 2 * H)], ONE_NODE)
    assert (a.start, a.end) == (0, 10 * H)
    assert b.end > b.deadline and c.end > c.deadline


def test_cloud_immediate_vs_never():
    """Node busy for 10 h, a 1 h job arrives. IMMEDIATE -> starts at 90 s on cloud, cost = price x 1 h.
    NEVER -> waits 10 h."""
    (_, b), pool = simulate([job(1, 10 * H), job(2, H)], ONE_NODE, burst=IMMEDIATE)
    assert b.on_cloud and b.start == 90 and b.end == 90 + H
    assert pool.total_cost(b.end) == CPU_SMALL_PRICE * 1.0

    (_, b), pool = simulate([job(1, 10 * H), job(2, H)], ONE_NODE, burst=NEVER)
    assert not b.on_cloud and b.start == 10 * H
    assert pool.total_cost(b.end) == 0
