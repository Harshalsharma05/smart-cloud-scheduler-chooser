from sim.engine import simulate
from sim.models import Job

H = 3600
ONE_NODE = {"cluster": {"node_classes": [{"name": "cpu", "count": 1, "cores": 1, "mem_gb": 4, "gpus": 0}]}}


def job(i, runtime, deadline=None):
    return Job(id=i, arrival=0, cores=1, mem_gb=1, gpus=0, true_runtime=runtime,
               est_runtime=runtime, deadline=deadline)


def test_slide_example_fcfs():
    """One free node; A (10 h), B (1 h, deadline 2 h), C (1 h, deadline 2 h). FCFS -> B and C late."""
    done = simulate([job(1, 10 * H), job(2, H, 2 * H), job(3, H, 2 * H)], ONE_NODE)
    a, b, c = done
    assert (a.start, a.end) == (0, 10 * H)
    assert b.end > b.deadline and c.end > c.deadline
