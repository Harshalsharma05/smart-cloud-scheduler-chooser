"""Per-run metrics: deadline miss rate, mean wait, mean bounded slowdown, cloud cost, score J."""


def run_metrics(jobs, pool, cfg) -> dict:
    """Raw (unrounded) metrics of one finished run. Runtime = true runtime."""
    tau = cfg["score"]["slowdown_tau_s"]
    deadline_jobs = [j for j in jobs if j.deadline is not None]
    late = [j for j in deadline_jobs if j.end > j.deadline]
    waits = [j.start - j.arrival for j in jobs]
    slowdowns = [max(1.0, (w + j.true_runtime) / max(j.true_runtime, tau)) for j, w in zip(jobs, waits)]
    end = max(j.end for j in jobs)
    return {
        "jobs": len(jobs),
        "miss_rate": len(late) / len(deadline_jobs) if deadline_jobs else 0.0,
        "avg_wait_min": sum(waits) / len(jobs) / 60,
        "avg_bounded_slowdown": sum(slowdowns) / len(jobs),
        "cloud_cost": float(pool.total_cost(end)),
        "cloud_instance_hours": sum(pool.billed_seconds(i, end) for i in pool.instances.values()) / 3600,
    }


def score(m, miss_ref, slowdown_ref, cost_ref, weights) -> float:
    """J = w_miss*(miss/miss_ref) + w_slowdown*(slowdown/slowdown_ref) + w_cost*(cost/cost_ref).
    Lower is better. A zero slowdown or cost reference is replaced by 1."""
    return (weights["w_miss"] * m["miss_rate"] / miss_ref
            + weights["w_slowdown"] * m["avg_bounded_slowdown"] / (slowdown_ref or 1)
            + weights["w_cost"] * m["cloud_cost"] / (cost_ref or 1))
