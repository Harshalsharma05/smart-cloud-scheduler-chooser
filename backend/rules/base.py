"""RuleSet: order (FCFS/SJF/EDF), backfill (none/EASY), burst policy (NEVER/IMMEDIATE/DEADLINE_AWARE).
Placement is always best-fit. One generic RuleSet class configured by these fields - no class hierarchy."""
from dataclasses import dataclass

FCFS, SJF, EDF = "FCFS", "SJF", "EDF"
NONE, EASY = "NONE", "EASY"
NEVER, IMMEDIATE, DEADLINE_AWARE = "NEVER", "IMMEDIATE", "DEADLINE_AWARE"


@dataclass(frozen=True)
class RuleSet:
    id: str
    order: str
    backfill: str
    burst: str
    label: str = ""


def sort_key(order):
    """Queue sort key for an order. Schedulers only ever look at est_runtime."""
    if order == FCFS:
        return lambda j: (j.arrival, j.id)
    if order == SJF:
        return lambda j: (j.est_runtime, j.arrival, j.id)
    if order == EDF:   # no-deadline jobs after all deadline jobs, FCFS among them
        return lambda j: (j.deadline is None, j.deadline or 0, j.arrival, j.id)
    raise ValueError(f"unknown order {order}")
