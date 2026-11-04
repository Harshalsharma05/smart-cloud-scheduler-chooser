"""The rule-set menu for the progress check: B2, B3, B4, B5 (see IMPLEMENTATION.md, 'Rule sets')."""
from rules.base import DEADLINE_AWARE, EASY, EDF, FCFS, IMMEDIATE, NEVER, SJF, RuleSet

RULESETS = {
    "B2": RuleSet("B2", FCFS, EASY, NEVER, "FCFS + EASY backfill, cluster only"),
    "B3": RuleSet("B3", SJF, EASY, NEVER, "Shortest job first + EASY backfill, cluster only"),
    "B4": RuleSet("B4", EDF, EASY, DEADLINE_AWARE, "Earliest deadline first + EASY, rent cloud for at-risk deadlines"),
    "B5": RuleSet("B5", FCFS, EASY, IMMEDIATE, "FCFS + EASY, rent cloud for every job that cannot start"),
}
