"""Scenario presets: quiet, rush, deadline_heavy -> generator parameters.
Values are starting points; tune them in Phase 5 so the rule sets clearly differ."""

HORIZON_S = 24 * 3600

# rates: list of (start_hour, end_hour, jobs_per_hour); first matching segment wins
SCENARIOS = {
    "quiet": {
        "label": "Quiet day",
        "description": "Steady ~15 jobs/h, few deadlines with generous slack.",
        "rates": [(0, 24, 15)],
        "gpu_frac": 0.15, "deadline_frac": 0.20, "slack": (3.0, 6.0),
    },
    "rush": {
        "label": "Rush hour",
        "description": "~15 jobs/h with a 6-hour burst of ~80 jobs/h.",
        "rates": [(8, 14, 80), (0, 24, 15)],
        "gpu_frac": 0.20, "deadline_frac": 0.30, "slack": (2.0, 4.0),
    },
    "deadline_heavy": {
        "label": "Deadline heavy",
        "description": "~40 jobs/h, most jobs have tight deadlines.",
        "rates": [(0, 24, 40)],
        "gpu_frac": 0.20, "deadline_frac": 0.70, "slack": (1.3, 2.5),
    },
}
