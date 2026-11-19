"""Grid curtailment events (SIM_PLAN D13) and the local-time helpers they need.

Simulated time is integer seconds from an origin `epoch0` (a UTC epoch). Event start times are drawn in
America/Los_Angeles local time, so the same code serves the synthetic range and Philly (S8 passes its own epoch0).
An optional `tz` key in the grid config (step 35: Asia/Shanghai for Acme) changes the zone; without it nothing changes.
"""
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")


def epoch_from_iso(text) -> int:
    """UTC epoch seconds of an ISO timestamp such as '2017-11-15T00:00:00+00:00'."""
    return int(dt.datetime.fromisoformat(text).timestamp())


def to_local(utc_epoch, tz=TZ) -> dt.datetime:
    """Local (America/Los_Angeles unless tz is given) wall-clock time of a UTC epoch."""
    return dt.datetime.fromtimestamp(utc_epoch, tz=tz)


def local_hour(utc_epoch) -> float:
    """Local hour of day as a fraction, e.g. 17.5 for 17:30."""
    t = to_local(utc_epoch)
    return t.hour + t.minute / 60 + t.second / 3600


def local_epoch(date, hour, minute=0, tz=TZ) -> int:
    """UTC epoch of a local date at hour:minute."""
    return int(dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=tz).timestamp())


@dataclass(frozen=True)
class Event:
    event_id: str
    t_start: int        # simulated seconds
    t_end: int
    duration_h: int
    cut: float          # fraction of power to shed, e.g. 0.2


def generate_events(rng, epoch0, t_lo, t_hi, cfg):
    """`events_per_cell` start times (distinct local days, one per day), crossed with every (duration, cut):
    all cells share the same start times. The window is [t_lo, t_hi) in simulated seconds. cfg = flex.yaml `grid:`.

    Rules (D13): start = a half-hour slot between 17:00 and 20:00 local; only on local days at least
    `skip_first_days` after the day of t_lo; the event must end before t_hi - end_margin_days."""
    tz = ZoneInfo(cfg["tz"]) if cfg.get("tz") else TZ
    first_day = to_local(epoch0 + t_lo, tz).date()
    last_end = t_hi - cfg["end_margin_days"] * 86400
    days = []                                   # eligible local dates: the whole day fits before last_end
    d = first_day + dt.timedelta(days=cfg["skip_first_days"])
    while local_epoch(d + dt.timedelta(days=1), 0, tz=tz) - epoch0 <= last_end:
        days.append(d)
        d += dt.timedelta(days=1)
    n = cfg["events_per_cell"]
    if len(days) < n:
        raise ValueError(f"only {len(days)} eligible days for {n} events per cell")
    slots = []                                  # (hour, minute) start slots, 17:00, 17:30, ... < 20:00
    step = int(cfg["slot_minutes"])
    m = int(cfg["start_hour_min"] * 60)
    while m < cfg["start_hour_max"] * 60:
        slots.append((m // 60, m % 60))
        m += step
    starts = []
    for day in sorted(rng.sample(days, n)):
        hh, mm = rng.choice(slots)
        starts.append(local_epoch(day, hh, mm, tz=tz) - epoch0)
    return [Event(f"d{dur}h_c{round(cut * 100)}_{k}", t_s, t_s + dur * 3600, dur, cut)
            for dur in cfg["durations_h"] for cut in cfg["cuts"] for k, t_s in enumerate(starts)]
