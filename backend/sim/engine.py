"""Discrete-event engine: heapq of (time, priority, seq, event). Owns the clock, the queue and the main loop.
Calls the active rule set after every event to get start/rent decisions."""
