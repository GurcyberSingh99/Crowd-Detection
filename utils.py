"""
utils.py – Shared helpers for the crowd-occupancy system.
"""

from datetime import datetime, timezone


# ── Occupancy thresholds ────────────────────────────────────────────────────
# Adjust these to match your venue capacity.
THRESHOLDS = {
    "low":      (0,  5),    # 0–5 people  → Low
    "moderate": (6,  15),   # 6–15 people → Moderate
    "high":     (16, 30),   # 16–30 people → High
    "critical": (31, 9999), # 31+ people  → Critical
}


def classify_occupancy(count: int) -> str:
    """Return a human-readable occupancy label for a given person count."""
    for label, (lo, hi) in THRESHOLDS.items():
        if lo <= count <= hi:
            return label
    return "critical"


def now_utc() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 string with milliseconds."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
