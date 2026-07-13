"""
Database utilities for BAL Chatbot.

This module handles:
- Usage counter management
- Quota checking and increment
- Database operations
"""

from datetime import datetime, timezone
from typing import Tuple, Dict

try:
    from extensions import SessionLocal
    from models import UsageCounter
except ImportError:
    from web.extensions import SessionLocal
    from web.models import UsageCounter

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG


def utc_now():
    """Returns current UTC time as datetime."""
    return datetime.now(timezone.utc)


def today_key() -> str:
    """Returns today's date key for daily quota."""
    return utc_now().strftime("%Y-%m-%d")


def minute_key() -> str:
    """Returns current minute key for minute quota."""
    return utc_now().strftime("%Y-%m-%dT%H:%M")


def get_usage(subject_type: str, subject_id: str, period_type: str, period_key: str) -> int:
    """Gets the current usage count for a subject and period."""
    with SessionLocal() as db:
        row = db.get(UsageCounter, (subject_type, subject_id, period_type, period_key))
        return int(row.count) if row else 0


def quota_snapshot(identity: Dict) -> Dict:
    """Returns quota information for the given identity."""
    limits = CONFIG["limits"][identity["role"]]
    daily_used = get_usage(identity["subject_type"], identity["subject_id"], "day", today_key())
    minute_used = get_usage(identity["subject_type"], identity["subject_id"], "minute", minute_key())
    return {
        "daily_limit": limits["daily"],
        "daily_used": daily_used,
        "daily_remaining": max(limits["daily"] - daily_used, 0),
        "minute_limit": limits["minute"],
        "minute_used": minute_used,
        "minute_remaining": max(limits["minute"] - minute_used, 0),
    }


def check_quota(identity: Dict) -> Tuple[bool, Dict, str]:
    """
    Checks if the user has quota remaining.
    Returns (is_ok, usage_dict, error_message).
    """
    usage = quota_snapshot(identity)
    if usage["daily_remaining"] <= 0:
        return False, usage, "Günlük soru limitin doldu."
    if usage["minute_remaining"] <= 0:
        return False, usage, "Dakikalık soru limitine ulaştın. Biraz bekleyip tekrar dene."
    return True, usage, ""


def increment_usage(identity: Dict) -> Dict:
    """Increments usage counters for both daily and minute periods."""
    now = utc_now().isoformat()
    rows = [("day", today_key()), ("minute", minute_key())]
    with SessionLocal() as db:
        for period_type, period_key in rows:
            key = (
                identity["subject_type"],
                identity["subject_id"],
                period_type,
                period_key,
            )
            counter = db.get(UsageCounter, key)
            if counter is None:
                counter = UsageCounter(
                    subject_type=identity["subject_type"],
                    subject_id=identity["subject_id"],
                    period_type=period_type,
                    period_key=period_key,
                    count=1,
                    updated_at=now,
                )
                db.add(counter)
            else:
                counter.count += 1
                counter.updated_at = now
        db.commit()
    return quota_snapshot(identity)