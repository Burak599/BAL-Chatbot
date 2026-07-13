"""
Authentication utilities for BAL Chatbot.

This module handles:
- Fingerprint-based identity management
- User public data formatting
- Email normalization
"""

import re
import logging
from typing import Optional, Dict

from flask import request, session
from sqlalchemy.exc import IntegrityError

log = logging.getLogger(__name__)

# Import CONFIG, extensions, and models after config import in app.py
# These will be injected or imported locally
try:
    from extensions import SessionLocal
    from models import User
except ImportError:
    from web.extensions import SessionLocal
    from web.models import User

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG


def utc_now():
    """Returns current UTC time as datetime."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    """Normalizes email to lowercase stripped form."""
    return email.strip().lower()


def role_for_email(email: str) -> str:
    """Returns 'admin' if email is in admin_emails, else 'user'."""
    return "admin" if normalize_email(email) in CONFIG["admin_emails"] else "user"


def user_to_public(user: User) -> Dict:
    """Converts User model to public-safe dictionary."""
    role = user.role
    is_visitor = role == "visitor" or user.provider == "fingerprint"
    return {
        "id": user.id,
        "email": None if is_visitor else user.email,
        "role": "visitor" if is_visitor else role,
        "mode": "visitor" if is_visitor else "account",
    }


def get_client_fingerprint() -> Optional[str]:
    """Extracts and validates client fingerprint from request headers."""
    fingerprint = (request.headers.get("X-Client-Fingerprint") or "").strip()
    if not fingerprint:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,255}", fingerprint):
        log.warning("Rejected malformed client fingerprint: %r", fingerprint[:80])
        return None
    return fingerprint


def get_current_identity() -> Optional[Dict]:
    """
    Returns current user identity based on session or fingerprint.
    Creates a visitor user if fingerprint is provided but user doesn't exist.
    """
    user_id = session.get("user_id")
    if user_id:
        with SessionLocal() as db:
            user = db.get(User, int(user_id))
            if user:
                return {
                    "subject_type": "user",
                    "subject_id": str(user.id),
                    "role": user.role,
                    "public": user_to_public(user),
                }
        session.pop("user_id", None)

    fingerprint = get_client_fingerprint() or session.get("fingerprint")

    if fingerprint:
        try:
            with SessionLocal() as db:
                user = db.query(User).filter(User.fingerprint == fingerprint).first()
                if user is None:
                    log.info("No existing user with fingerprint found: %s", fingerprint)
                else:
                    log.info("Found existing user id=%s for fingerprint", user.id)

                if user is None:
                    user = User(
                        email=None,
                        fingerprint=fingerprint,
                        password_hash=None,
                        provider="fingerprint",
                        role="visitor",
                        created_at=utc_now().isoformat(),
                    )
                    db.add(user)
                    try:
                        db.commit()
                        db.refresh(user)
                        log.info("Created visitor user id=%s fingerprint=%s", user.id, fingerprint)
                    except IntegrityError:
                        db.rollback()
                        user = db.query(User).filter(User.fingerprint == fingerprint).first()
                        if user:
                            log.info("Detected concurrent creation; using existing user id=%s", user.id)

                if user:
                    session["fingerprint"] = fingerprint
                    return {
                        "subject_type": "user",
                        "subject_id": str(user.id),
                        "role": user.role,
                        "public": user_to_public(user),
                    }
        except Exception as e:
            log.exception("Failed to establish fingerprint identity (%s). Headers: %s", e, {
                "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
                "X-Client-Fingerprint": request.headers.get("X-Client-Fingerprint"),
                "User-Agent": request.headers.get("User-Agent"),
            })
            return None

    return None