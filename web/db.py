"""
Database utilities for BAL Chatbot.

This module handles:
- init_db(): Database initialization
- database_ready(): Health check for database connection
- Schema migration helpers
"""

import logging
from pathlib import Path

try:
    from extensions import engine, SessionLocal, Base
    from config import PROJECT_ROOT
except ImportError:
    from web.extensions import engine, SessionLocal, Base
    from web.config import PROJECT_ROOT

try:
    from models import User, UsageCounter, ChatLog
except ImportError:
    from web.models import User, UsageCounter, ChatLog

log = logging.getLogger(__name__)


def init_db() -> None:
    """
    Creates database tables if they don't exist.
    Handles schema migrations for existing databases.
    """
    if engine.url.startswith("sqlite"):
        Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    _ensure_user_fingerprint_column()
    _ensure_user_email_nullable()


def database_ready() -> bool:
    """Returns True if database connection is healthy."""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        log.exception("Database health check failed")
        return False


def _ensure_user_fingerprint_column() -> None:
    """Adds fingerprint column to users table if missing."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "fingerprint" in existing_columns:
        return

    log.warning("Adding missing users.fingerprint column to existing database schema.")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN fingerprint VARCHAR(255)"))
        try:
            conn.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_fingerprint ON users (fingerprint)")
            )
        except Exception:
            log.exception("Could not create index for users.fingerprint. Non-fatal.")


def _ensure_user_email_nullable() -> None:
    """Makes email column nullable for PostgreSQL if needed."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    if engine.dialect.name == "sqlite":
        return

    cols = {c["name"]: c for c in inspector.get_columns("users")}
    if "email" not in cols:
        return
    if cols["email"].get("nullable", True):
        return

    log.warning("Making users.email column nullable to support anonymous users.")
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ALTER COLUMN email DROP NOT NULL"))
    except Exception:
        log.exception("Failed to alter users.email to nullable; anonymous user creation may fail.")


