"""SQLAlchemy models and database initialization."""

import logging
from pathlib import Path

from sqlalchemy import Column, Integer, String, Text, inspect, text

try:
    from config import CONFIG, PROJECT_ROOT
except ImportError:
    from web.config import CONFIG, PROJECT_ROOT

try:
    import extensions
    from extensions import Base
except ImportError:
    import web.extensions as extensions
    from web.extensions import Base

log = logging.getLogger(__name__)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    fingerprint = Column(String(255), nullable=True, unique=True, index=True)
    password_hash = Column(Text, nullable=True)
    provider = Column(String(32), nullable=False, default="password")
    role = Column(String(32), nullable=False, default="user", index=True)
    created_at = Column(String(64), nullable=False)


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    subject_type = Column(String(32), primary_key=True)
    subject_id = Column(String(255), primary_key=True)
    period_type = Column(String(32), primary_key=True)
    period_key = Column(String(64), primary_key=True)
    count = Column(Integer, nullable=False, default=0)
    updated_at = Column(String(64), nullable=False)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    question_index = Column(Integer, nullable=False, default=0, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)
    feedback = Column(String(16), nullable=True)
    feedback_text = Column(Text, nullable=True)


def init_db() -> None:
    if CONFIG["database_url"].startswith("sqlite"):
        Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=extensions.engine)
    _ensure_user_fingerprint_column()
    _ensure_user_email_nullable()


def _ensure_user_fingerprint_column() -> None:
    inspector = inspect(extensions.engine)
    if not inspector.has_table("users"):
        return
    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "fingerprint" in existing_columns:
        return
    log.warning("Adding missing users.fingerprint column to existing database schema.")
    with extensions.engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN fingerprint VARCHAR(255)"))
        try:
            conn.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_fingerprint ON users (fingerprint)")
            )
        except Exception:
            log.exception("Could not create index for users.fingerprint. This is non-fatal.")


def _ensure_user_email_nullable() -> None:
    inspector = inspect(extensions.engine)
    if not inspector.has_table("users"):
        return
    if extensions.engine.dialect.name == "sqlite":
        return
    cols = {c["name"]: c for c in inspector.get_columns("users")}
    if "email" not in cols:
        return
    if cols["email"].get("nullable", True):
        return
    log.warning("Making users.email column nullable to support anonymous fingerprint users.")
    try:
        with extensions.engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ALTER COLUMN email DROP NOT NULL"))
    except Exception:
        log.exception("Failed to alter users.email to nullable; anonymous user creation may fail.")


def database_ready() -> bool:
    try:
        with extensions.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        log.exception("Database health check failed")
        return False
