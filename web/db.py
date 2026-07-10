from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from web.config import CONFIG, PROJECT_ROOT
from web.models import Base


def _build_engine(database_url: str):
    return create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )


engine = _build_engine(CONFIG["database_url"])
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def reconfigure_database(database_url: str) -> None:
    global engine, SessionLocal
    engine = _build_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    if CONFIG["database_url"].startswith("sqlite"):
        Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ensure_user_fingerprint_column()
    _ensure_user_email_nullable()


def _ensure_user_fingerprint_column() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "fingerprint" in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN fingerprint VARCHAR(255)"))
        try:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_fingerprint ON users (fingerprint)"))
        except Exception:
            pass


def _ensure_user_email_nullable() -> None:
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
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ALTER COLUMN email DROP NOT NULL"))


def database_ready() -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
