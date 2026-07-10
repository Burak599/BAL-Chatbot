"""Flask app, database engine/session, and shared runtime state."""

import secrets
import threading
from typing import Dict, List

from flask import Flask
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.middleware.proxy_fix import ProxyFix

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG


def _create_engine(database_url: str):
    connect_args = (
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {}
    )
    return create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args,
    )


def reinit_engine(database_url: str) -> None:
    """Recreate engine and SessionLocal (e.g. PostgreSQL → SQLite fallback)."""
    global engine, SessionLocal
    engine = _create_engine(database_url)
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.config.update(
    SECRET_KEY=CONFIG["secret_key"] or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=CONFIG["force_https"],
)

Base = declarative_base()
engine = _create_engine(CONFIG["database_url"])
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)

# Populated during startup()
vector_store = None
embedding_model = None
llm_gateway = None
embedding_executor = None

conversation_sessions: Dict[str, List[Dict]] = {}
active_requests = 0
active_requests_lock = threading.Lock()
