"""
Runtime utilities for BAL Chatbot.

This module handles:
- Application startup
- HTTPS enforcement
"""

import os
import sys
from pathlib import Path

try:
    from extensions import SessionLocal, reinit_engine, app, LOG_DIR
    from models import User, UsageCounter, ChatLog, init_db, database_ready
except ImportError:
    from web.extensions import SessionLocal, reinit_engine, app, LOG_DIR
    from web.models import User, UsageCounter, ChatLog, init_db, database_ready

try:
    from config import CONFIG, PROJECT_ROOT
except ImportError:
    from web.config import CONFIG, PROJECT_ROOT


def enforce_https():
    """Before request handler to enforce HTTPS redirects."""
    from flask import request
    if request.path.startswith("/api/health"):
        return None
    if CONFIG["force_https"] and not request.is_secure:
        host = request.headers.get("Host", "")
        is_local = host.startswith("127.0.0.1") or host.startswith("localhost")
        if not is_local:
            return "", 308, {"Location": request.url.replace("http://", "https://", 1)}
    return None


def startup():
    """
    Runs once before the Flask server accepts requests.
    Loads the vector store and validates Groq configuration.
    Falls back to SQLite if PostgreSQL is unreachable.
    """
    import logging
    import time

    log = logging.getLogger(__name__)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Try database connection; fall back to SQLite on failure ──────────────
    db_ok = False
    try:
        init_db()
        db_ok = True
        log.info("Database connection established: %s", CONFIG["database_url"][:50])
    except Exception as e:
        log.warning("Database connection failed (%s). Falling back to SQLite.", str(e)[:80])

    if not db_ok:
        sqlite_path = str(PROJECT_ROOT / "data" / "app.db")
        CONFIG["database_url"] = f"sqlite:///{sqlite_path}"
        log.info("Switching to SQLite: %s", CONFIG["database_url"])
        reinit_engine(CONFIG["database_url"])

        try:
            init_db()
            log.info("SQLite fallback successful.")
        except Exception as e2:
            log.error("SQLite fallback also failed: %s", e2)
            sys.exit(1)

    log.info("BAL Chatbot Web API starting...")
    log.info(f"Runtime pid={os.getpid()} cwd={Path.cwd()} log_file={LOG_DIR / 'web.log'}")
    log.info(f"Provider: {CONFIG['provider']}")
    log.info(f"Embedding model: {CONFIG['embedding_model']} (local, no API)")
    log.info(f"HTTPS enforcement: {CONFIG['force_https']}")
    if not CONFIG["secret_key"]:
        log.warning("FLASK_SECRET_KEY is not set. Sessions will reset after server restart.")

    # ── Pre-load embedding model at startup on HF Space (2 vCPU) ──────────────
    # Loading ~500MB model takes ~5-10s on CPU; do it here so first request is fast
    log.info("Pre-loading local embedding model (this may take a moment)...")
    from sentence_transformers import SentenceTransformer
    try:
        t0 = time.time()
        import extensions as ext_module
        ext_module.embedding_model = SentenceTransformer(CONFIG["embedding_model"])
        log.info(
            f"✓ Embedding model loaded in {time.time() - t0:.1f}s — "
            f"dim={ext_module.embedding_model.get_embedding_dimension()}"
        )
    except Exception as e:
        log.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

    # ── Load vector store (passing the pre-loaded embedding model) ────────────
    from rag import VectorStore
    try:
        import extensions as ext_module
        ext_module.vector_store = VectorStore(
            CONFIG["faiss_index_file"],
            CONFIG["chunks_meta_file"],
            CONFIG["embedding_model"],
            embedding_model=ext_module.embedding_model,
        )
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    # ── Groq configuration check ─────────────────────────────────────────────
    if not CONFIG["groq_api_keys"]:
        log.error(
            "No Groq API key is set. "
            "Set GROQ_API_KEY, GROQ_API_KEYS or GROQ_API_KEY_1..5 before starting the server."
        )
        sys.exit(1)
    log.info(
        "Groq configured — key_count=%s primary_model=%s",
        len(CONFIG["groq_api_keys"]),
        CONFIG["groq_model_chain"][0],
    )

    from llm import LLMGateway
    import extensions as ext_module
    ext_module.llm_gateway = LLMGateway(CONFIG)
    log.info(f"LLM gateway ready — active provider: {ext_module.llm_gateway.active_provider}")
    log.info(f"HF Space tuning — congestion_threshold={CONFIG['congestion_threshold']}, "
             f"max_workers={CONFIG['embedding_max_workers']}")
    port = int(os.getenv("PORT", "5000"))
    scheme = "https" if CONFIG["local_https"] and not os.getenv("PORT") else "http"
    log.info(f"Server starting on {scheme}://0.0.0.0:{port}")