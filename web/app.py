"""
=============================================================
 BAL Chatbot — Flask Web API
 Usage: python web/app.py
=============================================================
This script:
  1. Uses Groq as the only LLM provider
  2. Loads FAISS index and chunk metadata
  3. For each /api/chat request:
       a. Retrieves the most relevant chunks (ONCE per query)
       b. Builds an augmented prompt (context + question)
       c. Sends the request through the LLM gateway
       d. Streams the response from Groq
  4. Exposes /api/health, /api/chat, /api/clear endpoints
=============================================================
Prerequisites:
  - A valid Groq API key in the GROQ_API_KEY environment variable
  - 01_build_vectorstore.py must have been run
=============================================================
"""

import os
import sys
import json
import time
from pathlib import Path

import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Generator, Optional, Tuple

import numpy as np
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from flask import request, jsonify, Response, stream_with_context, send_from_directory, session
from curl_cffi import requests as curl_requests

# Import config first (needs to be before extensions)
try:
    from config import CONFIG, SYSTEM_PROMPT, PROJECT_ROOT, WEB_DIR, LOG_DIR
except ImportError:
    from web.config import CONFIG, SYSTEM_PROMPT, PROJECT_ROOT, WEB_DIR, LOG_DIR

try:
    import extensions
    from extensions import app
    from models import User, UsageCounter, ChatLog, init_db, database_ready
except ImportError:
    import web.extensions as extensions
    from web.extensions import app
    from web.models import User, UsageCounter, ChatLog, init_db, database_ready

# Import RAG components from the local rag module (now inside web/)
from rag import VectorStore, format_context, build_augmented_user_message, build_sources_payload

# Import llm utilities from modular structure
from llm import LLMGateway, strip_reasoning_blocks

# Import auth and db utilities from modular structure
from auth import (
    normalize_email,
    role_for_email,
    user_to_public,
    get_client_fingerprint,
    get_current_identity,
)
from db import (
    utc_now,
    today_key,
    minute_key,
    get_usage,
    quota_snapshot,
    check_quota,
    increment_usage,
)


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "web.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTPS Enforcement
# ═══════════════════════════════════════════════════════════════════════════════

@app.before_request
def enforce_https():
    if request.path.startswith("/api/health"):
        return None
    if CONFIG["force_https"] and not request.is_secure:
        host = request.headers.get("Host", "")
        is_local = host.startswith("127.0.0.1") or host.startswith("localhost")
        if not is_local:
            return "", 308, {"Location": request.url.replace("http://", "https://", 1)}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Flask Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serves the frontend HTML file."""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def serve_files(filename):
    return send_from_directory(WEB_DIR, filename)


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    identity = get_current_identity()
    if not identity:
        return jsonify({
            "authenticated": False,
            "google_configured": bool(CONFIG["google_client_id"]),
            "google_client_id": CONFIG["google_client_id"],
            "https_required": CONFIG["force_https"],
        })

    usage = quota_snapshot(identity)
    limits = CONFIG["limits"].get(identity["role"], CONFIG["limits"]["user"])
    return jsonify({
        "authenticated": True,
        "user": identity["public"],
        "role": identity["role"],
        "daily_used": usage["daily_used"],
        "minute_used": usage["minute_used"],
        "daily_limit": limits["daily"],
        "minute_limit": limits["minute"],
        "near_limit": usage["daily_used"] >= 30,
        "google_configured": bool(CONFIG["google_client_id"]),
        "google_client_id": CONFIG["google_client_id"],
        "https_required": CONFIG["force_https"],
    })


def unsupported_auth():
    return jsonify({"error": "Authentication flow is not supported for this app."}), 404


@app.route("/api/auth/guest", methods=["POST"])
def auth_guest():
    return unsupported_auth()


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    return unsupported_auth()


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    return unsupported_auth()


@app.route("/api/auth/google", methods=["POST"])
def auth_google():
    return unsupported_auth()


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    return unsupported_auth()


@app.route("/api/health", methods=["GET"])
def health():
    """
    Returns a JSON status object.
    """
    status = {
        "provider": CONFIG["provider"],
        "vectorstore": extensions.vector_store is not None,
        "embedding_model": CONFIG["embedding_model"],
        "database": database_ready(),
        "chunks": extensions.vector_store.index.ntotal if extensions.vector_store else 0,
    }

    if extensions.llm_gateway is None:
        status.update({"status": "degraded", "provider": None})
        return jsonify(status)

    provider_status = extensions.llm_gateway.status()
    status.update(provider_status)

    if not extensions.vector_store or not status["database"]:
        status["status"] = "degraded"

    return jsonify(status)


@app.route("/api/chat/feedback", methods=["POST"])
def chat_feedback():
    body = request.get_json()
    if not body or "question_index" not in body:
        return jsonify({"error": "question_index gerekli"}), 400

    identity = get_current_identity()
    if not identity:
        return jsonify({"error": "Kimlik alınamadı"}), 401

    question_index = body["question_index"]
    feedback = body.get("feedback")
    feedback_text = body.get("feedback_text", "").strip()

    if feedback is not None and feedback not in ("like", "dislike"):
        return jsonify({"error": "feedback sadece 'like' veya 'dislike' olabilir"}), 400

    user_id = int(identity["subject_id"])
    try:
        with extensions.SessionLocal() as db:
            log_entry = db.query(ChatLog).filter(
                ChatLog.user_id == user_id,
                ChatLog.question_index == question_index,
            ).first()
            if not log_entry:
                return jsonify({"error": "Soru bulunamadı"}), 404

            if feedback is not None:
                log_entry.feedback = feedback
            if feedback_text:
                log_entry.feedback = "feedback"
                log_entry.feedback_text = feedback_text
            db.commit()
        return jsonify({"ok": True})
    except Exception as e:
        log.exception("Feedback save failed for user_id=%s question_index=%s", user_id, question_index)
        return jsonify({"error": "Geri bildirim kaydedilemedi"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint — Server-Sent Events (SSE) streaming.
    """
    body = request.get_json()
    if not body or not body.get("message"):
        return jsonify({"error": "message alanı gerekli", "error_type": "technical"}), 400

    user_message = body["message"].strip()
    session_id = body.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Boş mesaj", "error_type": "technical"}), 400

    identity = get_current_identity()

    if not identity:
        fallback_fingerprint = (request.headers.get("X-Client-Fingerprint") or "").strip()
        if fallback_fingerprint and re.fullmatch(r"[A-Za-z0-9_-]{8,255}", fallback_fingerprint):
            identity = {
                "subject_type": "fingerprint_fallback",
                "subject_id": fallback_fingerprint,
                "role": "visitor",
                "public": {
                    "id": 0,
                    "email": None,
                    "role": "visitor",
                    "mode": "visitor_fallback",
                },
            }
            log.warning("Using fallback identity for fingerprint: %s", fallback_fingerprint[:20])
        else:
            try:
                headers_snapshot = {
                    "X-Client-Fingerprint": request.headers.get("X-Client-Fingerprint"),
                    "User-Agent": request.headers.get("User-Agent"),
                    "Accept-Language": request.headers.get("Accept-Language"),
                    "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
                }
                log.warning("Visitor identity missing for /api/chat. Request cookies: %s, headers: %s, remote_addr: %s",
                            dict(request.cookies), headers_snapshot, request.remote_addr)
            except Exception:
                log.exception("Failed to log missing identity details")
            return jsonify({"error": "Ziyaretçi kimliği alınamadı; lütfen sayfayı yenileyin.", "error_type": "technical"}), 401

    log.info(
        "CHAT REQUEST user=%s session=%s msg=%s",
        identity["subject_id"] if identity else "unknown",
        session_id,
        user_message[:200]
    )

    quota_ok, quota, quota_error = check_quota(identity)
    if not quota_ok:
        return jsonify({"error": quota_error, "error_type": "quota"}), 429

    quota = increment_usage(identity)

    if session_id not in extensions.conversation_sessions:
        extensions.conversation_sessions[session_id] = []

    history = extensions.conversation_sessions[session_id]

    # ── RAG: retrieve ONCE with local embedding ──────────────────────────────
    try:
        retrieved = extensions.vector_store.retrieve(user_message, top_k=CONFIG["retrieval_top_k"])
    except RuntimeError as e:
        error_msg = str(e)
        log.error("Embedding/retrieval failed: %s", error_msg)
        return jsonify({"error": "Şu anda çok yoğunuz. Lütfen biraz sonra tekrar dene.", "error_type": "retry"}), 503
    except Exception as e:
        log.exception("Unexpected retrieval error")
        return jsonify({"error": "Şu anda çok yoğunuz. Lütfen biraz sonra tekrar dene.", "error_type": "retry"}), 503

    context = format_context(retrieved, CONFIG["retrieval_score_threshold"])
    augmented_message = build_augmented_user_message(user_message, context)

    recent_history = history[-(CONFIG["max_history_turns"] * 2):]

    CONGESTION_THRESHOLD = CONFIG["congestion_threshold"]

    def generate():
        """
        Inner generator that drives the SSE stream.
        Intercepts the __full_response__ marker to persist history,
        then emits the final 'done' event with source metadata.
        """
        full_response = ""
        had_error = False
        saved_question_index = None

        with extensions.active_requests_lock:
            extensions.active_requests += 1
            current_active = extensions.active_requests
            log.info("CONGESTION active_requests=%s threshold=%s", current_active, CONGESTION_THRESHOLD)

        try:
            # Send congestion warning if threshold met or exceeded
            if current_active >= CONGESTION_THRESHOLD:
                yield f"data: {json.dumps({'congestion': True, 'active_requests': current_active})}\n\n"

            token_stream = extensions.llm_gateway.stream_chat(recent_history, augmented_message)

            for event in token_stream:
                if "__full_response__" in event:
                    try:
                        payload = json.loads(event.replace("data: ", "").strip())
                        full_response = payload.get("__full_response__", "")
                    except Exception:
                        pass
                    continue

                if '"error"' in event:
                    had_error = True

                yield event

        except Exception:
            log.exception("Unexpected error during stream generation")
        finally:
            with extensions.active_requests_lock:
                extensions.active_requests -= 1
                log.info("CONGESTION active_requests decremented to %s", extensions.active_requests)

        # ── Persist history (only on success) ────────────────────────────────
        if full_response and not had_error:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})
            if len(history) > CONFIG["max_history_turns"] * 2:
                extensions.conversation_sessions[session_id] = history[-(CONFIG["max_history_turns"] * 2):]
            try:
                with extensions.SessionLocal() as db:
                    last_index = db.query(func.max(ChatLog.question_index)).filter(
                        ChatLog.user_id == int(identity["subject_id"])
                    ).scalar()
                    saved_question_index = (last_index or 0) + 1
                    log_entry = ChatLog(
                        user_id=int(identity["subject_id"]),
                        question_index=saved_question_index,
                        question=user_message,
                        answer=full_response,
                        created_at=utc_now().isoformat(),
                    )
                    db.add(log_entry)
                    db.commit()
            except Exception:
                log.exception("Failed to save chat log for user_id=%s", identity["subject_id"])

        # ── Final event: sources ──────────────────────────────────────────────
        sources = build_sources_payload(retrieved, CONFIG["retrieval_score_threshold"])
        done_payload = {
            'done': True,
            'sources': sources,
            'near_limit': quota['daily_used'] >= 30,
        }
        if saved_question_index:
            done_payload['question_index'] = saved_question_index
        yield f"data: {json.dumps(done_payload)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Startup
# ═══════════════════════════════════════════════════════════════════════════════

def startup():
    """
    Runs once before the Flask server accepts requests.
    Loads the vector store and validates Groq configuration.
    Falls back to SQLite if PostgreSQL is unreachable.
    """
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
        extensions.reinit_engine(CONFIG["database_url"])

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
        extensions.embedding_model = SentenceTransformer(CONFIG["embedding_model"])
        log.info(
            f"✓ Embedding model loaded in {time.time() - t0:.1f}s — "
            f"dim={extensions.embedding_model.get_sentence_embedding_dimension()}"
        )
    except Exception as e:
        log.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

    # ── Load vector store (passing the pre-loaded embedding model) ────────────
    try:
        extensions.vector_store = VectorStore(
            CONFIG["faiss_index_file"],
            CONFIG["chunks_meta_file"],
            CONFIG["embedding_model"],
            embedding_model=extensions.embedding_model,
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

    extensions.llm_gateway = LLMGateway(CONFIG)
    log.info(f"LLM gateway ready — active provider: {extensions.llm_gateway.active_provider}")
    log.info(f"HF Space tuning — congestion_threshold={CONFIG['congestion_threshold']}, "
             f"max_workers={CONFIG['embedding_max_workers']}")
    port = int(os.getenv("PORT", "5000"))
    scheme = "https" if CONFIG["local_https"] and not os.getenv("PORT") else "http"
    log.info(f"Server starting on {scheme}://0.0.0.0:{port}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    startup()
    port = int(os.getenv("PORT", "7860"))
    ssl_context = "adhoc" if CONFIG["local_https"] and not os.getenv("PORT") else None
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )
