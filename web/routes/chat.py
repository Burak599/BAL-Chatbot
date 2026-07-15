"""
Chat routes for BAL Chatbot.

This module handles:
- chat() - main chat endpoint with SSE streaming
- chat_feedback() - feedback submission endpoint
"""

import json
import re
from typing import Dict, List

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG

try:
    from extensions import app
except ImportError:
    from web.extensions import app

try:
    from auth import get_current_identity
except ImportError:
    from web.auth import get_current_identity

try:
    from quota import quota_snapshot, check_quota, increment_usage, utc_now
except ImportError:
    from web.quota import quota_snapshot, check_quota, increment_usage, utc_now

try:
    from models import ChatLog
except ImportError:
    from web.models import ChatLog

try:
    from rag import format_context, build_augmented_user_message, build_sources_payload
except ImportError:
    from web.rag import format_context, build_augmented_user_message, build_sources_payload

try:
    from llm import LLMGateway, strip_reasoning_blocks
except ImportError:
    from web.llm import LLMGateway, strip_reasoning_blocks

import logging
from flask import request, jsonify, Response, stream_with_context

log = logging.getLogger(__name__)


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
        import web.extensions as ext
        with ext.SessionLocal() as db:
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
    import web.extensions as extensions

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
                    from sqlalchemy import func
                    last_index = db.query(func.max(ChatLog.question_index)).filter(
                        ChatLog.user_id == int(identity["subject_id"])
                    ).scalar()
                    saved_question_index = (last_index or 0) + 1
                    log_entry = ChatLog(
                        user_id=int(identity["subject_id"]),
                        question_index=saved_question_index,
                        question=user_message,
                        answer=full_response,
                        created_at=utc_now(),
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