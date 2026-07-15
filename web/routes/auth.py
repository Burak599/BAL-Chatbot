"""
Authentication routes for BAL Chatbot.

This module handles:
- auth_status()
- unsupported_auth()
- auth_guest(), auth_register(), auth_login(), auth_google(), auth_logout()
"""

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG

try:
    from extensions import app
except ImportError:
    from web.extensions import app

try:
    from quota import quota_snapshot
except ImportError:
    from web.quota import quota_snapshot

try:
    from auth import (
        get_current_identity,
        normalize_email,
    )
except ImportError:
    from web.auth import (
        get_current_identity,
        normalize_email,
    )

from flask import jsonify


def unsupported_auth():
    return jsonify({"error": "Authentication flow is not supported for this app."}), 404


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