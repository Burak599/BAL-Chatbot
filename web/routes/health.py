"""
Health check route for BAL Chatbot.

This module handles:
- health() - returns API status
"""

try:
    from extensions import app, database_ready
except ImportError:
    import web.extensions as extensions
    app = extensions.app
    database_ready = extensions.database_ready

try:
    from models import ChatLog
except ImportError:
    from web.models import ChatLog

from flask import jsonify


@app.route("/api/health", methods=["GET"])
def health():
    """
    Returns a JSON status object.
    """
    import web.extensions as ext
    
    status = {
        "provider": ext.CONFIG.get("provider", "groq"),
        "vectorstore": ext.vector_store is not None,
        "embedding_model": ext.CONFIG.get("embedding_model", "intfloat/multilingual-e5-small"),
        "database": database_ready(),
        "chunks": ext.vector_store.index.ntotal if ext.vector_store else 0,
    }

    if ext.llm_gateway is None:
        status.update({"status": "degraded", "provider": None})
        return jsonify(status)

    provider_status = ext.llm_gateway.status()
    status.update(provider_status)

    if not ext.vector_store or not status.get("database"):
        status["status"] = "degraded"

    return jsonify(status)