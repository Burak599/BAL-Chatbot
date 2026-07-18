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
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import routes module to register all routes
from routes import pages, auth, health, chat


import logging
# ── Logging ───────────────────────────────────────────────────────────────────
try:
    from config import LOG_DIR
except ImportError:
    from web.config import LOG_DIR

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

try:
    from config import CONFIG
except ImportError:
    from web.config import CONFIG

try:
    import extensions
except ImportError:
    import web.extensions as extensions

try:
    from extensions import app
except ImportError:
    from web.extensions import app

from flask import request


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
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    try:
        from config import CONFIG
    except ImportError:
        from web.config import CONFIG
    
    from runtime import startup
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