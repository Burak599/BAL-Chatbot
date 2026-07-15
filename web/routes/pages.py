"""
Page routes for BAL Chatbot.

This module handles:
- index() - serves frontend HTML
- serve_files() - serves static files
"""

try:
    from config import WEB_DIR
except ImportError:
    from web.config import WEB_DIR

try:
    from extensions import app
except ImportError:
    from web.extensions import app

from flask import send_from_directory


@app.route("/")
def index():
    """Serves the frontend HTML file."""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def serve_files(filename):
    return send_from_directory(WEB_DIR, filename)