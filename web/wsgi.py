"""Production WSGI entry point for Render/Gunicorn."""

from web.app import app, startup


startup()

