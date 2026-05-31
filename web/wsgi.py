"""Production WSGI entry point for Render/Gunicorn."""

from app import app, startup


startup()

