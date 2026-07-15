"""
Routes module for BAL Chatbot.

This module registers all Flask routes.
"""

# Import all route modules to register them with the Flask app
from . import pages
from . import auth
from . import health
from . import chat

__all__ = ["pages", "auth", "health", "chat"]