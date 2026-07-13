"""
RAG (Retrieval-Augmented Generation) module for BAL Chatbot.

This module provides:
- VectorStore: FAISS-based vector database with local embedding support
- Prompting utilities for context formatting and source generation
"""

from .vectorstore import VectorStore
from .prompting import (
    format_context,
    build_augmented_user_message,
    build_sources_payload,
    strip_reasoning_blocks,
)

__all__ = [
    "VectorStore",
    "format_context",
    "build_augmented_user_message",
    "build_sources_payload",
    "strip_reasoning_blocks",
]
