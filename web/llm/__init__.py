"""
LLM module for BAL Chatbot.

This module provides:
- LLMGateway: Single entry point for model calls
- stream_groq: Groq API streaming function
- stream_groq_model: Single model attempt streaming
- strip_reasoning_blocks: Removes reasoning traces from model output
"""

from .gateway import (
    LLMGateway,
    stream_groq,
    stream_groq_model,
    strip_reasoning_blocks,
)

__all__ = [
    "LLMGateway",
    "stream_groq",
    "stream_groq_model",
    "strip_reasoning_blocks",
]