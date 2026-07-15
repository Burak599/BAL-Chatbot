"""
LLM Gateway for BAL Chatbot.

This module handles:
- LLMGateway class
- Reasoning block stripping (for removing think blocks from model output)
"""

import json
import logging
import re
from typing import Generator, List, Dict

log = logging.getLogger(__name__)

try:
    from config import CONFIG, SYSTEM_PROMPT
except ImportError:
    from web.config import CONFIG, SYSTEM_PROMPT


def strip_reasoning_blocks(text: str) -> str:
    """Removes reasoning traces emitted by models that expose <think> blocks."""
    if not text:
        return text

    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think\b[^>]*>.*\Z", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking\b[^>]*>.*\Z", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


from .groq_client import stream_groq


class LLMGateway:
    """
    Single backend-facing entry point for all model calls.
    """

    def __init__(self, config: Dict):
        self.config = config

    @property
    def active_provider(self) -> str:
        return self.config["provider"]

    def status(self) -> Dict:
        """Returns provider readiness for /api/health."""
        return {
            "provider": self.active_provider,
            "groq": bool(self.config["groq_api_keys"]),
            "groq_key_count": len(self.config["groq_api_keys"]),
            "model_name": self.config["groq_model_chain"][0],
            "model_chain": self.config["groq_model_chain"],
            "status": "ok" if self.config["groq_api_keys"] else "degraded",
        }

    def stream_chat(
        self,
        recent_history: List[Dict],
        augmented_message: str,
    ) -> Generator[str, None, None]:
        """
        Routes one chat turn to Groq.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + recent_history
            + [{"role": "user", "content": augmented_message}]
        )
        yield from stream_groq(messages)