"""
LLM Gateway for BAL Chatbot.

This module handles:
- Groq API streaming
- LLMGateway class
- Reasoning block stripping
"""

import json
import logging
import re
from typing import Generator, Optional, Tuple, List, Dict

from curl_cffi import requests as curl_requests

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


def stream_groq_model(messages: List[Dict], model: str, api_key: str, key_index: int) -> Tuple[str, Optional[Dict]]:
    """
    Streams one Groq model attempt using curl_cffi to bypass Cloudflare 403 blocks.
    Returns (full_response, failure_info).
    """
    full_response = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": CONFIG["llm_temperature"],
        "max_tokens": CONFIG["llm_max_tokens"],
        "top_p": CONFIG["llm_top_p"],
    }

    try:
        log.info(
            "GROQ REQUEST -> model=%s key_index=%s url=%s",
            model,
            key_index,
            CONFIG["groq_url"]
        )
        resp = curl_requests.post(
            CONFIG["groq_url"],
            headers=headers,
            json=payload,
            stream=True,
            timeout=CONFIG["groq_timeout"],
            impersonate="chrome"
        )

        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            if isinstance(line, bytes):
                line_str = line.decode("utf-8")
            else:
                line_str = str(line)

            if not line_str.startswith("data: "):
                continue

            data_text = line_str[6:].strip()
            if data_text == "[DONE]":
                break

            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            delta = data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

    except curl_requests.errors.RequestsError as e:
        error_msg = str(e).lower()
        reason = "exception"

        if "timeout" in error_msg:
            reason = "timeout"
            return "Groq API zaman aşımına uğradı. Lütfen tekrar deneyin.", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }
        elif "connect" in error_msg or "resolve" in error_msg:
            reason = "connection"
            return "Groq API bağlantısı kurulamadı. Lütfen daha sonra tekrar deneyin.", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }

        status_code = 0
        response_text = ""
        rate_headers = {}

        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text
            log.error("FULL GROQ BODY:\n%s", response_text)
            rate_headers = {
                key: value
                for key, value in e.response.headers.items()
                if key.lower().startswith(("x-ratelimit", "retry-after", "x-request-id"))
            }
            reason = f"http_{status_code}"

        log.error(
            "Groq API HTTP error status=%s model=%s rate_headers=%s body=%s",
            status_code,
            model,
            rate_headers,
            response_text,
            exc_info=True,
        )

        if status_code > 0:
            retryable = status_code in {404, 429} or 500 <= status_code <= 599
            return f"Groq API hatası: HTTP {status_code}", {
                "retryable": retryable,
                "model": model,
                "key_index": key_index,
                "reason": reason,
                "status_code": status_code,
                "rate_headers": rate_headers,
            }
        else:
            return f"Groq API hatası: {str(e)}", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }

    except Exception as e:
        log.exception("Groq streaming error model=%s", model)
        return f"Groq API hatası: {str(e)}", {
            "retryable": True,
            "model": model,
            "key_index": key_index,
            "reason": "exception",
        }

    return full_response, None


def stream_groq(messages: List[Dict]) -> Generator[str, None, None]:
    """
    Streams tokens from Groq's OpenAI-compatible Chat Completions API.
    Tries the full model chain for one API key, then rotates to the next key
    and starts from the strongest model again.
    """
    api_keys = CONFIG["groq_api_keys"]
    model_chain = CONFIG["groq_model_chain"]
    last_error = "Groq API hatası."

    if not api_keys:
        yield f"data: {json.dumps({'error': 'GROQ_API_KEY ayarlı değil.'})}\n\n"
        return

    for key_index, api_key in enumerate(api_keys, 1):
        for model_index, model in enumerate(model_chain):
            response_text = ""
            failure_info = None

            attempt = stream_groq_model(messages, model, api_key, key_index)
            while True:
                try:
                    yield next(attempt)
                except StopIteration as stop:
                    if stop.value:
                        response_text, failure_info = stop.value
                    break

            if failure_info is None:
                if model_index > 0:
                    notice = {
                        "from_model": model_chain[0],
                        "to_model": model,
                        "message": "Yoğunluk nedeniyle model düşürüldü.",
                    }
                    yield f"data: {json.dumps({'model_fallback': notice})}\n\n"
                    log.warning(
                        "Groq fallback succeeded key_index=%s original_model=%s active_model=%s",
                        key_index,
                        model_chain[0],
                        model,
                    )
                yield f"data: {json.dumps({'__full_response__': strip_reasoning_blocks(response_text)})}\n\n"
                return

            last_error = response_text
            if not failure_info.get("retryable"):
                yield f"data: {json.dumps({'error': last_error})}\n\n"
                return

            if model_index < len(model_chain) - 1:
                next_model = model_chain[model_index + 1]
                log.warning(
                    "Groq fallback switching key_index=%s from_model=%s to_model=%s reason=%s",
                    key_index,
                    model,
                    next_model,
                    failure_info.get("reason"),
                )
                continue

            if key_index < len(api_keys):
                log.warning(
                    "Groq API key exhausted key_index=%s next_key_index=%s last_model=%s reason=%s",
                    key_index,
                    key_index + 1,
                    model,
                    failure_info.get("reason"),
                )
                break

    yield f"data: {json.dumps({'error': last_error})}\n\n"
    return


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